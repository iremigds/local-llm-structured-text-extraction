import json
import re
import time

import ollama
import pandas as pd
from datasets import load_dataset

# ---------------- Ayarlar ----------------
MODEL = "llama3.2:3b"
N_SAMPLES = 100         # test için 10, final koşu için 100
NUM_CTX = 4096
SEED = 42
TEMPERATURES = [0, 1.0]
SETTINGS = ["prompt_only", "schema"]

ALLOWED_CATEGORIES = ["drama", "comedy", "action", "horror", "sci-fi", "romance", "other"]
ALLOWED_SENTIMENTS = ["positive", "negative"]

SCHEMA = {
    "type": "object",
    "properties": {
        "category": {"type": "string", "enum": ALLOWED_CATEGORIES},
        "sentiment": {"type": "string", "enum": ALLOWED_SENTIMENTS},
        "sentiment_score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "entities": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["category", "sentiment", "sentiment_score", "entities"],
}

SYSTEM_PROMPT = """You are a text analysis assistant.
Return ONLY a JSON object with exactly these fields:
{
  "category": one of ["drama","comedy","action","horror","sci-fi","romance","other"],
  "sentiment": one of ["positive","negative"],
  "sentiment_score": a float between 0.0 and 1.0 (0.0 = very negative, 1.0 = very positive),
  "entities": a list of PERSON names mentioned in the review (use [] if there are none)
}
Do not add explanations or markdown."""


# ---------------- Yardımcı fonksiyonlar ----------------
def clean_output(raw: str) -> str:
    """Strip markdown fences and conversational text; keep the first {...} block."""
    raw = re.sub(r"```(?:json)?", "", raw).strip()
    start, end = raw.find("{"), raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        return raw[start:end + 1]
    return raw


def structural_valid(d) -> bool:
    """Hocanın tanımı: zorunlu alanlar var ve tipleri doğru."""
    if not isinstance(d, dict):
        return False
    try:
        return (
            isinstance(d["category"], str)
            and isinstance(d["sentiment"], str)
            and isinstance(d["sentiment_score"], (int, float))
            and not isinstance(d["sentiment_score"], bool)
            and isinstance(d["entities"], list)
            and all(isinstance(e, str) for e in d["entities"])
        )
    except KeyError:
        return False


def value_valid(d) -> bool:
    """Ek kontrol: izin verilen değerler ve skor aralığı."""
    return (
        structural_valid(d)
        and d["category"] in ALLOWED_CATEGORIES
        and d["sentiment"] in ALLOWED_SENTIMENTS
        and 0.0 <= d["sentiment_score"] <= 1.0
    )


def analyze_review(review: str, setting: str, temperature: float) -> dict:
    result = {"raw": None, "first_try_ok": False, "cleaned_ok": False,
              "struct_ok": False, "value_ok": False, "parsed": None,
              "error": None, "latency": None}
    kwargs = dict(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Analyze this movie review:\n{review[:1500]}"},
        ],
        options={"temperature": temperature, "seed": SEED, "num_ctx": NUM_CTX},
    )
    if setting == "schema":
        kwargs["format"] = SCHEMA      # prompt_only'de format parametresi HİÇ verilmez

    t0 = time.time()
    try:
        raw = ollama.chat(**kwargs)["message"]["content"]
        result["raw"] = raw
    except Exception as e:
        result["error"] = f"ollama_error: {e}"
        result["latency"] = time.time() - t0
        return result
    result["latency"] = time.time() - t0

    try:                                   # 1) first try: temizlik yok
        result["parsed"] = json.loads(raw)
        result["first_try_ok"] = True
    except json.JSONDecodeError:
        try:                               # 2) temizleyip tekrar dene
            result["parsed"] = json.loads(clean_output(raw))
            result["cleaned_ok"] = True
        except json.JSONDecodeError as e:
            result["error"] = f"parse_error: {e}"
            return result

    result["struct_ok"] = structural_valid(result["parsed"])
    result["value_ok"] = value_valid(result["parsed"])
    return result


def load_samples():
    ds = load_dataset("stanfordnlp/imdb", split="test").shuffle(seed=42)
    pos = [x for x in ds if x["label"] == 1][: N_SAMPLES // 2]
    neg = [x for x in ds if x["label"] == 0][: N_SAMPLES // 2]
    return pos + neg


# ---------------- Deney ----------------
if __name__ == "__main__":
    samples = load_samples()
    label_map = {0: "negative", 1: "positive"}

    # warm-up: modeli belleğe yükle
    ollama.chat(model=MODEL, messages=[{"role": "user", "content": "hi"}],
                options={"num_ctx": NUM_CTX})

    summary = []
    for setting in SETTINGS:
        for temp in TEMPERATURES:
            rows = []
            for i, s in enumerate(samples):
                r = analyze_review(s["text"], setting, temp)
                r["idx"] = i
                r["true"] = label_map[s["label"]]
                r["pred"] = r["parsed"].get("sentiment") if isinstance(r["parsed"], dict) else None
                rows.append(r)
            df = pd.DataFrame(rows)
            ok = df["first_try_ok"] | df["cleaned_ok"]
            summary.append({
                "setting": setting, "temp": temp,
                "raw_parse_%": 100 * df["first_try_ok"].mean(),
                "after_clean_%": 100 * ok.mean(),
                "struct_valid_%": 100 * df["struct_ok"].mean(),
                "value_valid_%": 100 * df["value_ok"].mean(),
                "accuracy_%": 100 * (df["pred"] == df["true"]).mean(),
                "latency_s": df["latency"].mean(),
            })
            name = f"results_{MODEL.replace(':', '_')}_{setting}_t{temp}.csv"
            df.to_csv(name, index=False)
            print(f"bitti: {name}")

    out = pd.DataFrame(summary).round(2)
    print(out.to_string(index=False))
    out.to_csv(f"summary_{MODEL.replace(':', '_')}.csv", index=False)