import json
import pandas as pd
from sklearn.metrics import f1_score, confusion_matrix

M = "llama3.2_3b"
ALLOWED_CAT = ["drama", "comedy", "action", "horror", "sci-fi", "romance", "other"]
ALLOWED_SENT = ["positive", "negative"]
SETTINGS = ["prompt_only", "schema"]
TEMPS = ["0", "1.0"]

dfs = {(s, t): pd.read_csv(f"results_{M}_{s}_t{t}.csv") for s in SETTINGS for t in TEMPS}


def reasons(raw):
    d = json.loads(raw)
    out = []
    if d.get("category") not in ALLOWED_CAT:
        out.append(f"category={d.get('category')}")
    if d.get("sentiment") not in ALLOWED_SENT:
        out.append(f"sentiment={d.get('sentiment')}")
    sc = d.get("sentiment_score")
    if isinstance(sc, (int, float)) and not 0.0 <= sc <= 1.0:
        out.append(f"score={sc}")
    return ", ".join(out)


print("1) value_valid olmayan cevaplar")
for (s, t), df in dfs.items():
    bad = df[df["value_ok"] == False]
    for _, r in bad.iterrows():
        print(f"  {s} t={t} idx={r['idx']} -> {reasons(r['raw'])}")

print("\n2) Macro F1 ve confusion matrix (satırlar: gerçek, sütunlar: tahmin)")
for (s, t), df in dfs.items():
    f1 = f1_score(df["true"], df["pred"], average="macro", labels=ALLOWED_SENT)
    cm = confusion_matrix(df["true"], df["pred"], labels=ALLOWED_SENT)
    print(f"  {s} t={t}: macro F1={f1:.3f}  [neg->neg,neg->pos | pos->neg,pos->pos] = {cm.flatten().tolist()}")

print("\n3) Ayarlar arası uyum (aynı temperature)")
for t in TEMPS:
    a, b = dfs[("prompt_only", t)], dfs[("schema", t)]
    same_pred = (a["pred"] == b["pred"]).sum()
    same_raw = (a["raw"] == b["raw"]).sum()
    print(f"  t={t}: aynı tahmin {same_pred}/100, birebir aynı ham cevap {same_raw}/100")

print("\n4) Aynı ayarda temperature 0 vs 1.0 (aynı tahmin sayısı)")
for s in SETTINGS:
    a, b = dfs[(s, "0")], dfs[(s, "1.0")]
    print(f"  {s}: aynı tahmin {(a['pred'] == b['pred']).sum()}/100")

print("\n5) Entities istatistikleri")
for (s, t), df in dfs.items():
    ents = [json.loads(r)["entities"] for r in df["raw"] if isinstance(r, str)]
    ents = [e for e in ents if isinstance(e, list)]
    empty = sum(1 for e in ents if len(e) == 0)
    junk = sum(1 for e in ents if any(str(x).strip().lower() in {"none", "n/a", "null", ""} for x in e))
    print(f"  {s} t={t}: boş liste {empty}, 'None' benzeri {junk}, ort. eleman {sum(len(e) for e in ents)/len(ents):.2f}")