import ollama

from model_v2 import MODEL, NUM_CTX, SETTINGS, TEMPERATURES, analyze_review, load_samples

DEMO_IDX = [4, 6, 49, 50, 80]    # veri setinden 5 yorum

NEW_SAMPLES = [
    "A warm, funny and beautifully shot movie. Tom Hanks has never been better.",
    "The plot made no sense, the acting was wooden and I walked out after an hour.",
    "Oh great, another three-hour masterpiece that cured my insomnia. Truly unforgettable.",
    "The visuals were stunning, but the story dragged and the ending felt rushed.",
    "The film follows Maria and her brother Luca as they travel from Rome to Naples.",
]


def show(tag, r, true=None):
    p = r["parsed"] if isinstance(r["parsed"], dict) else {}
    parse = "OK" if r["first_try_ok"] else "FAIL"
    t = f" true={true}" if true else ""
    print(f"{tag:12} parse={parse} valid={r['value_ok']}{t} | "
          f"{p.get('sentiment')} {p.get('sentiment_score')} {p.get('entities')}")


print(f"Model: {MODEL} | num_ctx={NUM_CTX}")
samples = load_samples()
ollama.chat(model=MODEL, messages=[{"role": "user", "content": "hi"}],
            options={"num_ctx": NUM_CTX})          # warm-up

print("\n=== PART 1: dataset samples, 2 settings x 2 temperatures ===")
for temp in TEMPERATURES:
    for setting in SETTINGS:
        print(f"\n--- {setting}, temperature={temp} ---")
        for i in DEMO_IDX:
            r = analyze_review(samples[i]["text"], setting, temp)
            show(f"idx {i}", r, "positive" if samples[i]["label"] == 1 else "negative")

print("\n=== PART 2: new samples (not in the dataset), temperature=0 ===")
for text in NEW_SAMPLES:
    print("\nTEXT:", text)
    for setting in SETTINGS:
        show(setting, analyze_review(text, setting, 0))