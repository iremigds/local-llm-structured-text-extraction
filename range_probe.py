import json
import ollama

MODEL = "llama3.2:3b"

SCHEMA = {
    "type": "object",
    "properties": {
        "sentiment": {"type": "string", "enum": ["positive", "negative"]},
        "sentiment_score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
    },
    "required": ["sentiment", "sentiment_score"],
}

# Kasıtlı çelişki: prompt -1..1 istiyor, şema 0..1 diyor
SYSTEM_PROMPT = (
    "Analyze the movie review. sentiment_score must be between -1.0 "
    "(very negative) and 1.0 (very positive)."
)

reviews = [
    "This was the worst movie I have ever seen. Terrible acting and a boring plot.",
    "I hated every minute of it. A complete waste of time.",
    "Awful, painful to watch. I want my money back.",
]

for review in reviews:
    raw = ollama.chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": review},
        ],
        format=SCHEMA,
        options={"temperature": 0, "seed": 42, "num_ctx": 4096},
    )["message"]["content"]
    print(raw)