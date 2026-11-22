import json
import re

from model_v2 import clean_output



# Kasıtlı olarak bozulmuş "sahte model cevapları"
bozuk_ornekler = {
    "zaten temiz": '{"sentiment": "positive"}',
    "markdown blok": '```json\n{"sentiment": "positive"}\n```',
    "giriş cümlesi": 'Here is your JSON:\n{"sentiment": "positive"}',
    "giriş + markdown + kapanış": 'Sure!\n```json\n{"sentiment": "positive"}\n```\nHope this helps!',
    "JSON hiç yok (kurtarılamaz)": "Sorry, I cannot analyze this review.",
}

print(f"{'Örnek':32} {'first-try':12} {'cleaning sonrası'}")
print("-" * 62)
for ad, raw in bozuk_ornekler.items():
    try:
        json.loads(raw)
        ilk = "OK"
    except json.JSONDecodeError:
        ilk = "HATA"
    try:
        json.loads(clean_output(raw))
        temiz = "OK"
    except json.JSONDecodeError:
        temiz = "HATA"
    print(f"{ad:32} {ilk:12} {temiz}")