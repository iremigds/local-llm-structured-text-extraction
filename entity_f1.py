import json
import re
import pandas as pd

M = "llama3.2_3b"


ANNOTATED_IDXS = list(range(100))


def norm(s):
    s = s.lower().replace("’", "'")
    s = re.sub(r"'s\b", "", s)               # iyelik ekini at
    s = re.sub(r"[^\w\s\-]", " ", s)         # noktalama at
    return " ".join(s.split())


def load_gold(path="annotate.txt"):
    text = open(path, encoding="utf-8").read()
    parts = re.split(r"=== idx (\d+) ===", text)
    gold = {}
    for k in range(1, len(parts), 2):
        idx, body = int(parts[k]), parts[k + 1]
        m = re.search(r"^GOLD:(.*)$", body, flags=re.M)
        names = [norm(x) for x in m.group(1).split(";")] if m else []
        gold[idx] = {n for n in names if n}
    return gold


def load_pred(setting, temp):
    df = pd.read_csv(f"results_{M}_{setting}_t{temp}.csv")
    preds = {}
    for _, r in df.iterrows():
        ents = []
        try:
            d = json.loads(r["raw"])
            ents = d.get("entities", []) if isinstance(d, dict) else []
            ents = [e for e in ents if isinstance(e, str)]
        except Exception:
            pass
        preds[int(r["idx"])] = {norm(e) for e in ents if norm(e)}
    return preds


def partial(a, b):
    ta, tb = set(a.split()), set(b.split())
    return ta <= tb or tb <= ta               # biri diğerinin parçası mı


def score(gold, pred, idxs, mode):
    tp = fp = fn = 0
    for i in idxs:
        g, p = set(gold.get(i, set())), set(pred.get(i, set()))
        if mode == "exact":
            m = len(g & p)
        else:
            free, m = set(g), 0
            for x in sorted(p):
                hit = next((y for y in sorted(free) if partial(x, y)), None)
                if hit is not None:
                    free.remove(hit)
                    m += 1
        tp += m
        fp += len(p) - m
        fn += len(g) - m
    P = tp / (tp + fp) if tp + fp else 0.0
    R = tp / (tp + fn) if tp + fn else 0.0
    F = 2 * P * R / (P + R) if P + R else 0.0
    return P, R, F

if __name__ == "__main__":
    gold = load_gold()
    print(f"Değerlendirilen yorum sayısı: {len(ANNOTATED_IDXS)}")
    print(f"Gold'daki toplam isim sayısı: {sum(len(gold.get(i, set())) for i in ANNOTATED_IDXS)}\n")

    rows = []
    for s in ["prompt_only", "schema"]:
        for t in ["0", "1.0"]:
            pred = load_pred(s, t)
            row = {"setting": s, "temp": t}
            for mode in ["exact", "partial"]:
                P, R, F = score(gold, pred, ANNOTATED_IDXS, mode)
                row[f"{mode}_P"], row[f"{mode}_R"], row[f"{mode}_F1"] = P, R, F
            rows.append(row)

    print(pd.DataFrame(rows).round(3).to_string(index=False))


    print("\nTeşhis (prompt_only, temp 0)")
    pred = load_pred("prompt_only", "0")
    with_gold = [i for i in ANNOTATED_IDXS if gold.get(i)]
    missed_all = [i for i in with_gold if not pred.get(i)]
    print("En az 1 gold ismi olan yorum sayısı:", len(with_gold))
    print("Bunların kaçında model hiç isim vermedi:", len(missed_all))
    fp_list = [(i, sorted(pred[i] - gold.get(i, set())))
            for i in ANNOTATED_IDXS if pred.get(i, set()) - gold.get(i, set())]
    print("Modelin yazdığı ama gold'da olmayan isimler:", fp_list)