import json
import re

INPUT_FILE = "eval_results_local.json"

def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())

def soft_match(expected: str, actual: str) -> bool:
    e = normalize(expected)
    a = normalize(actual)

    # If expected says "cannot find", accept any variant that contains it
    if "cannot find" in e:
        return "cannot find" in a

    # If expected is very short (like "€500."), require it to appear in actual
    if len(e) <= 12:
        return e.strip(".") in a

    # Otherwise: require most key tokens to appear
    key_tokens = [t for t in re.findall(r"[a-z0-9%€]+", e) if len(t) > 2]

    if not key_tokens:
        return False

    hits = sum(1 for t in key_tokens if t in a)
    return hits / len(key_tokens) >= 0.6  # 60% token overlap

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    rows = json.load(f)

correct = 0
total = len(rows)

for r in rows:
    if soft_match(r["expected_answer"], r["actual_answer"]):
        correct += 1

print(f"Soft-match correct answers: {correct}/{total}")
print(f"Soft-match accuracy: {correct/total:.3f}")