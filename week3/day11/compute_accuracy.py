import json

INPUT_FILE = "eval_results_local.json"

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    rows = json.load(f)

correct = 0
total = len(rows)

for r in rows:
    expected = r["expected_answer"].strip().lower()
    actual = r["actual_answer"].strip().lower()

    if expected == actual:
        correct += 1

accuracy = correct / total

print(f"Correct answers: {correct}/{total}")
print(f"Exact match accuracy: {accuracy:.3f}")