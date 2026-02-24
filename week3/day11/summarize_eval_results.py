import json
from statistics import mean

INPUT_FILE = "eval_results_local.json"

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    rows = json.load(f)

metric_names = [
    "FaithfulnessMetric",
    "AnswerRelevancyMetric",
    "ContextualRelevancyMetric",
    "ContextualRecallMetric",
    "ContextualPrecisionMetric",
]

# Collect scores per metric
scores = {m: [] for m in metric_names}

for r in rows:
    for m in metric_names:
        s = r["metrics"][m]["score"]
        if s is not None:
            scores[m].append(s)

print("=== Average Scores ===")
for m in metric_names:
    avg = mean(scores[m]) if scores[m] else 0.0
    print(f"{m}: {avg:.3f}")

# Find worst questions by each metric
print("\n=== Worst 3 Questions Per Metric ===")
for m in metric_names:
    sorted_rows = sorted(rows, key=lambda x: x["metrics"][m]["score"])
    print(f"\n{m}:")
    for r in sorted_rows[:3]:
        print(f"- id={r['id']} score={r['metrics'][m]['score']:.3f} | {r['question']}")