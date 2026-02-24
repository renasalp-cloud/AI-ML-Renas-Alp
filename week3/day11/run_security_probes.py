import json
from pipeline import answer_question

INPUT_FILE = "security_probes.json"
OUTPUT_FILE = "security_probe_results_local.json"

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    probes = json.load(f)

results = []

for p in probes:
    out = answer_question(p["probe"])

    results.append({
        "id": p["id"],
        "category": p["category"],
        "probe": p["probe"],
        "answer": out.get("answer", ""),
        "model": out.get("model", ""),
        "retrieval_top_k": out.get("retrieval", {}).get("top_k"),
        "retrieved_sources": [
            (c.get("source"), c.get("page"))
            for c in out.get("retrieval", {}).get("retrieved_chunks", [])
        ],
    })

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"Saved {len(results)} probe results to {OUTPUT_FILE}")