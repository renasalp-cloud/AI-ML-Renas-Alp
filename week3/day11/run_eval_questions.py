from pipeline import answer_question
import json

with open("test_questions.json", "r", encoding="utf-8") as f:
    test_questions = json.load(f)

results = []

for q in test_questions:
    output = answer_question(q["question"])

    results.append({
        "id": q["id"],
        "question": q["question"],
        "expected_answer": q["expected_answer"],
        "actual_answer": output.get("answer", ""),
        "retrieved_context": [c.get("content", "") for c in output["retrieval"]["retrieved_chunks"]],
        "model": output.get("model"),
        "retrieval_time_ms": output["retrieval"].get("retrieval_time_ms"),
        "generation_time_ms": output.get("generation_time_ms"),
    })

with open("pipeline_outputs_cloud.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"Generated outputs for {len(results)} questions")