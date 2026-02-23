import json
import time
from pathlib import Path

from pipeline import answer_question


TEST_FILE = "test_questions.json"
OUTPUT_FILE = "benchmark_results.json"


def load_questions(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def simple_correctness_check(expected, actual):
    if not expected or not actual:
        return False

    expected = expected.lower().strip()
    actual = actual.lower().strip()

    return expected in actual


def run_benchmark():
    questions = load_questions(TEST_FILE)

    results = []
    total_start = time.time()

    for q in questions:
        print(f"Running question {q['id']}...")

        start = time.time()
        output = answer_question(q["question"])
        end = time.time()

        total_time_ms = int((end - start) * 1000)

        correctness = simple_correctness_check(
            q["expected_answer"],
            output["answer"]
        )

        result_entry = {
            "id": q["id"],
            "difficulty": q["difficulty"],
            "question": q["question"],
            "expected_answer": q["expected_answer"],
            "model_answer": output["answer"],
            "correct": correctness,
            "retrieval_time_ms": output["retrieval"]["retrieval_time_ms"],
            "generation_time_ms": output["generation_time_ms"],
            "total_time_ms": total_time_ms
        }

        results.append(result_entry)

    total_elapsed = int((time.time() - total_start) * 1000)

    summary = {
        "total_questions": len(results),
        "total_time_ms": total_elapsed,
        "correct_count": sum(r["correct"] for r in results),
        "accuracy": round(
            sum(r["correct"] for r in results) / len(results), 2
        ),
        "results": results
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nBenchmark complete.")
    print(f"Accuracy: {summary['accuracy']}")
    print(f"Results saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    run_benchmark()