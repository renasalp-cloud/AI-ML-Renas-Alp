import json
import os
import time
from dotenv import load_dotenv

from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualRelevancyMetric,
    ContextualRecallMetric,
    ContextualPrecisionMetric,
)
from deepeval.models import AzureOpenAIModel
from deepeval.test_case import LLMTestCase

load_dotenv()

INPUT_FILE = "pipeline_outputs_cloud.json"
OUTPUT_FILE = "eval_results_cloud.json"

api_key = os.getenv("AZURE_OPENAI_API_KEY", "")
base_url = os.getenv("AZURE_OPENAI_BASE_URL", "")
deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "")
api_version = os.getenv("AZURE_OPENAI_API_VERSION", "")

if not api_key:
    raise ValueError("AZURE_OPENAI_API_KEY is missing. Check your .env file.")

judge = AzureOpenAIModel(
    model="gpt-4.1-mini",
    deployment_name=deployment,
    api_key=api_key,
    api_version=api_version,
    base_url=base_url,
)

metrics = [
    FaithfulnessMetric(threshold=0.7, model=judge, include_reason=True),
    AnswerRelevancyMetric(threshold=0.7, model=judge, include_reason=True),
    ContextualRelevancyMetric(threshold=0.7, model=judge, include_reason=True),
    ContextualRecallMetric(threshold=0.7, model=judge, include_reason=True),
    ContextualPrecisionMetric(threshold=0.7, model=judge, include_reason=True),
]

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    rows = json.load(f)

results = []

for r in rows:
    tc = LLMTestCase(
        input=r["question"],
        actual_output=r["actual_answer"],
        expected_output=r["expected_answer"],
        retrieval_context=r["retrieved_context"],
    )

    metric_scores = {}

    for m in metrics:
        while True:
            try:
                m.measure(tc)
                break
            except Exception as e:
                msg = str(e)
                if "429" in msg or "RateLimit" in msg or "RateLimitReached" in msg:
                    print("Rate limit hit. Sleeping 35 seconds...")
                    time.sleep(35)
                else:
                    raise

        time.sleep(2.5)  # throttle to avoid Azure rate limit

        metric_scores[m.__class__.__name__] = {
            "score": m.score,
            "threshold": m.threshold,
            "passed": (m.score is not None and m.score >= m.threshold),
            "reason": m.reason,
        }

    results.append({
        "id": r.get("id"),
        "question": r["question"],
        "expected_answer": r["expected_answer"],
        "actual_answer": r["actual_answer"],
        "metrics": metric_scores,
    })

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"Saved {len(results)} eval rows to {OUTPUT_FILE}")