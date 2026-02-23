import json
from typing import Dict, Any

from retrieve import retrieve_chunks
from generate import generate_answer


def answer_question(query: str) -> Dict[str, Any]:
    retrieval_result = retrieve_chunks(query)
    generation_result = generate_answer(
        query=query,
        chunks=retrieval_result["retrieved_chunks"],
    )

    return {
        **generation_result,
        "retrieval": retrieval_result,
    }


if __name__ == "__main__":
    q = "What is the maximum number of remote work days per week?"
    out = answer_question(q)
    print(json.dumps(out, indent=2))