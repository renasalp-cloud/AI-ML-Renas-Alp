import time
from typing import List, Dict, Any, Optional, Union

from openai import OpenAI

from config import (
    LLM_BASE_URL,
    LLM_API_KEY,
    LLM_MODEL,
    SYSTEM_PROMPT,
    MAX_TOKENS,
    TEMPERATURE,
    MAX_CONTEXT_CHUNKS,
    MAX_CHUNK_CHARS,
)


def _clean_text(s: str) -> str:
    # Keep it simple: remove weird control chars that can confuse some local models
    return "".join(ch if ch.isprintable() else " " for ch in s)


def build_context(chunks: List[Dict[str, Any]]) -> str:
    limited = chunks[:MAX_CONTEXT_CHUNKS]

    parts = []
    for i, ch in enumerate(limited, start=1):
        src = ch.get("source")
        page = ch.get("page")
        text = ch.get("content", "")

        if len(text) > MAX_CHUNK_CHARS:
            text = text[:MAX_CHUNK_CHARS] + "..."

        parts.append(f"[{i}] Source: {src}, Page: {page}")
        parts.append(_clean_text(text))
        parts.append("")

    return "\n".join(parts).strip()


def _extract_text_from_message(msg: Any) -> str:
    """
    Ollama/OpenAI-compatible responses can vary:
    - msg.content can be a string
    - msg.content can be a list of content parts
    - some models may return text in other attributes
    """
    content = getattr(msg, "content", "")

    # If it's a list of parts, join text parts
    if isinstance(content, list):
        texts = []
        for part in content:
            if isinstance(part, dict):
                if part.get("type") == "text" and isinstance(part.get("text"), str):
                    texts.append(part["text"])
            elif isinstance(part, str):
                texts.append(part)
        content = "\n".join(texts)

    if isinstance(content, str) and content.strip():
        return content.strip()

    # Try other common fields (best-effort)
    for attr in ["reasoning", "reasoning_content", "thinking", "analysis"]:
        val = getattr(msg, attr, None)
        if isinstance(val, str) and val.strip():
            return val.strip()

    return ""


def call_llm(
    client: OpenAI,
    query: str,
    system_prompt: str,
) -> tuple[str, Optional[int]]:
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ],
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
    )

    msg = resp.choices[0].message
    answer = _extract_text_from_message(msg)

    tokens_generated = None
    if getattr(resp, "usage", None) and resp.usage and resp.usage.completion_tokens is not None:
        tokens_generated = resp.usage.completion_tokens

    return answer, tokens_generated


def generate_answer(query: str, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)

    # Normal attempt
    context = build_context(chunks)
    prompt = SYSTEM_PROMPT.format(context=context)

    t0 = time.time()
    answer, tokens_generated = call_llm(client, query, prompt)
    elapsed_ms = int((time.time() - t0) * 1000)

    # Retry with simpler prompt + only top chunk
    if not answer.strip():
        short_context = build_context(chunks[:1])
        simple_prompt = (
            "You are a helpful assistant.\n"
            "Answer using ONLY the context.\n"
            'If the answer is not in the context, say: "I cannot find this information in the provided documents."\n'
            "Provide FINAL ANSWER ONLY (no extra commentary).\n\n"
            f"Context:\n{short_context}\n"
        )

        t1 = time.time()
        answer_retry, tokens_retry = call_llm(client, query, simple_prompt)
        elapsed_ms = int((time.time() - t1) * 1000)

        if answer_retry.strip():
            answer = answer_retry
            tokens_generated = tokens_retry
            context = short_context

    # Final fallback
    if not answer.strip():
        answer = "I cannot find this information in the provided documents."

    return {
        "query": query,
        "answer": answer,
        "context_used": context,
        "model": LLM_MODEL,
        "generation_time_ms": elapsed_ms,
        "tokens_generated": tokens_generated,
    }


if __name__ == "__main__":
    # Quick sanity test
    sample_chunks = [
        {
            "content": "Current policy allows up to 3 days remote per week with Tuesday and Thursday as mandatory in-office days.",
            "source": "documents/meeting_notes.pdf",
            "page": 5,
            "relevance_score": 0.12,
        }
    ]
    q = "What is the maximum number of remote work days per week?"
    out = generate_answer(q, sample_chunks)
    print(out["answer"])