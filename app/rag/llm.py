from typing import List, Dict
import requests
import json
from app.core.config import get_settings

SYS_PROMPT = """You are a helpful research assistant. Use ONLY the provided context to answer.
If the answer is not in the context, say you don't know. Be concise and cite sources like [filename].
"""

def _messages(question: str, passages: List[Dict]):
    context = "\n\n---\n\n".join(
        f"Source: {p['source']} | Chunk: {p['chunk_id']}\n{p['text']}"
        for p in passages
    )
    user = f"Question: {question}\n\nContext:\n{context}\n\nInstructions: Cite sources like [filename]."
    return [
        {"role": "system", "content": SYS_PROMPT},
        {"role": "user", "content": user},
    ]

def _endpoint(base: str) -> str:
    base = base.rstrip("/")
    # DeepSeek expects /v1/chat/completions
    if not base.endswith("/v1"):
        base = base + "/v1"
    return base + "/chat/completions"

def generate_answer(question: str, passages: List[Dict]) -> Dict:
    s = get_settings()
    url = _endpoint(s.DEEPSEEK_BASE_URL)

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {s.DEEPSEEK_API_KEY or ''}",
    }

    payload = {
        "model": s.DEEPSEEK_MODEL,
        "messages": _messages(question, passages),
        "temperature": 0.2,
        "stream": False,
        # safe defaults; adjust freely
        "max_tokens": 512,
        "top_p": 0.95,
        "presence_penalty": 0,
        "frequency_penalty": 0,
    }

    try:
        resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=60)
    except Exception as e:
        raise RuntimeError(f"DeepSeek request failed to {url}. Error: {e}")

    if resp.status_code != 200:
        # Bubble up a clear message (401s show up here)
        raise RuntimeError(
            f"DeepSeek error {resp.status_code}: {resp.text}. "
            f"Check DEEPSEEK_API_KEY and DEEPSEEK_BASE_URL in .env (currently hitting {url})."
        )

    data = resp.json()
    text = data["choices"][0]["message"]["content"].strip()
    sources = sorted({p["source"] for p in passages})
    return {"answer": text, "sources": sources, "passages": passages}
