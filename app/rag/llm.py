from typing import List, Dict
import requests, json
from app.core.config import get_settings

SYS_PROMPT = """
You are a helpful assistant.
- Answer using the provided context.
- If the answer is not supported by the context, say "I don't know based on the provided documents."
- Be concise and structure the answer clearly.
- Cite inline using [filename p.X] where evidence appears.
"""

# threshold below which we choose to say "I don't know"
MIN_SIM_THRESHOLD = 0.1

def _build_context(passages: List[Dict]) -> str:
    blocks = []
    for p in passages:
        page = p.get("page")
        cite = f"[{p['source']} p.{page}]" if page is not None else f"[{p['source']}]"
        blocks.append(f"{cite}\n{p['text']}")
    return "\n\n---\n\n".join(blocks)

def _messages_scaffold(question: str, passages: List[Dict]):
    context = _build_context(passages)
    user = (
        f"Question: {question}\n\n"
        f"Context (each section starts with its citation):\n{context}\n\n"
        f"Instructions:\n"
        f"1) Provide a short direct answer first.\n"
        f"2) Then list 2–5 bullet points of key evidence with inline citations like [filename p.X].\n"
        f"3) If context is insufficient, answer: \"I don't know based on the provided documents.\""
    )
    return [
        {"role": "system", "content": SYS_PROMPT.strip()},
        {"role": "user", "content": user}
    ]

def _endpoint(base: str) -> str:
    base = base.rstrip("/")
    if not base.endswith("/v1"):
        base = base + "/v1"
    return base + "/chat/completions"

def _should_abstain(passages: List[Dict]) -> bool:
    if not passages:
        return True
    # if top passage score is very low, abstain
    top = max([p.get("score", 0.0) for p in passages]) if passages else 0.0
    return top < MIN_SIM_THRESHOLD

def generate_answer(question: str, passages: List[Dict]) -> Dict:
    s = get_settings()

    # fallback if retrieval is weak
    if _should_abstain(passages):
        return {
            "answer": "I don't know based on the provided documents.",
            "sources": [],
            "passages": passages
        }

    # Optional: compress evidence before final answer (single extra call)
    # Toggle by setting COMPRESS_BEFORE_ANSWER = True in config (optional)
    compress = getattr(s, "COMPRESS_BEFORE_ANSWER", False)
    final_passages = passages

    url = _endpoint(s.DEEPSEEK_BASE_URL)
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {s.DEEPSEEK_API_KEY or ''}"}

    if compress:
        # Make one call to compress passages into dense evidence bullets
        comp_messages = [
            {"role": "system", "content": "Summarize each passage into 1–2 evidence bullets with inline citation kept intact."},
            {"role": "user", "content": _build_context(passages)}
        ]
        comp_payload = {
            "model": s.DEEPSEEK_MODEL,
            "messages": comp_messages,
            "temperature": 0.0,
            "max_tokens": 400
        }
        comp_resp = requests.post(url, headers=headers, data=json.dumps(comp_payload), timeout=60)
        if comp_resp.status_code == 200:
            compressed = comp_resp.json()["choices"][0]["message"]["content"].strip()
            # use compressed bullets as the new context
            final_passages = [{"text": compressed, "source": "compressed", "chunk_id": "compressed", "page": None, "score": 1.0}]

    payload = {
        "model": s.DEEPSEEK_MODEL,
        "messages": _messages_scaffold(question, final_passages),
        "temperature": 0.2,
        "max_tokens": 800,
        "top_p": 0.95
    }

    resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=120)
    if resp.status_code != 200:
        raise RuntimeError(f"DeepSeek error {resp.status_code}: {resp.text}")

    text = resp.json()["choices"][0]["message"]["content"].strip()
    sources = sorted({p["source"] for p in passages if p.get("source")})
    return {"answer": text, "sources": sources, "passages": passages}
