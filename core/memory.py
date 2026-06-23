# core/memory.py — Redis episodic memory

import json
# pyrefly: ignore [missing-import]
import redis
from core.config import REDIS_HOST, REDIS_PORT, REDIS_DB, MEMORY_TTL, MAX_MEMORY_TURNS


def _get_client() -> redis.Redis | None:
    try:
        client = redis.Redis(
            host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB,
            decode_responses=True, socket_connect_timeout=2,
        )
        client.ping()
        return client
    except Exception:
        return None


def save_turn(session_id: str, query: str, answer: str) -> None:
    client = _get_client()
    if not client:
        print("[Memory] Redis not available — skipping.")
        return
    key = f"rag:memory:{session_id}"
    raw = client.get(key)
    turns = json.loads(raw) if raw else []
    turns.append({"query": query, "answer": answer})
    turns = turns[-MAX_MEMORY_TURNS:]
    client.setex(key, MEMORY_TTL, json.dumps(turns))
    print(f"[Memory] Saved. Session '{session_id}' → {len(turns)} turn(s).")


def get_context(session_id: str) -> str:
    client = _get_client()
    if not client:
        return ""
    raw = client.get(f"rag:memory:{session_id}")
    if not raw:
        return ""
    turns = json.loads(raw)
    if not turns:
        return ""
    lines = ["Past conversation:"]
    for i, t in enumerate(turns, 1):
        lines.append(f"[Turn {i}] User: {t['query']}")
        lines.append(f"[Turn {i}] Assistant: {t['answer'][:300]}...\n")
    print(f"[Memory] Loaded {len(turns)} turn(s) for '{session_id}'.")
    return "\n".join(lines)


def clear_session(session_id: str) -> None:
    """Clear all memory for a specific session"""
    client = _get_client()
    if client:
        client.delete(f"rag:memory:{session_id}")
        print(f"[Memory] Cleared session '{session_id}'.")