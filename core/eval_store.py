import json
from pathlib import Path
from datetime import datetime

EVAL_LOG_PATH = Path("data/eval_log.json")


def save_eval(
    query:              str,
    final_answer:       str,
    agents_used:        list,
    faithfulness:       float,
    answer_relevance:   float,
    completeness:       float,
    hallucination_free: float,
    overall:            float,
    verdict:            str,
    critic_score:       float,
    retry_count:        int,
    avg_confidence:     float,
    latency_ms:         float = 0.0,
) -> None:
    """
    Append one evaluation record to the JSON log.
    Called after every query from the Streamlit dashboard.
    """
    EVAL_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Load existing records
    records = []
    if EVAL_LOG_PATH.exists():
        try:
            records = json.loads(EVAL_LOG_PATH.read_text())
        except Exception:
            records = []

    # Add new record
    records.append({
        "timestamp":          datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "query":              query[:80],   # truncate for display
        "agents_used":        agents_used,
        "faithfulness":       round(faithfulness, 3),
        "answer_relevance":   round(answer_relevance, 3),
        "completeness":       round(completeness, 3),
        "hallucination_free": round(hallucination_free, 3),
        "overall":            round(overall, 3),
        "verdict":            verdict,
        "critic_score":       round(critic_score, 3),
        "retry_count":        retry_count,
        "avg_confidence":     round(avg_confidence, 3),
        "latency_ms":         round(latency_ms, 1),
        "hallucination_rate": round(1 - hallucination_free, 3),
    })

    EVAL_LOG_PATH.write_text(json.dumps(records, indent=2))
    print(f"[EvalStore] Saved eval record #{len(records)}")


def load_evals() -> list[dict]:
    """Load all evaluation records for dashboard display."""
    if not EVAL_LOG_PATH.exists():
        return []
    try:
        return json.loads(EVAL_LOG_PATH.read_text())
    except Exception:
        return []


def clear_evals() -> None:
    """Clear all eval history (reset button in dashboard)."""
    if EVAL_LOG_PATH.exists():
        EVAL_LOG_PATH.write_text("[]")