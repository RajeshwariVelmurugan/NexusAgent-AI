# core/complexity.py
# Aggressive Complexity Scorer - Ensures full pipeline for business queries

import re
from core.config import COMPLEXITY_THRESHOLD


def score_complexity(query: str, has_memory: bool = False) -> float:
    """
    Aggressive complexity scorer - ensures full pipeline for business queries
    """
    q = query.lower().strip()
    score = 0.30  # Baseline injection - CRITICAL for business queries!
    
    # ── Target Enterprise Heavy Keywords ───────────────────────
    heavy_keywords = [
        "sales", "q3", "q2", "q4", "q1", "quarter", "revenue", 
        "policy", "trend", "compare", "product", "market", 
        "2024", "2025", "2026", "limit", "expense", "customer",
        "segment", "invest", "industry", "analysis", "report",
        "multi-agent", "system", "best practices", "ethical",
        "database", "query", "sql", "postgresql", "table",
        "customers", "orders", "inventory", "supply", "chain"
    ]
    
    # If ANY corporate keyword exists, push score high
    if any(word in q for word in heavy_keywords):
        score += 0.45  # Instant push to 0.75!
    
    # ── Multi-part questions ───────────────────────────────────
    if any(w in q for w in [" and ", " also ", " as well", " additionally"]):
        score += 0.25
    
    # ── Comparison / analysis ──────────────────────────────────
    if any(w in q for w in ["vs", "versus", "compare", "difference", "better"]):
        score += 0.20
    
    # ── Trend / reasoning ──────────────────────────────────────
    if any(w in q for w in ["why", "how", "trend", "analysis", "explain", "impact"]):
        score += 0.20
    
    # ── Specific numbers, dates ────────────────────────────────
    if re.search(r'\d+', q):
        score += 0.10
    
    # ── Length booster ─────────────────────────────────────────
    word_count = len(q.split())
    if word_count > 5:
        score += 0.25
    if word_count > 10:
        score += 0.15
    if word_count > 15:
        score += 0.10
    
    # ── Multi-source need ──────────────────────────────────────
    has_data = any(w in q for w in ["sales", "revenue", "customer", "order", "database"])
    has_doc = any(w in q for w in ["report", "document", "contract", "policy", "file"])
    has_web = any(w in q for w in ["latest", "current", "news", "trend", "today"])
    source_count = sum([has_data, has_doc, has_web])
    if source_count >= 2:
        score += 0.20
    
    # ── Memory context boost ───────────────────────────────────
    if has_memory:
        score += 0.20
    
    # Clamp to [0.0, 1.0]
    score = round(max(0.0, min(1.0, score)), 3)
    
    label = "COMPLEX" if score >= COMPLEXITY_THRESHOLD else "SIMPLE"
    print(f"[Complexity] Score: {score:.2f} → {label} | '{query[:50]}'")
    return score


def is_complex(query: str, has_memory: bool = False) -> bool:
    """Returns True if query needs full reflection pipeline."""
    return score_complexity(query, has_memory) >= COMPLEXITY_THRESHOLD