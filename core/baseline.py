# core/baseline.py — FIXED VERSION
# Issue was: judge_answer returns "overall_score" key
# but SystemResultOut expects "overall" key
# Fix: normalize all keys in run_comparison()

import os
import time
import json
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

JUDGE_PROMPT = """You are an expert evaluator for a RAG system.
Evaluate the quality of an AI-generated answer.

IMPORTANT:
- For baseline answers with NO sources: faithfulness MUST be low (0.0-0.3)
- For RAG answers with cited sources: judge based on source quality

1. FAITHFULNESS (0-1):
   1.0 = every claim directly supported by sources
   0.5 = some claims unsupported  
   0.0 = no sources provided (baseline)

2. ANSWER_RELEVANCE (0-1):
   1.0 = directly and completely answers the question
   0.5 = partially answers
   0.0 = misses the question entirely

3. COMPLETENESS (0-1):
   1.0 = covers all key information needed
   0.5 = key points there, details missing
   0.0 = major information missing

4. HALLUCINATION_FREE (0-1):
   1.0 = zero hallucinations (GOOD)
   0.5 = minor unsupported claims
   0.0 = major hallucinations (BAD)

Return ONLY valid JSON (no markdown, no extra text):
{
  "faithfulness": 0.0,
  "answer_relevance": 0.0,
  "completeness": 0.0,
  "hallucination_free": 0.0,
  "overall_score": 0.0,
  "verdict": "PASS",
  "feedback": "One sentence feedback."
}

verdict = PASS if overall_score >= 0.50, else FAIL."""


def _clean_json(text: str) -> str:
    """Strip markdown fences."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()
    return text


def judge_answer(query: str, answer: str, context: str) -> dict:
    """Judge a single answer using LLM-as-Judge. Returns normalized keys."""
    try:
        from core.config import FAST_MODEL
        model = FAST_MODEL
    except Exception:
        model = "llama-3.1-8b-instant"

    llm = ChatGroq(
        model=model,
        temperature=0,
        api_key=os.getenv("GROQ_API_KEY", ""),
    )

    try:
        response = llm.invoke([
            SystemMessage(content=JUDGE_PROMPT),
            HumanMessage(content=(
                f"Question: {query}\n\n"
                f"Context provided: {context}\n\n"
                f"Answer to evaluate:\n{answer}"
            )),
        ])
        scores = json.loads(_clean_json(response.content))
    except Exception as e:
        print(f"[Baseline Judge] Failed: {e}")
        scores = {
            "faithfulness": 0.0,
            "answer_relevance": 0.0,
            "completeness": 0.0,
            "hallucination_free": 0.0,
            "overall_score": 0.0,
            "verdict": "FAIL",
            "feedback": f"Judge failed: {str(e)}"
        }

    # NORMALIZE: ensure all required keys exist with CORRECT field names
    return {
        "faithfulness":       float(scores.get("faithfulness", 0.0)),
        "answer_relevance":   float(scores.get("answer_relevance", 0.0)),
        "completeness":       float(scores.get("completeness", 0.0)),
        "hallucination_free": float(scores.get("hallucination_free", 0.0)),
        "overall":            float(scores.get("overall_score", 0.0)),  # ← KEY FIX: overall_score → overall
        "verdict":            str(scores.get("verdict", "FAIL")),
        "feedback":           str(scores.get("feedback", "")),
    }


def run_baseline_llm(query: str) -> dict:
    """Run plain single-LLM — no agents, no retrieval."""
    try:
        from core.config import SMART_MODEL
        model = SMART_MODEL
    except Exception:
        model = "llama-3.3-70b-versatile"

    llm = ChatGroq(
        model=model,
        temperature=0.1,
        api_key=os.getenv("GROQ_API_KEY", ""),
    )

    start = time.time()
    try:
        response = llm.invoke([
            SystemMessage(content=(
                "You are a helpful AI assistant. "
                "Answer the user's question using only your general knowledge. "
                "You have NO access to any database, documents, or internet."
            )),
            HumanMessage(content=query),
        ])
        answer = response.content.strip()
    except Exception as e:
        answer = f"Baseline LLM error: {e}"

    latency = round((time.time() - start) * 1000, 2)
    print(f"[Baseline] Plain LLM answered in {latency}ms")

    return {
        "answer":      answer,
        "latency_ms":  latency,
        "agents_used": [],
        "sources":     0,
    }


def run_comparison(query: str, rag_answer: str, rag_meta: dict) -> dict:
    """
    Main function — runs baseline + judges both + returns comparison.
    All dict keys are normalized to match SystemResultOut Pydantic model.
    """
    print(f"\n[Baseline] Starting comparison for: '{query[:60]}'")

    # Step 1: Run plain LLM baseline
    baseline = run_baseline_llm(query)

    # Step 2: Judge baseline (no sources = low faithfulness)
    print(f"[Baseline] Judging baseline answer...")
    baseline_scores = judge_answer(
        query,
        baseline["answer"],
        "NO SOURCES — this is a plain LLM answer with no retrieval."
    )

    # Step 3: Judge RAG answer
    print(f"[Baseline] Judging RAG answer...")
    rag_scores = judge_answer(
        query,
        rag_answer,
        f"Multi-Agent RAG with {rag_meta.get('sources_count', 0)} retrieved sources."
    )

    # Step 4: Build result objects — ALL keys match SystemResultOut exactly
    baseline_result = {
        "answer":             baseline["answer"],
        "latency_ms":         baseline["latency_ms"],
        "agents_used":        [],
        "sources":            0,
        "faithfulness":       baseline_scores["faithfulness"],
        "answer_relevance":   baseline_scores["answer_relevance"],
        "completeness":       baseline_scores["completeness"],
        "hallucination_free": baseline_scores["hallucination_free"],
        "overall":            baseline_scores["overall"],  # ← KEY FIX: uses "overall"
        "verdict":            baseline_scores["verdict"],
        "feedback":           baseline_scores["feedback"],
    }

    rag_result = {
        "answer":             rag_answer,
        "latency_ms":         rag_meta.get("latency_ms", 0),
        "agents_used":        rag_meta.get("agents_used", []),
        "sources":            rag_meta.get("sources_count", 0),
        "faithfulness":       rag_scores["faithfulness"],
        "answer_relevance":   rag_scores["answer_relevance"],
        "completeness":       rag_scores["completeness"],
        "hallucination_free": rag_scores["hallucination_free"],
        "overall":            rag_scores["overall"],  # ← KEY FIX: uses "overall"
        "verdict":            rag_scores["verdict"],
        "feedback":           rag_scores["feedback"],
    }

    # Step 5: Calculate improvement percentages
    def pct(base_val: float, rag_val: float) -> float:
        if base_val == 0:
            return round(rag_val * 100, 1)
        return round(((rag_val - base_val) / abs(base_val)) * 100, 1)

    improvement = {
        "faithfulness":       pct(baseline_result["faithfulness"],       rag_result["faithfulness"]),
        "answer_relevance":   pct(baseline_result["answer_relevance"],   rag_result["answer_relevance"]),
        "completeness":       pct(baseline_result["completeness"],       rag_result["completeness"]),
        "hallucination_free": pct(baseline_result["hallucination_free"], rag_result["hallucination_free"]),
        "overall":            pct(baseline_result["overall"],            rag_result["overall"]),
    }

    print(f"[Baseline] Comparison done. Improvement: {improvement}")

    return {
        "query":       query,
        "baseline":    baseline_result,
        "rag":         rag_result,
        "improvement": improvement,
    }