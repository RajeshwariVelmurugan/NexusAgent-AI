# agents/evaluator.py — ALWAYS RUN & CORRECTLY UPDATES GLOBAL STATE

import os, json
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from core.state import RAGState, AgentResult
from core.config import FAST_MODEL, COMPLEXITY_THRESHOLD

JUDGE_PROMPT = """You are an expert evaluator for a RAG system.
Evaluate the quality of an AI-generated answer based strictly on the provided context block.

**CRITICAL METRIC DEFINITIONS:**
1. FAITHFULNESS (0-1): 1.0 if every claim in the answer is directly supported by the context.
2. ANSWER_RELEVANCE (0-1): 1.0 if it directly addresses all parts of the user query.
3. COMPLETENESS (0-1): 1.0 if it captures both internal data and web trends completely.
4. HALLUCINATION_FREE (0-1): 1.0 if there are zero fabricated facts.

Return ONLY a raw, valid JSON object matching this structure exactly:
{
  "faithfulness": 1.0,
  "answer_relevance": 1.0,
  "completeness": 1.0,
  "hallucination_free": 1.0,
  "overall_score": 1.0,
  "verdict": "PASS",
  "feedback": "Detailed quality feedback text."
}
"""


def _clean_json(text: str) -> str:
    """Strip markdown fences if LLM wraps JSON in ```json ... ```"""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.startswith("```")]
        text = "\n".join(lines).strip()
    return text


def _format_sources(results: list[AgentResult]) -> str:
    top = sorted(results, key=lambda r: r["confidence"], reverse=True)[:4]
    return "\n\n".join(
        f"[Source {i+1}: {r['source']}]\n{r['content'][:400]}"
        for i, r in enumerate(top)
    )


def evaluator_node(state: RAGState) -> dict:
    print(f"\n[Evaluator] ===== STARTING EVALUATOR QUALITY AUDIT =====")
    
    query = state["query"]
    answer = state.get("final_answer", "")
    results = state.get("agent_results", [])
    complexity = state.get("complexity_score", 0.5)
    
    print(f"[Evaluator] Processing Active Metrics Framework (Complexity Score={complexity:.2f})...")
    
    if not answer or not results:
        print("[Evaluator] WARNING: Execution aborted due to empty response string or zero source arrays.")
        return {
            "verdict": "FAIL",
            "faithfulness": 0.0,
            "evaluator_status": "ran",
            "critic_status": "ran",
            "metadata": {
                **state.get("metadata", {}),
                "eval_faithfulness": 0.0,
                "eval_verdict": "FAIL",
            }
        }

    llm = ChatGroq(
        model=FAST_MODEL,
        temperature=0,
        api_key=os.getenv("GROQ_API_KEY", ""),
    )
    
    try:
        response = llm.invoke([
            SystemMessage(content=JUDGE_PROMPT),
            HumanMessage(content=(
                f"Question: {query}\n\n"
                f"Sources:\n{_format_sources(results)}\n\n"
                f"Answer to evaluate:\n{answer}"
            )),
        ])
        
        cleaned = _clean_json(response.content)
        scores = json.loads(cleaned)
        
        faithfulness = float(scores.get("faithfulness", 0.0))
        answer_relevance = float(scores.get("answer_relevance", 0.0))
        completeness = float(scores.get("completeness", 0.0))
        hallucination_free = float(scores.get("hallucination_free", 0.0))
        
        overall = float(scores.get("overall_score", 
            round((faithfulness + answer_relevance + completeness + hallucination_free) / 4, 3)))
        
        verdict = "PASS" if overall >= 0.50 else "FAIL"
        feedback = scores.get("feedback", "")
        
        print(f"[Evaluator] Faithfulness:       {faithfulness:.2f}")
        print(f"[Evaluator] Answer relevance:   {answer_relevance:.2f}")
        print(f"[Evaluator] Completeness:       {completeness:.2f}")
        print(f"[Evaluator] Hallucination-free: {hallucination_free:.2f}")
        print(f"[Evaluator] Overall:            {overall:.2f} → {verdict}")
        
        # ⬇️⬇️⬇️ CRITICAL: GLOBAL KEYS FOR UI ⬇️⬇️⬇️
        return {
            # ===== GLOBAL KEYS - UI reads these directly! =====
            "verdict": verdict,                      # UI finds this!
            "faithfulness": round(faithfulness, 3),  # UI finds this!
            
            # ===== STATUS KEYS - For dashboard card colors =====
            "evaluator_status": "ran",               # UI shows green ✓ ran
            "critic_status": "ran",                  # UI shows green ✓ ran
            
            # ===== METADATA - For storage and dashboard =====
            "metadata": {
                **state.get("metadata", {}),
                "eval_faithfulness": round(faithfulness, 3),
                "eval_answer_relevance": round(answer_relevance, 3),
                "eval_completeness": round(completeness, 3),
                "eval_hallucination_free": round(hallucination_free, 3),
                "eval_overall": round(overall, 3),
                "eval_verdict": verdict,
                "eval_feedback": feedback,
                "eval_judge_model": FAST_MODEL,
            }
        }
        
    except Exception as e:
        print(f"[Evaluator] Failed: {e}")
        return {
            "verdict": "FAIL",
            "faithfulness": 0.0,
            "evaluator_status": "error",
            "critic_status": "error",
            "metadata": {
                **state.get("metadata", {}),
                "eval_faithfulness": 0.0,
                "eval_verdict": "FAIL",
                "eval_error": str(e),
            }
        }