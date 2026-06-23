# agents/reflection.py — Quality Evaluator with Unified CRAG Retry Logic
import os, json
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from core.state import RAGState, AgentResult
from core.config import GENERATOR_MODEL, COMPLEXITY_THRESHOLD

REFLECTION_PROMPT = """You are a quality evaluator for a RAG system.

Evaluate whether the retrieved context results contain sufficient factual depth to answer the user's question completely.

Criteria:
1. RELEVANCE    — Do the results directly address the subject of the question?
2. COMPLETENESS — Is there enough substantial data/metrics for a full business answer?
3. CONFIDENCE   — Are source precision vectors high?

Return ONLY a valid, raw JSON object matching this structure:
{
  "verdict": "GOOD",
  "reason": "One concise sentence summarizing data quality.",
  "avg_relevance": 0.90
}

Allowed Verdicts:
- GOOD  = Data is sufficient. Proceed directly to answer generation.
- RETRY = Data is missing context or insufficient. Trigger a web search fallback.
"""


def _clean_json(text: str) -> str:
    """Strip markdown fences if LLM wraps JSON in ```json ... ```"""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.startswith("```")]
        text = "\n".join(lines).strip()
    return text


def _format_results(results: list[AgentResult]) -> str:
    """Format top results for LLM evaluation"""
    top = sorted(results, key=lambda r: r["confidence"], reverse=True)[:3]
    return "\n\n".join(
        f"Result {i+1}: [{r['agent']}] source={r['source']} conf={r['confidence']}\n{r['content'][:250]}"
        for i, r in enumerate(top)
    )


def reflection_node(state: RAGState) -> dict:
    """
    Evaluates retrieval quality and decides whether to retry or proceed.
    """
    query = state["query"]
    results = state.get("agent_results", [])
    complexity = state.get("complexity_score", 0.5)
    
    # FIX 1: Synchronize retry state variable to look at metadata['crag_retry_count'] natively
    metadata = state.get("metadata", {})
    retries = metadata.get("crag_retry_count", 0)

    # OPTIONAL GUARDRAIL BYPASS: Comment out or adjust threshold to force execution if UI skips
    if complexity < COMPLEXITY_THRESHOLD:
        print(f"\n[Reflection] Baseline check -> Low complexity ({complexity:.2f}). Running evaluation trace anyway for metric safety.")

    print(f"\n[Reflection] Evaluating {len(results)} payload source(s)... Active CRAG iteration: #{retries}")

    # Immediate fallback if agents found absolutely nothing
    if not results:
        print("[Reflection] Critical Alert: Source arrays are empty -> Triggering immediate RETRY path.")
        return {
            "reflection": "RETRY",
            "metadata": {**metadata, "reflection_verdict": "RETRY"}
        }

    # Hard boundary block inside the node itself
    if retries >= 1:
        print("[Reflection] Maximum fallback optimization ceiling reached. Forcing GOOD path to eliminate latency drift.")
        return {
            "reflection": "GOOD",
            "metadata": {**metadata, "reflection_verdict": "GOOD"}
        }

    try:
        llm = ChatGroq(
            model=GENERATOR_MODEL, 
            temperature=0,
            api_key=os.getenv("GROQ_API_KEY", ""),
        )

        response = llm.invoke([
            SystemMessage(content=REFLECTION_PROMPT),
            HumanMessage(content=f"User Query: {query}\n\nRetrieved Payload Context Blocks:\n{_format_results(results)}"),
        ])

        cleaned = _clean_json(response.content)
        parsed = json.loads(cleaned)
        
        verdict = parsed.get("verdict", "GOOD").upper().strip()
        reason = parsed.get("reason", "")
        rel = parsed.get("avg_relevance", 0.5)

        print(f"[Reflection] LLM Evaluation Finished -> Verdict: {verdict} | Relevance: {rel:.2f} | Reason: {reason}")

        return {
            "reflection": verdict,
            "reflection_status": "ran",  
            "metadata": {
                **metadata,
                "reflection_verdict": verdict,
                "reflection_reason": reason,
                "reflection_relevance": rel,
            },
        }


    except Exception as e:
        print(f"[Reflection] Execution Exception fallback triggered: {e} -> Defaulting state to GOOD.")
        return {
            "reflection": "GOOD",
            "metadata": {**metadata, "reflection_verdict": "GOOD"}
        }


def should_retry(state: RAGState) -> str:
    """
    LangGraph Conditional Edge Router - Guarantees execution breaks out after a single retry pass.
    """
    metadata = state.get("metadata", {})
    retries = metadata.get("crag_retry_count", 0)
    reflection_verdict = metadata.get("reflection_verdict", "GOOD")
    
    print(f"\n[CRAG Router] Verification Verdict State: {reflection_verdict} | Logged Loop Retries: {retries}")
    
    # STRICT SINGLE RETRY LIMIT ENFORCEMENT
    if reflection_verdict == "RETRY" and retries < 1:
        print(f"[CRAG Router] ⚠️ Evaluation failure intercepted. Activating CRAG corrective retrieval sequence. (Pass {retries + 1}/1)")
        metadata["crag_retry_count"] = retries + 1
        return "retry"
    
    print("[CRAG Router] ✅ Loop conditions satisfied or max retries exhausted. Transferring execution thread to Synthesizer.")
    return "synthesize"