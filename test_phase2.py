# test_phase2.py — tests all Phase 2 features
# Usage: python test_phase2.py

from dotenv import load_dotenv
load_dotenv()

from core.graph import graph
from core.state import RAGState
from core import memory

SESSION = "test-phase2"


def run(query: str, label: str = ""):
    print("\n" + "="*60)
    if label:
        print(f"[{label}]")
    print(f"QUERY: {query}")
    print("="*60)

    result = graph.invoke({
        "query": query,
        "route": [],
        "agent_results": [],
        "final_answer": None,
        "messages": [],
        "metadata": {},
        "retry_count": 0,
        "reflection": None,
        "memory_context": None,
        "session_id": SESSION,
        "complexity_score": 0.0,
    })

    meta = result.get("metadata", {})

    print(f"\n─── Complexity ────────────────────")
    c = meta.get("complexity", 0)
    print(f"Score: {c:.2f} → {'COMPLEX (full pipeline)' if c >= 0.5 else 'SIMPLE (skipped reflection/critic/eval)'}")

    print(f"\n─── Pipeline ───────────────────────")
    print(f"Route    : {result.get('route')}")
    print(f"Agents   : {meta.get('agents_used')}")
    print(f"Retries  : {meta.get('retry_count', 0)}")
    print(f"Reflection: {meta.get('reflection_verdict', 'N/A')}")

    print(f"\n─── Critic (Draft → Critique → Refine) ──")
    print(f"Score  : {meta.get('critic_score', 'N/A')}")
    print(f"Verdict: {meta.get('critic_verdict', 'N/A')}")
    print(f"Refined: {meta.get('answer_refined', False)}")

    print(f"\n─── LLM-as-Judge (Mixtral) ─────────")
    print(f"Judge model       : {meta.get('eval_judge_model', 'N/A')}")
    print(f"Faithfulness      : {meta.get('eval_faithfulness', 'N/A')}")
    print(f"Answer relevance  : {meta.get('eval_answer_relevance', 'N/A')}")
    print(f"Hallucination-free: {meta.get('eval_hallucination_free', 'N/A')}")
    print(f"Overall           : {meta.get('eval_overall', 'N/A')} → {meta.get('eval_verdict', 'N/A')}")

    print(f"\n─── Final answer ───────────────────")
    print(result.get("final_answer"))
    return result


if __name__ == "__main__":
    # Clear previous session
    memory.clear_session(SESSION)

    # Test 1: Simple query (should skip reflection/critic/eval)
    run("What is AI?", "Fix 3: Simple query — should skip 3 nodes")

    # Test 2: Complex multi-source query (full pipeline)
    run("What were our Q3 sales and what do industry trends say about that market?",
        "Complex query — full pipeline + Mixtral judge")

    # Test 3: Memory follow-up
    run("Which product had the highest revenue in that quarter?",
        "Memory continuity test")

    print("\n" + "="*60)
    print("Phase 2 — All tests complete!")