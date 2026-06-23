# agents/synthesizer.py — Phase 2 with RRF + Memory
# Updated with Business Priority Rules

import os
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from core.state import RAGState, AgentResult
from core.config import SYNTHESIZER_MODEL
from core.rrf import apply_rrf
from core import memory

SYSTEM_PROMPT = """You are an expert answer synthesizer for a multi-agent RAG system.

STRICT PRIORITY RULES:
1. Questions with "our sales", "our data", "our company",
   "we", "invest" → ALWAYS start with DATABASE results first
2. Format: "Based on our sales data, [DB facts]. 
   Industry trends show [web context]."
3. NEVER lead with web/external recommendations
4. DB data = facts about OUR company
5. Web data = industry context only
6. Be concise — 3 to 4 sentences max
7. NEVER mention confidence scores or internal metrics to users
8. Use inline citations: [source: database] or [source: url]

Example for business query:
❌ WRONG: "Industry trends suggest investing in AI..."
✅ CORRECT: "Based on our sales data, our ERP System generated $20,000 in Q3 [source: database]. Industry trends show AI and e-commerce sectors growing 15% in 2026 [source: report]."

Example for simple query:
"What is AI?" → Direct answer without DB or web priority rules.
"""


def _format_context(results: list[AgentResult], mem_ctx: str) -> str:
    """Format results with priority: DB first, then Web, then Doc"""
    blocks = []
    if mem_ctx:
        blocks.append(f"[Past conversation]\n{mem_ctx}")
    
    # Prioritize DB results first for business queries
    db_results = [r for r in results if r["agent"] == "db_agent"]
    web_results = [r for r in results if r["agent"] == "web_agent"]
    doc_results = [r for r in results if r["agent"] == "doc_agent"]
    
    # Reorder: DB → Web → Doc
    sorted_results = db_results + web_results + doc_results
    
    for i, r in enumerate(sorted_results, 1):
        blocks.append(
            f"[Context {i}] Source: {r['source']} | Confidence: {r['confidence']}\n"
            f"{r['content']}"
        )
    return "\n\n---\n\n".join(blocks)


def synthesizer_node(state: RAGState) -> dict:
    query = state["query"]
    results = state.get("agent_results", [])
    session_id = state.get("session_id", "default")
    mem_ctx = state.get("memory_context", "")

    print(f"\n[Synthesizer] Applying RRF to {len(results)} result(s)...")
    ranked = apply_rrf(results)

    if not ranked:
        return {
            "final_answer": "I could not find relevant information. Check documents and API keys.",
            "metadata": {**state.get("metadata", {}), "result_count": 0},
        }

    llm = ChatGroq(
        model=SYNTHESIZER_MODEL, 
        temperature=0.1,
        api_key=os.getenv("GROQ_API_KEY", ""),
    )

    response = llm.invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"Question: {query}\n\nContext:\n\n{_format_context(ranked, mem_ctx)}"),
    ])

    answer = response.content.strip()
    agents_used = list({r["agent"] for r in ranked})
    avg_conf = round(sum(r["confidence"] for r in ranked) / len(ranked), 3)

    memory.save_turn(session_id, query, answer)
    print(f"[Synthesizer] Done. agents={agents_used} avg_conf={avg_conf}")

    return {
        "final_answer": answer,
        "agent_results": ranked,
        "metadata": {
            **state.get("metadata", {}),
            "result_count": len(ranked),
            "agents_used": agents_used,
            "avg_confidence": avg_conf,
            "rrf_applied": True,
        },
    }