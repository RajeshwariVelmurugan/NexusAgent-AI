# agents/orchestrator.py — Phase 2 with Complexity + Memory
# Fix: Enhanced db_agent examples for better routing

import json, os
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from core.state import RAGState
from core.config import GENERATOR_MODEL
from core.complexity import score_complexity
from core import memory

SYSTEM_PROMPT = """You are a query router for a multi-agent RAG system.

Available agents:
  - doc_agent : searches uploaded PDF and text documents
               Examples: "summarize the report", "what does the contract say",
                         "key highlights from uploaded files", "find in documents"

  - db_agent  : queries PostgreSQL database (sales, orders, users, structured data)
               Examples: "top 5 customers by revenue", "Q3 sales figures",
                         "average order value", "monthly revenue",
                         "our sales data", "our revenue", 
                         "which segment should we invest",
                         "based on our sales"  # ← ADDED THIS

  - web_agent : searches the live internet for current information
               Examples: "latest AI news 2026", "current market trends",
                         "recent developments in", "today's news"

Instructions:
  1. Read the user query AND past conversation context if provided.
  2. Use context for follow-up questions (e.g. "tell me more" = same topic).
  3. Pick the best agent(s) — use multiple for multi-source questions.
  4. Return ONLY valid JSON — no markdown, no extra text.

Output format:
{
  "route": ["doc_agent"],
  "reason": "One sentence explaining your choice."
}"""


def orchestrator_node(state: RAGState) -> dict:
    query = state["query"]
    session_id = state.get("session_id", "default")

    # Load memory context FIRST
    mem_context = memory.get_context(session_id)

    # Score complexity with memory context info
    complexity = score_complexity(query, has_memory=bool(mem_context))

    llm = ChatGroq(
        model=GENERATOR_MODEL, temperature=0,
        api_key=os.getenv("GROQ_API_KEY", ""),
    )

    user_msg = f"Query: {query}"
    if mem_context:
        user_msg = f"{mem_context}\n\nCurrent Query: {query}"

    response = llm.invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_msg),
    ])

    try:
        parsed = json.loads(response.content)
        route = parsed.get("route", ["doc_agent", "web_agent"])
        reason = parsed.get("reason", "")
    except json.JSONDecodeError:
        route = ["doc_agent", "web_agent", "db_agent"]
        reason = "JSON parse failed — using all agents."

    valid = {"doc_agent", "web_agent", "db_agent"}
    route = [a for a in route if a in valid] or list(valid)

    print(f"\n[Orchestrator] Route      → {route}")
    print(f"[Orchestrator] Complexity → {complexity} ({'COMPLEX' if complexity >= 0.5 else 'SIMPLE'})")
    print(f"[Orchestrator] Reason     → {reason}")

    return {
        "route": route,
        "memory_context": mem_context,
        "complexity_score": complexity,
        "metadata": {"routing_reason": reason, "complexity": complexity},
    }


def route_to_agents(state: RAGState) -> list[str]:
    return state["route"]