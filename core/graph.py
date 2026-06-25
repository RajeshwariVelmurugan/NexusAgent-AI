# core/graph.py — Phase 2 with full pipeline
# START → orchestrator → agents → reflection → crag? → synthesizer → critic → evaluator → END

from langgraph.graph import StateGraph, START, END
from core.state import RAGState
from agents.orchestrator import orchestrator_node, route_to_agents
from agents.doc_agent import doc_agent_node
from agents.web_agent import web_agent_node
from agents.db_agent import db_agent_node
from agents.reflection import reflection_node, should_retry
from agents.crag import crag_node
from agents.synthesizer import synthesizer_node
from agents.critic_agent import critic_agent_node
from agents.evaluator import evaluator_node


def build_graph():
    builder = StateGraph(RAGState)

    # Add all nodes - CHANGED: "reflection" → "reflection_node"
    builder.add_node("orchestrator", orchestrator_node)
    builder.add_node("doc_agent", doc_agent_node)
    builder.add_node("web_agent", web_agent_node)
    builder.add_node("db_agent", db_agent_node)
    builder.add_node("reflection_node", reflection_node)  # ← CHANGED
    builder.add_node("crag", crag_node)
    builder.add_node("synthesizer", synthesizer_node)
    builder.add_node("critic_agent", critic_agent_node)
    builder.add_node("evaluator", evaluator_node)

    # Start -> orchestrator
    builder.add_edge(START, "orchestrator")

    # Orchestrator -> parallel agents
    builder.add_conditional_edges(
        "orchestrator", route_to_agents,
        {"doc_agent": "doc_agent", "web_agent": "web_agent", "db_agent": "db_agent"},
    )

    # All agents -> reflection_node (CHANGED)
    builder.add_edge("doc_agent", "reflection_node")
    builder.add_edge("web_agent", "reflection_node")
    builder.add_edge("db_agent", "reflection_node")

    # Reflection -> CRAG retry OR synthesizer (CHANGED)
    builder.add_conditional_edges(
        "reflection_node", should_retry,
        {"retry": "crag", "synthesize": "synthesizer"},
    )

    # CRAG -> back to reflection_node (CHANGED)
    builder.add_edge("crag", "reflection_node")

    # Synthesizer -> critic -> evaluator -> END
    builder.add_edge("synthesizer", "critic_agent")
    builder.add_edge("critic_agent", "evaluator")
    builder.add_edge("evaluator", END)

    return builder.compile()


graph = build_graph()