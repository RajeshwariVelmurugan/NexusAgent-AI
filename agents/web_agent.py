# agents/web_agent.py - OPTIMIZED with timeout

import os
import signal
from tavily import TavilyClient
from core.state import RAGState, AgentResult
from core.config import TOP_K_WEB


def web_agent_node(state: RAGState) -> dict:
    query = state["query"]
    api_key = os.getenv("TAVILY_API_KEY", "").strip()

    print(f"\n[WebAgent] Searching web for: '{query}'")

    if not api_key:
        print("[WebAgent] TAVILY_API_KEY not set — skipping web search.")
        return {"agent_results": []}

    try:
        client = TavilyClient(api_key=api_key)
        
        response = client.search(
            query=query,
            search_depth="basic",      # "basic" is faster (1-2 sec vs 3-5 sec)
            max_results=TOP_K_WEB,     # Now 1 result only!
            include_answer=True,
        )

        results = []

        if response.get("answer"):
            results.append(AgentResult(
                agent="web_agent",
                content=response["answer"],
                source="Tavily synthesized answer",
                confidence=0.85,
            ))

        for item in response.get("results", []):
            content = item.get("content", "").strip()
            if not content:
                continue
            score = float(item.get("score", 0.5))
            results.append(AgentResult(
                agent="web_agent",
                content=content,
                source=item.get("url", "unknown"),
                confidence=round(score, 4),
            ))
            print(f"  [WebAgent] {score:.3f} | {item.get('url', '')}")

        print(f"[WebAgent] {len(results)} result(s) found.")
        return {"agent_results": results}

    except Exception as e:
        print(f"[WebAgent] Search failed: {e}")
        return {"agent_results": []}