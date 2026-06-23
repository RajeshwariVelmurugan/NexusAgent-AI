import os
from tavily import TavilyClient
from core.state import RAGState, AgentResult
from core.config import TOP_K_WEB


def crag_node(state: RAGState) -> dict:
    query = state["query"]
    retry_count = state.get("retry_count", 0)
    old_results = state.get("agent_results", [])

    print(f"\n[CRAG] Corrective retrieval! retry={retry_count + 1}")

    api_key = os.getenv("TAVILY_API_KEY", "").strip()
    if not api_key:
        print("[CRAG] No Tavily key — forcing proceed.")
        return {"retry_count": retry_count + 1, "reflection": "GOOD"}

    kept = [r for r in old_results if r["confidence"] >= 0.7]
    print(f"[CRAG] Kept {len(kept)} high-conf result(s).")

    try:
        client = TavilyClient(api_key=api_key)
        response = client.search(
            query=f"{query} detailed facts",
            search_depth="advanced",
            max_results=TOP_K_WEB,
            include_answer=True,
        )

        new_results = []
        if response.get("answer"):
            new_results.append(AgentResult(
                agent="web_agent", content=response["answer"],
                source="CRAG web search", confidence=0.88,
            ))
        for item in response.get("results", []):
            content = item.get("content", "").strip()
            if not content:
                continue
            score = float(item.get("score", 0.6))
            new_results.append(AgentResult(
                agent="web_agent", content=content,
                source=item.get("url", "unknown"),
                confidence=round(score, 4),
            ))

        print(f"[CRAG] {len(new_results)} new result(s) retrieved.")
        return {"agent_results": kept + new_results, "retry_count": retry_count + 1}

    except Exception as e:
        print(f"[CRAG] Failed: {e}")
        return {"retry_count": retry_count + 1, "reflection": "GOOD"}