# core/rrf.py — Reciprocal Rank Fusion scoring

from core.state import AgentResult

RRF_K = 60


def apply_rrf(agent_results: list[AgentResult]) -> list[AgentResult]:
    if not agent_results:
        return []
    if len(agent_results) == 1:
        return agent_results

    by_agent: dict[str, list[AgentResult]] = {}
    for r in agent_results:
        by_agent.setdefault(r["agent"], []).append(r)

    for agent in by_agent:
        by_agent[agent].sort(key=lambda r: r["confidence"], reverse=True)

    rrf_scores: dict[str, float] = {}
    result_map: dict[str, AgentResult] = {}

    for agent, results in by_agent.items():
        for rank, result in enumerate(results, start=1):
            key = result["source"]
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (RRF_K + rank)
            result_map[key] = result

    sorted_keys = sorted(rrf_scores, key=lambda k: rrf_scores[k], reverse=True)
    max_score = max(rrf_scores.values()) if rrf_scores else 1.0

    reranked: list[AgentResult] = []
    for key in sorted_keys:
        r = dict(result_map[key])
        r["confidence"] = round(rrf_scores[key] / max_score, 4)
        reranked.append(r)

    print(f"[RRF] Re-ranked {len(reranked)} results from {len(by_agent)} agent(s).")
    return reranked