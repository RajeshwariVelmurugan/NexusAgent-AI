# main.py — FastAPI server
# Run: python main.py
# API docs: http://localhost:8000/docs

import time
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from core.graph import graph
from core.state import RAGState

app = FastAPI(
    title="Multi-Agent RAG — Phase 1 (Groq + Llama 3.1)",
    version="1.0.0",
    description="Free, fast, production-ready RAG with Groq",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    query: str
    session_id: str = "default"


class AgentResultOut(BaseModel):
    agent: str
    content: str
    source: str
    confidence: float


class QueryResponse(BaseModel):
    query: str
    final_answer: str
    agent_results: list[AgentResultOut]
    agents_used: list[str]
    avg_confidence: float
    latency_ms: float


@app.post("/query", response_model=QueryResponse)
async def run_query(req: QueryRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    initial_state: RAGState = {
        "query": req.query,
        "route": [],
        "agent_results": [],
        "final_answer": None,
        "messages": [],
        "metadata": {"session_id": req.session_id},
    }

    start = time.time()
    try:
        final_state = graph.invoke(initial_state)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    latency = (time.time() - start) * 1000

    meta = final_state.get("metadata", {})
    return QueryResponse(
        query=req.query,
        final_answer=final_state.get("final_answer", "No answer generated."),
        agent_results=[AgentResultOut(**r) for r in final_state.get("agent_results", [])],
        agents_used=meta.get("agents_used", []),
        avg_confidence=meta.get("avg_confidence", 0.0),
        latency_ms=round(latency, 2),
    )


@app.get("/health")
async def health():
    return {"status": "ok", "llm": "groq/llama-3.1-70b-versatile", "phase": 1}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)