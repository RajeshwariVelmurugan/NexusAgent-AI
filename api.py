# api.py — FastAPI Backend Server
# ─────────────────────────────────────────────────────────────
# THE GATEKEEPER between any frontend and LangGraph.
#
# Run: python api.py
# Docs: http://localhost:8000/docs (auto Swagger UI)
# ─────────────────────────────────────────────────────────────

import time
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional

from core.graph import graph
from core.state import RAGState
from core.eval_store import save_eval
from core import memory
from core.baseline import run_comparison  # ← IMPORTANT: This must exist!

app = FastAPI(
    title       = "Multi-Agent RAG API",
    description = "FastAPI + LangGraph backend. Connect any frontend to this.",
    version     = "3.0.0",
)

# Allow Streamlit (8501) + any other frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)


# ── Request / Response models ─────────────────────────────────

class ChatRequest(BaseModel):
    query:      str              = Field(..., min_length=1, description="User question")
    session_id: str              = Field("default", description="Session ID for memory")

    class Config:
        json_schema_extra = {
            "example": {
                "query":      "What were our Q3 sales and industry trends?",
                "session_id": "user-abc-123"
            }
        }


class AgentResultOut(BaseModel):
    agent:      str
    content:    str
    source:     str
    confidence: float


class EvalScores(BaseModel):
    faithfulness:       Optional[float] = None
    answer_relevance:   Optional[float] = None
    completeness:       Optional[float] = None
    hallucination_free: Optional[float] = None
    overall:            Optional[float] = None
    verdict:            Optional[str]   = None
    feedback:           Optional[str]   = None
    judge_model:        Optional[str]   = None


class PipelineInfo(BaseModel):
    route:             list[str]
    agents_used:       list[str]
    complexity:        float
    reflection:        Optional[str]
    critic_verdict:    Optional[str]
    answer_refined:    bool
    retry_count:       int
    rrf_applied:       bool
    avg_confidence:    float


class ChatResponse(BaseModel):
    query:         str
    answer:        str
    sources:       list[AgentResultOut]
    pipeline:      PipelineInfo
    eval_scores:   EvalScores
    latency_ms:    float


# ── NEW: Comparison Models ──────────────────────────────────

class CompareRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Question to compare")
    session_id: str = Field("baseline-test", description="Session ID")


class SystemResultOut(BaseModel):
    answer: str
    latency_ms: float
    agents_used: list[str]
    sources: int
    faithfulness: float
    answer_relevance: float
    completeness: float
    hallucination_free: float
    overall: float
    verdict: str
    feedback: str


class CompareResponse(BaseModel):
    query: str
    baseline: SystemResultOut
    rag: SystemResultOut
    improvement: dict


# ── Main chat endpoint ────────────────────────────────────────

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """
    Main endpoint. Send a question, get a grounded cited answer.

    The full multi-agent pipeline runs:
    orchestrator → agents → reflection → CRAG → synthesizer → critic → evaluator
    """
    start = time.time()

    print(f"\n[API] Received query: {req.query[:100]}...")
    print(f"[API] Session ID: {req.session_id}")

    try:
        result = graph.invoke({
            "query":            req.query,
            "route":            [],
            "agent_results":    [],
            "final_answer":     None,
            "messages":         [],
            "metadata":         {},
            "retry_count":      0,
            "reflection":       None,
            "memory_context":   None,
            "session_id":       req.session_id,
            "complexity_score": 0.0,
        })
    except Exception as e:
        print(f"[API] Error during graph.invoke: {e}")
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")

    latency_ms = round((time.time() - start) * 1000, 2)
    meta       = result.get("metadata", {})
    answer     = result.get("final_answer", "No answer generated.")
    agents     = meta.get("agents_used", [])

    print(f"[API] Completed in {latency_ms}ms | Agents: {agents}")

    # Save eval to JSON store
    save_eval(
        query              = req.query,
        final_answer       = answer,
        agents_used        = agents,
        faithfulness       = meta.get("eval_faithfulness", 1.0),
        answer_relevance   = meta.get("eval_answer_relevance", 1.0),
        completeness       = meta.get("eval_completeness", 1.0),
        hallucination_free = meta.get("eval_hallucination_free", 1.0),
        overall            = meta.get("eval_overall", 1.0),
        verdict            = meta.get("eval_verdict", "SKIPPED"),
        critic_score       = meta.get("critic_score", 1.0),
        retry_count        = meta.get("retry_count", 0),
        avg_confidence     = meta.get("avg_confidence", 0.0),
        latency_ms         = latency_ms,
    )

    return ChatResponse(
        query  = req.query,
        answer = answer,
        sources = [
            AgentResultOut(**s)
            for s in result.get("agent_results", [])
        ],
        pipeline = PipelineInfo(
            route          = result.get("route", []),
            agents_used    = agents,
            complexity     = meta.get("complexity", 0.0),
            reflection     = meta.get("reflection_verdict"),
            critic_verdict = meta.get("critic_verdict"),
            answer_refined = meta.get("answer_refined", False),
            retry_count    = meta.get("retry_count", 0),
            rrf_applied    = meta.get("rrf_applied", False),
            avg_confidence = meta.get("avg_confidence", 0.0),
        ),
        eval_scores = EvalScores(
            faithfulness       = meta.get("eval_faithfulness"),
            answer_relevance   = meta.get("eval_answer_relevance"),
            completeness       = meta.get("eval_completeness"),
            hallucination_free = meta.get("eval_hallucination_free"),
            overall            = meta.get("eval_overall"),
            verdict            = meta.get("eval_verdict"),
            feedback           = meta.get("eval_feedback"),
            judge_model        = meta.get("eval_judge_model"),
        ),
        latency_ms = latency_ms,
    )


# ── NEW: Comparison Endpoint ──────────────────────────────────

@app.post("/compare", response_model=CompareResponse)
async def compare(req: CompareRequest):
    """
    Runs the SAME query through:
      1. A plain single-LLM baseline (no agents, no RAG)
      2. Your full Multi-Agent RAG pipeline

    Both answers are scored by the same LLM-as-Judge rubric.
    Returns a side-by-side comparison with improvement percentages.
    """
    print(f"\n[API] Running comparison for: {req.query[:100]}...")
    
    start = time.time()

    try:
        # Run the full RAG pipeline
        result = graph.invoke({
            "query":            req.query,
            "route":            [],
            "agent_results":    [],
            "final_answer":     None,
            "messages":         [],
            "metadata":         {},
            "retry_count":      0,
            "reflection":       None,
            "memory_context":   None,
            "session_id":       req.session_id,
            "complexity_score": 0.0,
        })
    except Exception as e:
        print(f"[API] Error during graph.invoke: {e}")
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")

    rag_latency = round((time.time() - start) * 1000, 2)
    meta = result.get("metadata", {})
    rag_answer = result.get("final_answer", "No answer generated.")

    rag_meta = {
        "latency_ms":     rag_latency,
        "agents_used":    meta.get("agents_used", []),
        "sources_count":  len(result.get("agent_results", [])),
    }

    print(f"[API] RAG pipeline completed in {rag_latency}ms")

    # Run comparison — baseline + both judged
    try:
        comparison = run_comparison(req.query, rag_answer, rag_meta)
    except Exception as e:
        print(f"[API] Comparison failed: {e}")
        raise HTTPException(status_code=500, detail=f"Comparison error: {str(e)}")

    return CompareResponse(
        query       = comparison["query"],
        baseline    = SystemResultOut(**comparison["baseline"]),
        rag         = SystemResultOut(**comparison["rag"]),
        improvement = comparison["improvement"],
    )


# ── Memory endpoints ──────────────────────────────────────────

@app.delete("/memory/{session_id}")
async def clear_memory(session_id: str):
    """Clear Redis memory for a session (new chat button)."""
    memory.clear_session(session_id)
    return {"message": f"Session '{session_id}' cleared."}


# ── Eval endpoints ────────────────────────────────────────────

@app.get("/evals")
async def get_evals():
    """Get all evaluation records for dashboard charts."""
    from core.eval_store import load_evals
    return load_evals()


@app.delete("/evals")
async def clear_evals_endpoint():
    """Clear all evaluation history."""
    from core.eval_store import clear_evals
    clear_evals()
    return {"message": "Eval history cleared."}


# ── Health check ──────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status":  "ok",
        "version": "3.0.0",
        "backend": "FastAPI + LangGraph",
        "model":   "Groq Llama 3.3 70B",
    }


# ── Run directly ──────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True,
        timeout_keep_alive=300,
        timeout_graceful_shutdown=30
    )