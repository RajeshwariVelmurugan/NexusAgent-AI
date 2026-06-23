# core/state.py 

from typing import TypedDict, Annotated, List, Optional
from langchain_core.messages import BaseMessage
import operator


class AgentResult(TypedDict):
    agent: str
    content: str
    source: str
    confidence: float


class RAGState(TypedDict):
    # Phase 1
    query: str
    route: List[str]
    agent_results: Annotated[List[AgentResult], operator.add]
    final_answer: Optional[str]
    messages: Annotated[List[BaseMessage], operator.add]
    metadata: dict

    # Phase 2 (new fields)
    retry_count: int
    reflection: Optional[str]
    memory_context: Optional[str]
    session_id: str
    complexity_score: float