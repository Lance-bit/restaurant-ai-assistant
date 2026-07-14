"""Shared data models for requests, responses, and internal state."""
from typing import Optional, List, Literal
from pydantic import BaseModel


class ChatRequest(BaseModel):
    session_id: str
    message: str
    branch: Optional[str] = None


class RetrievedChunk(BaseModel):
    text: str
    source: str  # "menu" | "policy"
    score: float
    item_name: Optional[str] = None
    section: Optional[str] = None


class AgentResult(BaseModel):
    answer: str
    agent: Literal["rag", "operations", "clarify"]
    sources: List[str] = []
    grounded: bool = True
    tool_calls: List[dict] = []


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    agent_used: str
    sources: List[str] = []
    grounded: bool = True
