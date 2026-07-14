"""
FastAPI entrypoint. Wires: FAISS index -> Retriever -> RAG Agent
                           tools -> Operations Agent
                           both -> Orchestrator (LangGraph, memory-checkpointed)

Run:
    python scripts/build_index.py   # once, or whenever KB json changes
    uvicorn app.main:app --reload
"""
import os
from pathlib import Path
from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv

from app.models.schemas import ChatRequest, ChatResponse
from app.rag.embeddings import SentenceTransformerEmbeddings
from app.rag.vectorstore import FAISSVectorStore
from app.rag.retriever import Retriever
from app.rag.ingest import load_json, get_all_menu_item_names
from app.agents.rag_agent import RAGAgent
from app.agents.operations_agent import OperationsAgent
from app.orchestrator.graph import build_orchestrator, run_turn
from app.memory.memory_store import get_checkpointer
from app.llm_provider import get_llm

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
INDEX_DIR = BASE_DIR / "data" / "faiss_index"
MENU_PATH = BASE_DIR / "data" / "knowledge_base" / "menu.json"

app = FastAPI(title="Restaurant AI Assistant")

_state = {}


@app.on_event("startup")
def startup():
    if not INDEX_DIR.exists():
        raise RuntimeError(
            f"FAISS index not found at {INDEX_DIR}. Run `python scripts/build_index.py` first."
        )

    embedder = SentenceTransformerEmbeddings()
    vectorstore = FAISSVectorStore.load(str(INDEX_DIR))
    retriever = Retriever(vectorstore, embedder)

    menu_json = load_json(str(MENU_PATH))
    all_menu_items = get_all_menu_item_names(menu_json)

    llm = get_llm()

    rag_agent = RAGAgent(retriever, llm, all_menu_item_names=all_menu_items)
    operations_agent = OperationsAgent(llm)
    checkpointer = get_checkpointer()

    orchestrator = build_orchestrator(llm, rag_agent, operations_agent, checkpointer)

    _state["orchestrator"] = orchestrator


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    if "orchestrator" not in _state:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")

    result = run_turn(
        _state["orchestrator"],
        session_id=request.session_id,
        message=request.message,
        branch=request.branch,
    )
    return ChatResponse(**result)


@app.get("/health")
def health():
    return {"status": "ok"}
