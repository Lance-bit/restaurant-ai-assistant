"""
Runs the example queries listed in the assessment PDF end-to-end through
the real orchestrator (requires GROQ_API_KEY + a built FAISS index)
and prints the agent used, sources, groundedness, and answer for each.

Also demonstrates memory continuity via a follow-up question, and the two
required Operations tools.

Usage:
    python scripts/build_index.py
    python tests/test_queries.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

from app.llm_provider import get_llm
from app.rag.embeddings import SentenceTransformerEmbeddings
from app.rag.vectorstore import FAISSVectorStore
from app.rag.retriever import Retriever
from app.rag.ingest import load_json, get_all_menu_item_names
from app.agents.rag_agent import RAGAgent
from app.agents.operations_agent import OperationsAgent
from app.orchestrator.graph import build_orchestrator, run_turn
from app.memory.memory_store import get_checkpointer

BASE_DIR = Path(__file__).resolve().parent.parent
INDEX_DIR = BASE_DIR / "data" / "faiss_index"
MENU_PATH = BASE_DIR / "data" / "knowledge_base" / "menu.json"

RAG_QUERIES = [
    "Do you have vegan pasta?",
    "Is the chicken grilled or fried?",
    "What are your opening hours on weekends?",
    "Do you host birthday events?",
    "What's included in the premium catering package?",
]

OPERATIONS_QUERIES = [
    ("Do you have a table for 2 at the Maadi branch tomorrow at 8pm?", "Maadi"),
    ("What's today's special at Downtown?", "Downtown"),
]

FOLLOW_UP_DEMO = [
    "Do you have vegan pasta?",
    "Is it gluten-free too?",  # follow-up relying on memory to resolve "it"
]


def print_result(label, result):
    print(f"\n--- {label} ---")
    print(f"agent_used: {result['agent_used']}")
    print(f"grounded:   {result['grounded']}")
    print(f"sources:    {result['sources']}")
    print(f"answer:     {result['answer']}")


def main():
    embedder = SentenceTransformerEmbeddings()
    vectorstore = FAISSVectorStore.load(str(INDEX_DIR))
    retriever = Retriever(vectorstore, embedder)

    menu_json = load_json(str(MENU_PATH))
    all_menu_items = get_all_menu_item_names(menu_json)

    llm = get_llm()
    rag_agent = RAGAgent(retriever, llm, all_menu_item_names=all_menu_items)
    operations_agent = OperationsAgent(llm)
    orchestrator = build_orchestrator(llm, rag_agent, operations_agent, get_checkpointer())

    print("=" * 60)
    print("RAG AGENT QUERIES")
    print("=" * 60)
    for i, q in enumerate(RAG_QUERIES):
        result = run_turn(orchestrator, session_id=f"rag-demo-{i}", message=q)
        print_result(q, result)

    print("\n" + "=" * 60)
    print("OPERATIONS AGENT QUERIES")
    print("=" * 60)
    for i, (q, branch) in enumerate(OPERATIONS_QUERIES):
        result = run_turn(orchestrator, session_id=f"ops-demo-{i}", message=q, branch=branch)
        print_result(q, result)

    print("\n" + "=" * 60)
    print("MEMORY CONTINUITY DEMO (same session_id across turns)")
    print("=" * 60)
    session_id = "memory-demo"
    for q in FOLLOW_UP_DEMO:
        result = run_turn(orchestrator, session_id=session_id, message=q)
        print_result(q, result)


if __name__ == "__main__":
    main()
