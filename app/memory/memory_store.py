"""
Memory design.

We use LangGraph's checkpointer (MemorySaver for dev; swap for a
SqliteSaver/Postgres checkpointer in production) keyed by a `thread_id`
that maps 1:1 to our `session_id`. This gives the orchestrator graph
automatic conversation continuity: every node reads/writes a shared
`messages` list in graph state, and LangGraph persists that state between
HTTP requests for the same session_id.

This is what makes follow-ups like:
  User: "Do you have vegan pasta?"
  Assistant: "Yes -- Vegan Pasta Primavera ..."
  User: "Is it gluten-free?"
work: the second query has no explicit subject, so the orchestrator's
intent classifier is given the full message history and can resolve "it"
to "Vegan Pasta Primavera" before routing to the RAG agent.

Swapping MemorySaver for a persistent backend (SQLite/Postgres) is a
one-line change in graph.py and requires no change to agent code, since
agents never touch the checkpointer directly -- only the orchestrator does.
"""
from langgraph.checkpoint.memory import MemorySaver


def get_checkpointer():
    return MemorySaver()
