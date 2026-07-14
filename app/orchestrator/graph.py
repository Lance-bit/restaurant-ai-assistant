"""
Main Orchestrator Agent.

Responsibilities (per the assessment spec) live ONLY here, never in the
sub-agents:
  - Classify user intent (rag / operations / clarify)
  - Route to the correct sub-agent
  - Maintain conversation memory (via LangGraph checkpointer, see memory/)
  - Merge/validate sub-agent responses into a single ChatResponse shape
  - Decide when to call tools (by routing to the Operations Agent at all)
  - Handle ambiguity (the "clarify" branch)

The orchestrator contains NO business logic of its own: it never answers a
menu question or checks availability directly. It only classifies, routes,
and packages the sub-agent's result.
"""
from typing import TypedDict, List, Optional, Any
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END

INTENT_SYSTEM_PROMPT = """You classify a restaurant customer's message into exactly one intent.

Intents:
- "rag": questions about menu items, ingredients, allergens, dietary info, opening hours,
  branch policies, loyalty program rules, refund policy, or event/catering information.
- "operations": requests to check table availability, book a table, ask about today's
  special, or check loyalty point balance (i.e. anything requiring a live/operational
  lookup rather than static knowledge).
- "clarify": the message is too ambiguous to route (e.g. missing which branch, or an
  unrelated/unclear message).

Use the conversation history to resolve follow-ups and pronouns (e.g. "is it vegan?"
referring to a dish named earlier).

Respond with ONLY a JSON object, no other text:
{"intent": "rag" | "operations" | "clarify", "branch": "<branch name or null>", "resolved_query": "<the user's question, rewritten to be self-contained using conversation history if needed>"}
"""


class OrchestratorState(TypedDict):
    messages: List[BaseMessage]
    branch: Optional[str]
    intent: Optional[str]
    resolved_query: Optional[str]
    answer: Optional[str]
    agent_used: Optional[str]
    sources: List[str]
    grounded: bool
    tool_calls: List[Any]


def build_orchestrator(llm, rag_agent, operations_agent, checkpointer):
    import json

    def classify_intent(state: OrchestratorState) -> OrchestratorState:
        history = state["messages"]
        prompt_messages = [SystemMessage(content=INTENT_SYSTEM_PROMPT)] + history
        response = llm.invoke(prompt_messages)
        raw = response.content.strip()
        # Defensive parsing: strip markdown fences if the model adds them.
        raw = raw.replace("```json", "").replace("```", "").strip()
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = {"intent": "clarify", "branch": state.get("branch"), "resolved_query": None}

        return {
            **state,
            "intent": parsed.get("intent", "clarify"),
            "branch": parsed.get("branch") or state.get("branch"),
            "resolved_query": parsed.get("resolved_query") or history[-1].content,
        }

    def route(state: OrchestratorState) -> str:
        return state["intent"] if state["intent"] in ("rag", "operations") else "clarify"

    def rag_node(state: OrchestratorState) -> OrchestratorState:
        result = rag_agent.answer(state["resolved_query"], branch=state.get("branch"))
        return {
            **state,
            "answer": result["answer"],
            "agent_used": "rag",
            "sources": result["sources"],
            "grounded": result["grounded"],
            "messages": state["messages"] + [AIMessage(content=result["answer"])],
        }

    def operations_node(state: OrchestratorState) -> OrchestratorState:
        result = operations_agent.answer(state["resolved_query"], branch=state.get("branch"))
        return {
            **state,
            "answer": result["answer"],
            "agent_used": "operations",
            "sources": result["sources"],
            "grounded": result["grounded"],
            "tool_calls": result["tool_calls"],
            "messages": state["messages"] + [AIMessage(content=result["answer"])],
        }

    def clarify_node(state: OrchestratorState) -> OrchestratorState:
        clarification = (
            "Could you clarify a bit? For example, let me know which branch you mean, "
            "or whether you're asking about our menu/policies or about booking/availability."
        )
        return {
            **state,
            "answer": clarification,
            "agent_used": "clarify",
            "sources": [],
            "grounded": True,
            "messages": state["messages"] + [AIMessage(content=clarification)],
        }

    graph = StateGraph(OrchestratorState)
    graph.add_node("classify_intent", classify_intent)
    graph.add_node("rag", rag_node)
    graph.add_node("operations", operations_node)
    graph.add_node("clarify", clarify_node)

    graph.set_entry_point("classify_intent")
    graph.add_conditional_edges(
        "classify_intent", route, {"rag": "rag", "operations": "operations", "clarify": "clarify"}
    )
    graph.add_edge("rag", END)
    graph.add_edge("operations", END)
    graph.add_edge("clarify", END)

    return graph.compile(checkpointer=checkpointer)


def run_turn(compiled_graph, session_id: str, message: str, branch: Optional[str] = None) -> dict:
    config = {"configurable": {"thread_id": session_id}}
    existing = compiled_graph.get_state(config)
    prior_messages = existing.values.get("messages", []) if existing and existing.values else []

    input_state = {
        "messages": prior_messages + [HumanMessage(content=message)],
        "branch": branch,
        "intent": None,
        "resolved_query": None,
        "answer": None,
        "agent_used": None,
        "sources": [],
        "grounded": True,
        "tool_calls": [],
    }
    result = compiled_graph.invoke(input_state, config=config)
    return {
        "session_id": session_id,
        "answer": result["answer"],
        "agent_used": result["agent_used"],
        "sources": result["sources"],
        "grounded": result["grounded"],
    }
