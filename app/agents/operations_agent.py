"""
Operations Agent (tool-based / MCP-style).

Uses the LLM's native tool-calling to (a) decide which tool to call and
(b) extract structured parameters (date, time, branch, user_id) from the
user's natural-language message, then executes the corresponding simulated
backend function. This mirrors how a real MCP client would hand off to an
MCP server's tool_use blocks -- the LLM never fabricates the result, it only
fabricates the *call*; the actual data always comes from operations_tools.py.
"""
from typing import Dict, Any, Optional, List
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from datetime import datetime
import re

from app.tools import operations_tools as ops

SYSTEM_PROMPT = """You are the Operations Agent for a restaurant chain.
You have tools to check table availability, book a table, get today's
special, and check loyalty points. Always call a tool rather than guessing
an answer. If required parameters (date, time, branch, name, user_id) are
missing from the user's message, ask a short clarifying question instead of
calling a tool with made-up values. Today's date context will be given to
you if relevant.
"""


@tool
def check_table_availability(date: str, time: str, branch: str) -> Dict[str, Any]:
    """Check how many tables are available at a branch on a given date/time.
    date format: YYYY-MM-DD. time format: HH:MM (24h). branch: Downtown, Maadi, or New Cairo."""
    return ops.check_table_availability(date, time, branch)


@tool
def book_table(name: str, date: str, time: str, branch: str) -> Dict[str, Any]:
    """Book a table for a guest. date format: YYYY-MM-DD, time format: HH:MM (24h),
    branch: Downtown, Maadi, or New Cairo."""
    return ops.book_table(name, date, time, branch)


@tool
def get_today_special(branch: str) -> Dict[str, Any]:
    """Get today's chef special for a branch (Downtown, Maadi, or New Cairo)."""
    return ops.get_today_special(branch)


@tool
def check_loyalty_points(user_id: str) -> Dict[str, Any]:
    """Check a customer's loyalty point balance by user_id."""
    return ops.check_loyalty_points(user_id)


ALL_TOOLS = [check_table_availability, book_table, get_today_special, check_loyalty_points]
TOOLS_BY_NAME = {t.name: t for t in ALL_TOOLS}
MAX_TOOL_ROUNDS = 3


def _fallback_answer_from_tool_result(tr: Dict[str, Any]) -> str:
    """
    Templated answer built directly from real tool output, used only when the
    LLM returns empty content after a tool round-trip (observed with some
    Groq/Llama responses). Guarantees the customer never sees a blank reply.
    """
    tool_name, result = tr["tool"], tr["result"]

    if tool_name == "check_table_availability":
        if not result.get("success"):
            return result.get("error", "I couldn't check availability for that branch.")
        if result["status"] == "available":
            return (
                f"Yes, we have {result['available_tables']} table(s) available at "
                f"{result['branch']} on {result['date']} at {result['time']}."
            )
        return f"Sorry, we're fully booked at {result['branch']} on {result['date']} at {result['time']}."

    if tool_name == "book_table":
        if result.get("success"):
            b = result["booking"]
            return (
                f"Booked! Your reservation ID is {b['booking_id']} for {b['name']} "
                f"at {b['branch']} on {b['date']} at {b['time']}."
            )
        return result.get("message", "I couldn't complete that booking.")

    if tool_name == "get_today_special":
        if not result.get("success"):
            return result.get("error", "I couldn't find today's special for that branch.")
        return f"Today's special at {result['branch']} is {result['special']}."

    if tool_name == "check_loyalty_points":
        if not result.get("success"):
            return result.get("error", "I couldn't find a loyalty record for that user.")
        msg = f"You have {result['points']} loyalty points."
        if result["redeemable"]:
            msg += f" That's enough for a {result['discount_available_egp']} EGP discount."
        return msg

    return str(result)


def _check_operations_groundedness(answer_text: str, tool_results: List[Dict[str, Any]]) -> bool:
    """
    Lightweight numeric groundedness check, the Operations-agent counterpart
    to RAGAgent's menu-name check. The LLM only fabricates the *phrasing* of
    an operations answer, never the underlying data -- but nothing stops it
    from misquoting a number while paraphrasing (e.g. saying "3 tables" when
    the tool actually returned 4). This checks the specific quantities most
    likely to matter -- table count, loyalty points, discount amount --
    against the real tool output, rather than trusting the paraphrase blindly.
    Dates/times are intentionally not checked here: they appear in many
    natural phrasings and checking them generically would produce false
    positives; the quantities below are the ones a customer would actually
    act on if wrong.
    """
    answer_lower = answer_text.lower()

    for tr in tool_results:
        tool_name, result = tr["tool"], tr["result"]

        if tool_name == "check_table_availability" and result.get("success"):
            expected = result["available_tables"]
            for match in re.finditer(r'(\d+)\s*tables?\b', answer_lower):
                if int(match.group(1)) != expected:
                    return False

        if tool_name == "check_loyalty_points" and result.get("success"):
            expected_points = result["points"]
            for match in re.finditer(r'(\d+)\s*(?:loyalty\s*)?points?\b', answer_lower):
                if int(match.group(1)) != expected_points:
                    return False
            if result.get("redeemable"):
                expected_discount = result["discount_available_egp"]
                for match in re.finditer(r'(\d+)\s*egp\b', answer_lower):
                    if int(match.group(1)) != expected_discount:
                        return False

    return True


class OperationsAgent:
    def __init__(self, llm):
        self.llm = llm.bind_tools(ALL_TOOLS)

    def answer(self, query: str, branch: Optional[str] = None) -> Dict[str, Any]:
        today_str = datetime.now().strftime("%A, %Y-%m-%d")
        context_hint = f"\nToday's date is {today_str}."
        if branch:
            context_hint += f" User's current branch context, if relevant: {branch}."
        messages = [
            SystemMessage(content=SYSTEM_PROMPT + context_hint),
            HumanMessage(content=query),
        ]
        ai_message = self.llm.invoke(messages)

        if not getattr(ai_message, "tool_calls", None):
            # LLM decided it needs clarification rather than calling a tool.
            return {
                "answer": ai_message.content,
                "tool_calls": [],
                "sources": [],
                "grounded": True,
            }

        all_tool_results = []
        rounds = 0

        # Loop rather than a single tool round-trip: some models chain a
        # second tool call (e.g. checking availability before booking)
        # instead of answering in plain text after the first result.
        while getattr(ai_message, "tool_calls", None) and rounds < MAX_TOOL_ROUNDS:
            messages.append(ai_message)
            for call in ai_message.tool_calls:
                tool_fn = TOOLS_BY_NAME[call["name"]]
                result = tool_fn.invoke(call["args"])
                tr = {"tool": call["name"], "args": call["args"], "result": result}
                all_tool_results.append(tr)
                messages.append(ToolMessage(content=str(result), tool_call_id=call["id"]))
            ai_message = self.llm.invoke(messages)
            rounds += 1

        answer_text = ai_message.content
        grounded = True

        if not answer_text and all_tool_results:
            # Never surface a blank reply to the customer -- fall back to a
            # templated answer built directly from the last real tool result.
            answer_text = _fallback_answer_from_tool_result(all_tool_results[-1])
        elif answer_text and all_tool_results and not _check_operations_groundedness(answer_text, all_tool_results):
            # The LLM's phrasing contradicts the real tool output (e.g. wrong
            # table count) -- discard it and use the templated fallback
            # instead of risking a wrong number reaching the customer.
            answer_text = _fallback_answer_from_tool_result(all_tool_results[-1])
            grounded = False

        return {
            "answer": answer_text,
            "tool_calls": all_tool_results,
            "sources": [f"tool:{tr['tool']}" for tr in all_tool_results],
            "grounded": grounded,
        }
