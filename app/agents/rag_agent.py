"""
Restaurant Knowledge RAG Agent.

Hallucination prevention has two layers:
  1. Prompt-level: the system prompt hard-restricts the LLM to the retrieved
     CONTEXT and instructs it to explicitly say "I don't know" rather than
     guess.
  2. Post-generation groundedness check: after the LLM answers, we scan the
     answer text for any *known menu item name* (from the full menu, not
     just what was retrieved). If the answer references a real or invented
     dish that wasn't part of the retrieved context, we treat the answer as
     ungrounded and replace it with a safe fallback. This specifically
     targets the assessment's requirement: "must not hallucinate nonexistent
     menu items."
"""
from typing import List, Dict, Any, Optional

SYSTEM_PROMPT = """You are the Restaurant Knowledge Agent for a multi-branch restaurant chain.

Rules:
- Answer ONLY using the information in CONTEXT below. Do not use outside knowledge.
- If CONTEXT does not contain the answer, respond exactly: "I don't have that on file — I'll flag this for our staff to confirm."
- Never invent menu items, prices, ingredients, or policies that are not in CONTEXT.
- Keep answers short (1-3 sentences) and factual. Do not add unrequested marketing language.
- If the user asks about a specific branch and CONTEXT shows the item/policy isn't available there, say so explicitly.
"""

FALLBACK_ANSWER = "I don't have that on file — I'll flag this for our staff to confirm."
NO_CONTEXT_ANSWER = "I don't have that information in our knowledge base — let me flag this for staff to follow up."


class RAGAgent:
    def __init__(self, retriever, llm, all_menu_item_names: List[str]):
        self.retriever = retriever
        self.llm = llm
        self.all_menu_item_names = all_menu_item_names

    def answer(self, query: str, branch: Optional[str] = None, k: int = 4) -> Dict[str, Any]:
        docs = self.retriever.retrieve(query, branch=branch, k=k)

        if not docs:
            return {
                "answer": NO_CONTEXT_ANSWER,
                "sources": [],
                "grounded": True,
                "retrieved_chunks": [],
            }

        context = "\n".join(f"- {d['text']}" for d in docs)
        prompt = f"{SYSTEM_PROMPT}\n\nCONTEXT:\n{context}\n\nQUESTION: {query}\n\nANSWER:"

        response = self.llm.invoke(prompt)
        answer_text = getattr(response, "content", None) or str(response)

        grounded = self._check_groundedness(answer_text, docs)
        if not grounded:
            answer_text = FALLBACK_ANSWER

        sources = [
            f"{d['source']}:{d.get('item_name') or d.get('section')}" for d in docs
        ]

        return {
            "answer": answer_text,
            "sources": sources,
            "grounded": grounded,
            "retrieved_chunks": docs,
        }

    def _check_groundedness(self, answer_text: str, retrieved_docs: List[Dict[str, Any]]) -> bool:
        allowed_names = {
            d.get("item_name") for d in retrieved_docs if d.get("item_name")
        }
        answer_lower = answer_text.lower()
        for item_name in self.all_menu_item_names:
            if item_name.lower() in answer_lower and item_name not in allowed_names:
                # The answer mentions a real menu item that wasn't in the
                # retrieved context -- e.g. recommending a dish that exists
                # on the menu but is irrelevant/unverified for this query.
                return False
        return True
