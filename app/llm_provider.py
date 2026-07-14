"""
LLM provider factory.

The orchestrator, RAG agent, and Operations agent all just take an `llm`
object that implements LangChain's chat model interface (`.invoke(...)`,
`.bind_tools(...)`) -- none of them care which provider is behind it. This
factory is the single place that decides which provider to instantiate,
controlled by the LLM_PROVIDER env var, so switching providers never means
touching orchestrator/agent code.

Supported:
  LLM_PROVIDER=anthropic  (default if unset)  -> requires ANTHROPIC_API_KEY
  LLM_PROVIDER=groq                            -> requires GROQ_API_KEY (free)
  LLM_PROVIDER=gemini                          -> requires GOOGLE_API_KEY (free)

Note on tool-calling reliability: the orchestrator's classify_intent step
requires the model to reliably return a strict JSON object, and the
Operations agent relies on bind_tools() for structured parameter extraction.
Anthropic's Claude models are the most reliable at both. Groq's
llama-3.3-70b-versatile and Gemini's gemini-2.0-flash both support tool
calling and generally follow "JSON only" instructions well, but expect
occasional parse failures -- the orchestrator already falls back to a
"clarify" response rather than crashing when that happens.
"""
import os


def get_llm(temperature: float = 0):
    provider = os.getenv("LLM_PROVIDER", "anthropic").lower()

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        model = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")
        return ChatAnthropic(model=model, temperature=temperature)

    if provider == "groq":
        from langchain_groq import ChatGroq
        model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        return ChatGroq(model=model, temperature=temperature)

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
        return ChatGoogleGenerativeAI(model=model, temperature=temperature)

    raise ValueError(
        f"Unknown LLM_PROVIDER '{provider}'. Use 'anthropic', 'groq', or 'gemini'."
    )
