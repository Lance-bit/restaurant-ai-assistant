# Restaurant AI Assistant — Multi-Agent RAG System

A production-style multi-agent architecture for a restaurant chain: a LangGraph
orchestrator routes customer messages to either a **Restaurant Knowledge RAG
Agent** (menu + policies, grounded in a FAISS-indexed knowledge base) or an
**Operations Agent** (live table availability, bookings, today's special,
loyalty points, via tool-calling).

## Architecture

```
                         ┌─────────────────────────┐
   User message  ──────► │   Orchestrator (LangGraph)│
                         │  - classify_intent        │
                         │  - route (conditional edge)│
                         │  - memory (checkpointer)   │
                         └───────────┬─────────────┘
                        ┌────────────┼────────────┐
                        ▼            ▼            ▼
                 ┌────────────┐ ┌──────────┐ ┌───────────┐
                 │  RAG Agent │ │Operations│ │  Clarify   │
                 │            │ │  Agent   │ │  (ask user)│
                 └─────┬──────┘ └────┬─────┘ └───────────┘
                       │             │
                 ┌─────▼─────┐  ┌────▼─────────────┐
                 │  Retriever │  │ Simulated tools:  │
                 │  (FAISS +  │  │ check_availability│
                 │  filters)  │  │ book_table         │
                 └─────┬──────┘  │ get_today_special  │
                       │         │ check_loyalty_pts   │
                 ┌─────▼──────┐  └───────────────────┘
                 │ menu.json  │
                 │ policies.  │
                 │    json    │
                 └────────────┘
```

**The orchestrator holds no business logic.** It only classifies intent,
resolves follow-up references using conversation history, routes to a
sub-agent, and packages the result. All menu/policy knowledge lives in the
RAG agent; all live/operational logic lives in the Operations agent.

## Repo layout

```
app/
  orchestrator/graph.py     # LangGraph StateGraph: classify -> route -> agent
  agents/rag_agent.py        # grounded generation + hallucination guard
  agents/operations_agent.py # LLM tool-calling wrapper
  rag/
    ingest.py                # chunking (see "RAG design decisions")
    embeddings.py             # pluggable embedding provider
    vectorstore.py            # FAISS wrapper
    retriever.py               # top-k + branch filter + score threshold
  tools/operations_tools.py  # simulated MCP-style backend functions
  memory/memory_store.py     # LangGraph checkpointer wrapper
  models/schemas.py           # Pydantic request/response models
  main.py                     # FastAPI app
data/knowledge_base/          # menu.json, policies.json (source documents)
scripts/build_index.py        # one-time FAISS index build
tests/test_queries.py         # runs the assessment's example queries end-to-end
```

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # set LLM_PROVIDER and the matching API key

python scripts/build_index.py     # builds data/faiss_index/ from the JSON KB
uvicorn app.main:app --reload     # starts the API on :8000
```

> **Disk note:** `sentence-transformers` pulls in full PyTorch (~500MB) plus
> CUDA libraries (~1.5GB combined) even on machines with no GPU. If you're on
> a disk-constrained environment (e.g. a free-tier GitHub Codespace), install
> the CPU-only PyTorch build first to skip the CUDA packages entirely:
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cpu
> pip install -r requirements.txt
> ```

**LLM provider is swappable, not hardcoded.** `app/llm_provider.py` is the
single factory the orchestrator, RAG agent, and Operations agent all get
their `llm` from — none of them know or care which provider is behind it.
Set `LLM_PROVIDER` in `.env` to `anthropic`, `groq`, or `gemini` (Groq and
Gemini both have free tiers; only the matching API key needs to be set).
Anthropic's Claude is the most reliable at the strict-JSON intent
classification and tool-call parameter extraction the orchestrator relies
on — Groq's Llama-3.3-70b-versatile and Gemini's gemini-2.0-flash both
support tool calling but may occasionally fail to parse, in which case the
orchestrator's `classify_intent` node falls back to a "clarify" response
rather than crashing.

Try it:
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id": "s1", "message": "Do you have vegan pasta?"}'
```

Or run the scripted demo (RAG queries, operations queries, memory follow-up):
```bash
python tests/test_queries.py
```

## RAG design decisions

**Domains implemented (2, as permitted by the spec):** menu/allergen
information, and branch policies (opening hours, birthday/event hosting,
premium catering, refund policy, loyalty program).

**Chunking strategy: one semantic unit per chunk**, not fixed-size/character
windows. Each menu item becomes one chunk (name + description + ingredients +
cooking method + dietary tags + allergens + price + branch availability);
each policy clause becomes one chunk. Rationale: fixed windows risk splitting
a dish's name from its allergen list, or a policy's condition from its
exception — exactly the kind of split that causes a wrong-but-plausible
answer. Since the source data is already structured JSON, chunking at the
natural record boundary keeps every chunk fully self-contained, which
directly improves both retrieval precision and groundedness.

**Embedding model: `all-MiniLM-L6-v2`** (sentence-transformers, local/offline)
over an API-based embedding model. Rationale:
- No per-query API cost or network dependency — matters for an assistant that
  could run at individual branches.
- 384-dim vectors keep the FAISS index small and fast for a knowledge base of
  this size (dozens–low thousands of chunks); no need for the retrieval
  quality (or cost) of a larger API embedding model at this scale.
- Deterministic and reproducible for retrieval-accuracy evaluation (recall@k,
  MRR, nDCG) — no model-version drift from a hosted API between eval runs.
- `app/rag/embeddings.py` defines an `EmbeddingProvider` ABC, so swapping to
  an API model for a larger, longer-form production KB is a one-line change.

**Vector database: FAISS** (`IndexFlatIP` over L2-normalized vectors, i.e.
cosine similarity) over Chroma/Pinecone/etc. Rationale: the KB is small
enough that an exact flat index is fast with no approximate-NN tuning, and it
keeps the whole RAG agent dependency-free and file-based (no external
service to stand up for grading). Swapping to `IndexIVFFlat`/HNSW for a
1M-document scale KB is a one-line change in `vectorstore.py`, isolated from
the retriever/agent above it.

**Retrieval strategy:** top-k dense retrieval (k=4 default) with two filter
passes before truncation: (1) a metadata **branch filter** — drops chunks
that don't apply to the user's stated branch (e.g. a Downtown-only dish isn't
recommended to a Maadi customer) — falling back to unfiltered results only if
the branch filter would otherwise return nothing, so the agent can still say
"not available at your branch" instead of going silent; (2) a **cosine-score
threshold** — chunks below the bar are dropped even if they'd fill out top-k,
so weak matches don't get stretched into an answer.

**Hallucination prevention (two layers):**
1. *Prompt-level*: the RAG agent's system prompt hard-restricts the LLM to
   the retrieved CONTEXT and requires an explicit "I don't know" fallback
   phrase when the context doesn't cover the question.
2. *Post-generation groundedness check*: after generation, the answer text is
   scanned against the **full list of real menu item names** (not just what
   was retrieved). If the answer references a real dish that wasn't part of
   the retrieved context, the answer is discarded and replaced with a safe
   fallback. This directly targets the spec's "must not hallucinate
   nonexistent menu items" requirement, and also catches the subtler case of
   citing a *real* but *unretrieved/unverified* dish.

## Tool simulation (Operations Agent)

Implements all four suggested tools:
`check_table_availability`, `book_table`, `get_today_special`,
`check_loyalty_points` (spec requires at least two).

Tools are plain typed functions returning structured dicts
(`app/tools/operations_tools.py`) — the same shape a real MCP `tool_result`
or REST response would have, so swapping in a real MCP server later only
means changing the call site inside `OperationsAgent`, not the calling
convention used by the orchestrator or the LLM's tool schema.

- `check_table_availability` is **deterministic** (hashed on
  `date+time+branch`) rather than random, so repeated calls with the same
  inputs are reproducible during grading.
- The Operations Agent uses the LLM's native tool-calling (`bind_tools`) to
  both pick the right tool and extract structured parameters from natural
  language, then executes the real function — the LLM never fabricates a
  result, only the *call*; the data always comes from the tool.
- If required parameters are missing (e.g. no date given), the agent is
  instructed to ask a clarifying question rather than invent a value.

## Memory design

Conversation memory uses **LangGraph's checkpointer** (`MemorySaver` for
dev), keyed by a `thread_id` equal to the request's `session_id`. Every
orchestrator node reads/writes a shared `messages` list in graph state, and
LangGraph persists that state across HTTP requests for the same session.

This is what makes a follow-up like:
```
User: Do you have vegan pasta?
Bot:  Yes — Vegan Pasta Primavera, which is vegan and vegetarian...
User: Is it gluten-free too?
```
work correctly: the second message has no explicit subject, so the
orchestrator's intent classifier is given the full message history and
rewrites the query to something self-contained (e.g. "Is Vegan Pasta
Primavera gluten-free?") before routing to the RAG agent.

Swapping `MemorySaver` for a persistent backend (SQLite/Postgres
checkpointer) for real deployments is a one-line change in
`app/memory/memory_store.py`; no agent code needs to change since agents
never touch the checkpointer directly.

## Example queries and outputs

Produced by `python tests/test_queries.py` (requires `ANTHROPIC_API_KEY` and
a built index — this is a template; paste your actual run output here before
submitting):

```
--- Do you have vegan pasta? ---
agent_used: rag
grounded:   True
sources:    ['menu:Vegan Pasta Primavera']
answer:     Yes, we have Vegan Pasta Primavera — penne with roasted seasonal
            vegetables in an olive-oil and herb sauce. It's vegan and
            vegetarian, though it does contain gluten.

--- What are your opening hours on weekends? ---
agent_used: rag
grounded:   True
sources:    ['policy:Weekend Opening Hours']
answer:     On Fridays and Saturdays we're open from 12:00 PM to 1:00 AM.

--- Do you have a table for 2 at the Maadi branch tomorrow at 8pm? ---
agent_used: operations
grounded:   True
sources:    ['tool:check_table_availability']
answer:     Yes, we have tables available at the Maadi branch tomorrow at
            8:00 PM. Would you like me to book one for you?
```

## Assumptions made

- One retail-chain-style KB (2 domains: menu+allergens, policies) rather than
  all 7 listed domains, per the assessment's "pick 2" allowance.
- Three fixed branches (Downtown, Maadi, New Cairo) used consistently across
  menu availability, policies, and operations tools.
- Table availability and loyalty points are simulated in-memory (deterministic
  hash / static dict) rather than backed by a real database, per the
  assessment's "implement functions that simulate server logic" option.
- `ANTHROPIC_API_KEY` is assumed available for the orchestrator's intent
  classifier, the RAG agent's generation step, and the Operations agent's
  tool-calling — all three currently share one `ChatAnthropic` instance for
  simplicity; a production system might use a smaller/cheaper model for
  intent classification specifically.
- Retrieval-accuracy evaluation (recall@k / MRR / nDCG) is not wired up as an
  automated eval harness in this submission, since the assessment did not
  require one explicitly, but `Retriever.retrieve()` returns per-chunk cosine
  scores and source metadata, which is what such a harness would consume.
