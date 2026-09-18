## Updated Architecture (Agentic Loop)

┌───────────────────────┐
│   Client / Web UI     │
└───────────┬───────────┘
            │
            ▼
┌─────────────────────────────────────┐
│       FastAPI Backend (Uvicorn)     │
│       Rate Limiting: SlowAPI        │
└───────────────┬─────────────────────┘
                │
                ▼
      ┌───────────────────┐
      │  ChromaDB (RAG)   │  <-- Initial Context Retrieval
      └─────────┬─────────┘
                │
                ▼
 ╭─────────────────────────────╮
 │       AGENTIC LOOP          │
 │                             │
 │  ┌───────────────────────┐  │
 │  │ Google GenAI (Gemini) │  │
 │  │  (Decision Engine)    │  │
 │  └───────┬───────▲───────┘  │
 │          │       │          │
 │   [SEARCH:]   [Context      │
 │          │   Compaction]    │
 │          ▼       │          │
 │  ┌───────────────────────┐  │
 │  │    Local Tools /      │  │
 │  │   Mock Web Search     │  │
 │  └───────────────────────┘  │
 ╰──────────────┬──────────────╯
                │
            [ANSWER:]
                │
                ▼
      ┌───────────────────┐
      │   JSON Response   │
      └───────────────────┘