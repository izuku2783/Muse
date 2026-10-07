# Muse — Development Plan

## Development Rules

- Build incrementally.
- Complete and test each stage before moving to the next.
- Do not introduce an LLM where deterministic code is more appropriate.
- Keep every major component independently testable.
- Do not expose chain-of-thought.
- Do not build the frontend before the backend AI pipeline is functional.
- Treat Spotify as an optional integration, not the core recommendation engine.
- Do not introduce audio/multimodal processing until the RAG + agent system is stable.

## Stage 0 — Specification and Architecture

Deliverables:

- `docs/architecture.md`
- `docs/development-plan.md`
- `README.md`
- repository conventions

Status: COMPLETE

## Stage 1 — Project Foundation

Build:

- Python backend
- FastAPI
- Pydantic settings
- environment configuration
- structured logging
- PostgreSQL connection
- Docker Compose
- `/health`
- initial tests

Checkpoint: backend starts, configuration loads, database connection works, health test passes.

## Stage 2 — Music Data Layer

Build:

- provider interface
- PostgreSQL music schema
- normalized music models
- initial provider implementation
- ingestion script
- validation and deduplication
- provider error handling

Checkpoint: `search_music("dark synthwave")` returns normalized structured records.

## Stage 3 — Music Knowledge Base

Build:

- document ingestion
- cleaning
- metadata extraction
- chunking
- embedding generation
- Qdrant indexing

## Stage 4 — Embeddings and Vector Search

Build:

- vector-store abstraction
- Qdrant implementation
- configurable embedding model
- metadata filtering
- retrieval tests
- retrieval experiments

## Stage 5 — Basic RAG

Build:

- retrieval service
- context construction
- prompt construction
- generation service
- grounded answers
- source references

No agentic behavior yet.

## Stage 6 — Hybrid Retrieval

Build:

- semantic retrieval
- structured metadata filtering
- deterministic numerical/date/popularity constraints
- combined candidate generation

## Stage 7 — Natural-Language Query Understanding

Build:

- Pydantic query schema
- LLM query parser
- validation
- query-understanding tests

## Stage 8 — Candidate Generation and Reranking

Build:

- candidate pool
- deterministic filtering
- configurable reranker
- ranking signals
- top-N recommendations
- recommendation explanations

## Stage 9 — Agent Tools

Build and test the core music, knowledge, web, and preference tools.

Add Spotify-specific tools/services only where they support an actual product workflow.

## Stage 10 — Agent Architecture

Build:

- LangGraph state
- intent node
- planning node
- tool selection
- tool execution
- result evaluation
- response generation

## Stage 11 — Agentic RAG

Build adaptive retrieval:

- retrieve
- evaluate evidence
- rewrite query when necessary
- retrieve again
- strict maximum iterations

## Stage 12 — User Memory

Build:

- short-term conversation state
- intentional long-term preference storage
- reference resolution such as "more like #3"

## Stage 13 — Personalized Recommendation

Combine user memory with retrieval and reranking.

Add Spotify-derived preference signals only when the user has explicitly connected Spotify and the required data is available.

## Stage 14 — Evaluation Harness

Build repeatable datasets and metrics for:

- retrieval
- RAG
- agent behavior
- recommendation quality
- engineering performance

## Stage 15 — Failure Analysis

Categorize failures and document:

`baseline → problem → hypothesis → change → evaluation → result`

## Stage 16 — Observability

Build structured logs, traces, latency/token tracking, and development execution traces.

## Stage 17 — FastAPI API

Expose:

- `POST /chat`
- `POST /search`
- `POST /recommend`
- `GET /tracks/{id}`
- `GET /artists/{id}`
- `GET /users/{id}/preferences`
- `POST /users/{id}/preferences`
- `POST /evaluation/run`
- `GET /health`

Add Spotify OAuth endpoints only after the core backend is stable enough to support them cleanly.

## Stage 18 — Frontend

Build the Next.js application with:

- chat
- recommendations
- research results
- source display
- Spotify connection UI
- high-level execution trace

## Stage 19 — Multimodal Audio Extension

Add audio analysis, audio embeddings/features, and multimodal retrieval.

## Stage 20 — Productionization

Add:

- containerization
- health checks
- retries
- timeouts
- validation
- rate limiting where appropriate
- caching where useful
- environment-specific configuration

## Stage 21 — Documentation

Complete README, architecture documentation, API documentation, evaluation results, failure analysis, setup instructions, and architecture decision records.

## Checkpoint Rule

After each stage:

1. Run tests.
2. Verify existing functionality.
3. Update documentation.
4. Report changes and known issues.
5. Stop for explicit approval before the next major stage.
