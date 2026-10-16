# Muse

Muse is an AI Music Research & Discovery Agent designed to combine:

- Agentic AI
- RAG
- Semantic Search
- Hybrid Retrieval
- Metadata Filtering
- Reranking
- User Memory
- Personalized Recommendations
- Music Research
- Evaluation
- Observability

Muse is designed as a provider-independent system.

Spotify is an optional integration that can provide additional user-specific context and personalization. The core Muse intelligence system does not depend on Spotify.

## Current Status

Stage 17 (Personalized Retrieval) is complete. The backend currently provides:

- normalized artist, album, track, and genre schemas;
- a provider-independent music interface and MusicBrainz adapter;
- deterministic canonical-release selection;
- SQLAlchemy repositories with conservative deduplication;
- idempotent, paginated ingestion with bounded retries;
- database-first music search with provider fallback;
- deterministic artist, album, track, and genre knowledge documents;
- typed document metadata and stable document IDs;
- repeatable database-to-JSONL corpus export;
- a provider-independent embedding interface and deterministic baseline embedder;
- a Qdrant vector-store adapter with metadata filtering;
- model/version/dimension compatibility enforcement;
- idempotent indexing, stale-document deletion, and explicit reindexing;
- local learned embeddings through FastEmbed and `BAAI/bge-small-en-v1.5`;
- semantic entity retrieval with similarity scores and structured filters;
- deterministic BM25 lexical retrieval over the knowledge corpus;
- reciprocal-rank fusion of semantic and lexical candidate lists;
- typed, validated metadata filtering across both candidate generators;
- artist exclusion, duration, popularity, instrumental, country, and album-type constraints;
- a configurable local cross-encoder reranker with candidate provenance;
- side-by-side hybrid/reranked evaluation with an explicit adoption decision;
- a curated Recall@K/MRR evaluation dataset and reproducible baseline;
- bounded structured evidence with stable citation IDs;
- Groq-backed strict structured answer generation;
- a provider-independent chat-completion and structured-output interface;
- typed tool definitions, tool calls, and tool-result messages;
- bounded retries for transient provider and network failures;
- provider/model overrides and provider-reported token usage;
- validated music intent, semantic-query, filter, and preference extraction;
- safe conversion from natural-language constraints to retrieval filters;
- an explicit allow-list registry of small, typed application tools;
- local catalog search and canonical track, artist, and album lookup tools;
- semantic knowledge search with typed metadata filters;
- external-provider search that does not implicitly persist results;
- stable validation, not-found, registration, and execution errors;
- a bounded LangGraph workflow for query understanding, planning, tool execution,
  evidence evaluation, optional external fallback, and grounded generation;
- configurable iteration, tool-call, evidence, and elapsed-time limits;
- safe high-level tool execution traces and evidence-preserving response fallbacks;
- bounded short-term conversation state with expiring conversation IDs;
- contextual follow-up resolution using the prior intent, query, filters, entities,
  recommendation order, and semantic target;
- inspectable and deletable conversation state kept separate from user memory;
- durable, explicit user preference records with category, stance, provenance,
  confidence, timestamps, and optimistic versions;
- inspectable create/list/update/delete preference APIs and a read-only preference tool;
- normalized duplicate prevention without automatic conversion of chat into memory;
- optional Spotify Authorization Code with PKCE and one-use expiring OAuth state;
- encrypted persistent Spotify tokens with refresh and explicit reauthorization state;
- user-scoped connection status, disconnect, top-track, and recent-playback APIs;
- stable Spotify `account_id` linkage and separate provider-to-canonical-entity mappings;
- clearly provider-local Spotify item IDs with no implicit catalog persistence;
- optional post-reranking personalization activated by a Muse user UUID;
- explainable relevance, explicit-preference, provider-affinity, novelty, and diversity scores;
- on-demand fusion of durable preferences with permitted Spotify top/recent signals;
- graceful fallback to explicit-only or ordinary retrieval when Spotify is unavailable;
- configurable scoring weights with a reproducible deterministic baseline;
- explicit retrieved-fact, recommendation-reasoning, uncertainty, and source sections;
- rejection of generated citations that do not refer to supplied evidence; and
- `GET /health`, `POST /search`, `POST /answer`, and `POST /agent` API endpoints.

Stage 18 and later work (audio analysis, similarity, expanded evaluation, observability,
and the frontend) has not started. Reranking remains
opt-in for standalone search because the current
evaluation did not measure an improvement over the perfect hybrid baseline; the RAG
path uses the reranker required by the master pipeline.

## Local Development

Install dependencies, start PostgreSQL and Qdrant, then run the API from PowerShell:

```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt
docker compose up -d postgres qdrant
$env:PYTHONPATH="backend"
.venv\Scripts\python.exe -m uvicorn app.main:app --reload
```

To enable generated RAG answers, create a Groq API key and set `GROQ_API_KEY` in
`.env`. Provider-neutral settings use the `LLM_*` prefix; the default model is
`qwen/qwen3.8-27b`, which supports the strict JSON-schema response used by Muse.
Keep provider keys on the backend and never expose them to browser code. The previous
`GROQ_MODEL`, `GROQ_TIMEOUT_SECONDS`, and `GROQ_MAX_COMPLETION_TOKENS` environment
names remain accepted as compatibility aliases.

Run the test suite:

```powershell
.venv\Scripts\python.exe -m pytest -q
```

Ingest one or more pages of normalized MusicBrainz results:

```powershell
$env:PYTHONPATH="backend"
.venv\Scripts\python.exe scripts\ingest_music.py "dark synthwave" --max-pages 2
```

Build the deterministic Stage 3 knowledge corpus from the current database:

```powershell
$env:PYTHONPATH="backend"
.venv\Scripts\python.exe scripts\build_knowledge_corpus.py
```

The default output is `data/knowledge/documents.jsonl`. Generated corpus files
are local build artifacts and are excluded from Git.

Index the current database corpus in Qdrant:

```powershell
$env:PYTHONPATH="backend"
.venv\Scripts\python.exe scripts\index_knowledge.py
```

If the configured embedding model, version, or dimension changes, recreate the
collection explicitly:

```powershell
.venv\Scripts\python.exe scripts\index_knowledge.py --recreate
```

The local Qdrant dashboard is available at http://localhost:6333/dashboard.
The default learned model is downloaded to `data/models` on first use. The hashing
embedder remains available for dependency-light pipeline tests.

Search the indexed corpus semantically:

```powershell
$env:PYTHONPATH="backend"
.venv\Scripts\python.exe scripts\search_semantic.py `
    "dark atmospheric electronic music" `
    --limit 5
```

Optional filters include `--type`, `--entity-id`, `--genre`, `--year-min`,
`--year-max`, `--provider`, and `--score-threshold`, plus the Stage 7 filters
listed below for hybrid search.

Run the curated Stage 5 retrieval evaluation:

```powershell
.venv\Scripts\python.exe scripts\evaluate_semantic_retrieval.py --k 3
```

Search the current PostgreSQL/Qdrant corpus with Stage 6 hybrid retrieval:

```powershell
$env:PYTHONPATH="backend"
.venv\Scripts\python.exe scripts\search_hybrid.py `
    "dark atmospheric electronic music" `
    --limit 5
```

The hybrid command supports `--type`, `--entity-id`, `--genre`, `--year-min`,
`--year-max`, `--exclude-artist-id`, `--exclude-artist`, `--duration-min-ms`,
`--duration-max-ms`, `--popularity-min`, `--popularity-max`, `--instrumental`,
`--no-instrumental`, `--country`, `--album-type`, `--provider`, and
`--semantic-score-threshold`. It runs semantic and BM25 candidate generation
and combines their ranks with reciprocal rank fusion.

For example:

```powershell
.venv\Scripts\python.exe scripts\search_hybrid.py `
    "dark synthwave album" `
    --type album `
    --album-type album `
    --year-min 2020 `
    --exclude-artist "Karl Casey"
```

Positive filters exclude documents where the requested metadata is unavailable.
Artist names, genres, countries, album types, and providers are matched
case-insensitively.

Compare semantic-only and hybrid retrieval on the curated dataset:

```powershell
.venv\Scripts\python.exe scripts\evaluate_hybrid_retrieval.py --k 3
```

Run Stage 8 cross-encoder reranking over the top 20 hybrid candidates:

```powershell
$env:PYTHONPATH="backend"
.venv\Scripts\python.exe scripts\search_reranked.py `
    "dark atmospheric electronic music" `
    --limit 5
```

The first run downloads the configured MiniLM ONNX model into `data/models`.
All Stage 7 filter options accepted by `search_hybrid.py` are also accepted by
`search_reranked.py`.

Compare hybrid and reranked retrieval:

```powershell
.venv\Scripts\python.exe scripts\evaluate_reranking.py --k 3
```

The recorded five-query baseline has the same Recall@3 and MRR for both paths.
Consequently, hybrid retrieval remains the default and reranking is an explicit,
experimental command until a larger evaluation set demonstrates improvement.

After indexing the corpus and configuring Groq, request a grounded answer through
Swagger at http://127.0.0.1:8000/docs or call `POST /answer`:

```json
{
  "query": "Recommend dark synthwave and explain the evidence",
  "filters": {
    "genres": ["synthwave"]
  },
  "semantic_score_threshold": 0.2
}
```

The response keeps its synthesis, cited facts, cited recommendation reasoning,
uncertainty, and complete evidence sources separate. If retrieval finds no evidence,
Muse returns a deterministic insufficient-evidence response without calling Groq.
Generated responses also report the provider, returned model, finish reason, and token
usage when the provider supplies it.
The RAG runtime snapshots the lexical corpus on its first request; restart the API
after ingesting or changing database records so that BM25 sees the updated corpus.

Inspect Stage 11 structured query understanding independently:

```powershell
$env:PYTHONPATH="backend"
.venv\Scripts\python.exe scripts\understand_query.py `
    "Give me dark electronic music from the 2010s but nothing too popular."
```

The result contains a validated intent, semantic query, explicit hard filters,
qualitative preferences, provider/model provenance, and token usage. Query
understanding does not execute retrieval or tools; those orchestration steps remain
later stages.

Run the independently testable Stage 12 tool checks:

```powershell
.venv\Scripts\python.exe -m pytest `
    backend/tests/test_tool_base.py `
    backend/tests/test_music_tools.py `
    backend/tests/test_retrieval_tool.py `
    backend/tests/test_tool_factory.py -q
```

The registry exposes `search_music`, `get_track`, `get_artist`, `get_album`, and
`semantic_search`. It adds `search_external_music_provider` only when an external
provider is explicitly supplied. Stage 12 does not let an LLM choose or chain tools;
that bounded orchestration begins in Stage 13.

Run the Stage 13 graph directly:

```powershell
$env:PYTHONPATH="backend"
.venv\Scripts\python.exe scripts\run_agent.py `
    "Recommend dark synthwave tracks."
```

Or call `POST /agent` through Swagger at http://127.0.0.1:8000/docs:

```json
{
  "query": "Recommend dark synthwave tracks."
}
```

The graph understands the query, creates a deterministic retrieval plan, invokes the
allow-listed hybrid/reranked evidence tool, evaluates sufficiency, optionally queries
the external music provider once for discovery requests, and generates a citation-
validated answer. The Stage 14 conversation wrapper supplies only compact prior-turn
state when a client continues with the returned conversation ID.

Run a two-turn conversation from the CLI:

```powershell
$env:PYTHONPATH="backend"
.venv\Scripts\python.exe scripts\run_agent.py `
    "Give me Radiohead-like albums." `
    --follow-up "Only ones after 2010."
```

For HTTP clients, omit `conversation_id` on the first `POST /agent` request. The response
contains `conversation.conversation_id`; include that UUID in the next request:

```json
{
  "query": "Only ones after 2010.",
  "conversation_id": "00000000-0000-0000-0000-000000000000"
}
```

Use `GET /agent/conversations/{conversation_id}` to inspect short-term state and
`DELETE /agent/conversations/{conversation_id}` to clear it. State expires after 30
minutes by default and is process-local. It stores only the compact previous-turn fields
needed for reference resolution and remains separate from the durable preferences below.

Manage explicit long-term preferences through `/users/{user_id}/preferences`:

```json
{
  "category": "artist",
  "value": "Radiohead",
  "stance": "like"
}
```

- `POST /users/{user_id}/preferences` creates an explicit preference.
- `GET /users/{user_id}/preferences` lists inspectable memory.
- `PATCH /users/{user_id}/preferences/{preference_id}` updates it using
  `expected_version`.
- `DELETE /users/{user_id}/preferences/{preference_id}?expected_version=1` removes it.

Explicit API writes are recorded with source `explicit_user` and confidence `1.0`.
Conversation turns are never copied into durable memory automatically. Preferences are
stored and available through `get_user_preferences`. Supplying the same `user_id` to
`POST /agent` lets Stage 17 apply them to ranking without mutating the profile.

### Optional Spotify connection

Muse works normally when Spotify is not configured. To enable it, create an app in the
Spotify Developer Dashboard and register the exact callback URI shown below. Spotify
requires HTTPS except for an explicit loopback address; `localhost` is not accepted.

Generate a token-encryption key once:

```powershell
.venv\Scripts\python.exe -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```

Put the following values in `.env` (never put the generated key in source control):

```dotenv
SPOTIFY_CLIENT_ID=your-client-id
SPOTIFY_REDIRECT_URI=http://127.0.0.1:8000/integrations/spotify/callback
SPOTIFY_TOKEN_ENCRYPTION_KEY=your-generated-fernet-key
```

The integration uses Authorization Code with PKCE and therefore does not require a
Spotify client secret. To verify it with a caller-controlled Muse user UUID:

1. Call `POST /users/{user_id}/integrations/spotify/authorize` in Swagger.
2. Open the returned `authorization_url` and approve the two requested read scopes.
3. Spotify redirects to the callback and returns the connected status.
4. Inspect `GET /users/{user_id}/integrations/spotify`.
5. Try `GET .../top-tracks` and `GET .../recently-played`, then disconnect with
   `DELETE /users/{user_id}/integrations/spotify`.

Only `user-top-read` and `user-read-recently-played` are requested. Returned listening
data is read on demand, attributed to Spotify, and is not persisted or used for
automatic memory writes. Stage 17 can use these signals transiently for personalized
ranking. Spotify item IDs are labeled `spotify_id` and never act as canonical Muse IDs.

Spotify Development Mode is suitable for a small demonstration, not unrestricted public
access: current accounts and authorized-user limits apply. Refresh tokens also expire
and eventually require the user to authorize again. See Spotify's current
[quota modes](https://developer.spotify.com/documentation/web-api/concepts/quota-modes),
[PKCE flow](https://developer.spotify.com/documentation/web-api/tutorials/code-pkce-flow),
and [refresh-token lifecycle](https://developer.spotify.com/documentation/web-api/tutorials/refreshing-tokens).

### Personalized retrieval

Add `user_id` to an agent request to personalize its canonical retrieval results:

```json
{
  "query": "Recommend atmospheric electronic tracks, but keep it varied.",
  "user_id": "00000000-0000-0000-0000-000000000001"
}
```

The agent loads that user's explicit preferences. If the same user has a usable Spotify
connection, it also reads bounded top-track and recent-playback signals without storing
the listening response. Each returned knowledge source can include a `personalization`
breakdown containing normalized relevance, explicit preference, provider affinity,
novelty, diversity penalty, final score, and concise reasons. Omitting `user_id`
preserves the existing ordering unless the current query explicitly requests a
qualitative popularity preference.

The development weights are configurable through the `PERSONALIZATION_*` environment
variables and must sum to `1.0`. Reproduce the small Stage 17 comparison with:

```powershell
$env:PYTHONPATH="backend"
.venv\Scripts\python.exe scripts\evaluate_personalization.py
```

The recorded baseline is in `evaluation/personalization_baseline.json`. It supports the
current development defaults but is intentionally synthetic and too small to establish
real-user recommendation quality.

MusicBrainz access requires network availability and is rate limited by the
provider adapter. Normal unit and API tests use fakes and make no external
requests.

## Architecture

```text
User
 |
 v
Next.js Web App
 |
 v
FastAPI
 |
 v
LangGraph Agent
 |
 +--------------------+
 |         |          |
 v         v          v
RAG     Music Data   Memory
 |         |          |
 v         v          v
Qdrant  PostgreSQL  PostgreSQL
           |
           v
 Music Providers
      /           \
 Spotify       Other Sources
```
