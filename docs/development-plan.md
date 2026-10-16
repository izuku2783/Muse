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

Status: COMPLETE

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

Status: COMPLETE (2026-10-07)

Implemented:

- provider-independent interface and MusicBrainz implementation
- deterministic official-release selection
- normalized Pydantic schemas and SQLAlchemy models
- artist, album, track, and genre repositories
- conservative provider-ID, ISRC, and title/artist deduplication
- track/genre and artist/album/track relationship persistence
- idempotent batch and paginated ingestion with bounded retries
- database-first lexical search with MusicBrainz fallback
- isolated schema, provider, repository, ingestion, service, and API tests

Known Stage 2 limitations:

- lexical search is intentionally simple and is not ranked full-text or semantic search
- a canonical entity retains one provider identity until external-ID mapping tables are justified
- artist genres are not populated because the current normalized `ArtistData` contract has no genre field
- `create_all()` remains the development bootstrap; Alembic is planned after model stabilization
- live PostgreSQL and MusicBrainz checks require those external services to be available

## Stage 3 — Music Knowledge Base

Build:

- deterministic artist, album, track, and genre documents
- text cleaning and normalization
- typed metadata extraction
- stable document identifiers
- database-to-corpus assembly
- repeatable JSONL export

Status: COMPLETE (2026-10-07)

Implementation notes:

- documents are generated from canonical database entities rather than raw provider JSON
- content and metadata are separate and deterministic
- one entity produces one atomic Stage 3 document
- missing optional fields are omitted without fabricated facts or `None` text
- corpus output is storage-neutral JSONL so Stage 4 can consume it without coupling document generation to a vector database
- embeddings, semantic chunking, and Qdrant indexing remain Stage 4 work under the authoritative master-plan boundary

Known Stage 3 limitations:

- documents contain only facts available in the normalized Stage 2 database
- MusicBrainz recording-search results often omit genres, so genre documents require genre associations to have been ingested
- entity IDs are stable within a database but are not portable identifiers across independently created databases
- entity documents are intentionally compact; long-source chunking will be designed when external knowledge sources are introduced

## Stage 4 — Embeddings and Vector Store

Build:

- vector-store abstraction
- Qdrant implementation
- configurable embedding model
- metadata filtering
- vector search and metadata-filter contract tests
- repeatable, versioned indexing
- deletion and explicit reindexing

Status: COMPLETE (2026-10-09)

Implemented:

- provider-independent `EmbeddingProvider` contract
- configurable deterministic hashing embedder for development and tests
- provider-independent `VectorStore` contract
- Qdrant collection creation, upsert, vector search, metadata filters, deletion, and counting
- collection-level embedding model, version, and dimension metadata
- deterministic document-to-point IDs
- idempotent corpus indexing and stale-document removal
- explicit collection recreation for incompatible embedding changes
- Docker Compose Qdrant service and in-memory Qdrant integration tests

Known Stage 4 limitations:

- the hashing embedder is a pipeline baseline, not a learned semantic embedding model
- semantic query behavior and retrieval-quality experiments belong to Stage 5 in the authoritative master plan
- payload indexes are not yet necessary for the small development corpus
- Qdrant batch writes are not transactional, but the idempotent indexing command can safely repair an interrupted run
- live Docker integration requires Docker Desktop; the completed automated integration suite uses Qdrant's official in-memory engine

## Stage 5 — Semantic Retrieval

Build:

- learned local embedding provider
- natural-language query embedding
- Qdrant similarity search
- normalized entity, score, and metadata results
- structured metadata-filter forwarding
- manual search command
- small retrieval evaluation dataset
- Recall@K and MRR baseline

Status: COMPLETE (2026-10-09)

Implemented:

- FastEmbed provider using `BAAI/bge-small-en-v1.5` with distinct query and passage embedding paths
- `SemanticRetriever` service independent of HTTP and agent layers
- typed ranked results containing entity type, canonical ID, similarity score, content, and metadata
- document type, entity ID, genre, year, provider, and score-threshold filtering support
- CLI semantic search against the configured Qdrant collection
- curated six-document/five-query evaluation dataset
- deterministic Recall@K and MRR evaluator
- recorded baseline of Recall@3 = 1.00 and MRR = 1.00
- live reindex of the 39-document development corpus and manual semantic-query verification

Known Stage 5 limitations:

- the evaluation dataset is intentionally small and easy; its perfect score is a smoke-test baseline, not evidence of production retrieval quality
- the current MusicBrainz-derived corpus often lacks genres and rich descriptions, limiting semantic evidence
- semantic search is exposed as a Python service and CLI; public API endpoints remain a later API-hardening concern
- retrieval is dense-only; lexical retrieval and rank fusion belong to Stage 6
- there is no context construction or answer generation yet

## Stage 6 — Hybrid Retrieval

Build:

- semantic retrieval
- lexical retrieval
- metadata filtering
- documented candidate fusion

Status: COMPLETE (2026-10-09)

Implemented:

- deterministic in-process BM25 lexical retrieval over Stage 3 knowledge documents
- shared document type, entity ID, genre, release-year, and provider filters for semantic and lexical candidates
- unweighted Reciprocal Rank Fusion with the documented formula `1 / (60 + rank)`
- typed hybrid results containing fusion score, source ranks, source scores, content, and canonical metadata
- deterministic deduplication and tie-breaking across candidate lists
- CLI hybrid search against the live PostgreSQL corpus and Qdrant collection
- semantic-versus-hybrid Recall@K/MRR comparison on the curated retrieval dataset
- recorded hybrid baseline of Recall@3 = 1.00 and MRR = 1.00
- live verification against the 39-document development corpus

Architecture decisions:

- BM25 runs in process because the current corpus has only 39 documents; PostgreSQL full-text search or OpenSearch would add schema and infrastructure before scale justifies it
- current structured constraints are applied during both candidate searches rather than appended as an unranked third list
- rank fusion is unweighted and does not combine incomparable cosine-similarity and BM25 score scales

Known Stage 6 limitations:

- BM25 statistics are rebuilt from the database corpus when the hybrid CLI starts and are not suitable for a large corpus
- the evaluation set remains a small regression smoke test and cannot demonstrate an improvement over the already-perfect semantic baseline
- source lists use a fixed 50-candidate window and RRF constant of 60; neither has been tuned on a representative dataset
- hybrid retrieval is a Python service and CLI, not yet a public HTTP endpoint
- there is no reranking, context construction, generation, RAG, or agent behavior

## Stage 7 — Metadata Filtering

Build:

- deterministic structured filters
- year and genre constraints
- artist exclusions
- duration and popularity ranges
- instrumental, country, and album-type constraints
- SQL/vector metadata filtering wherever possible

Status: COMPLETE (2026-10-09)

Implemented:

- expanded frozen Pydantic filter schema with validation, normalization, deduplication, and ordered-range checks
- include filters for document type, entity ID, genre, release year, duration, popularity, instrumental status, country, album type, and provider
- artist exclusions by canonical database ID or normalized artist name
- knowledge-document metadata for artist name, popularity, instrumental classification, country, and album type
- case-insensitive filter facets stored alongside each Qdrant payload
- Qdrant `must` constraints and `must_not` artist exclusions applied during semantic candidate generation
- equivalent pre-score filtering for the in-process BM25 candidate generator
- semantic and hybrid CLI support for every Stage 7 constraint
- live reindex and filtered-query verification against the 39-document corpus

Architecture decisions:

- positive constraints reject documents with missing values instead of treating unknown metadata as a match
- string filters normalize whitespace and case while result metadata retains its display form
- `instrumental` is derived only when an existing provider supplies `instrumentalness`; values at or above 0.5 are true, and no audio analysis is performed
- Qdrant owns semantic-branch filtering, while the current small-corpus BM25 branch filters deterministic documents in process before scoring

Known Stage 7 limitations:

- MusicBrainz search data frequently omits popularity, instrumentalness, country, and genre values, so filters on those fields may return few or no live results
- artist-name exclusions require an exact normalized name; aliases and fuzzy matching are not included
- duration CLI values are milliseconds rather than natural-language units
- new filterable metadata requires reindexing Qdrant after corpus changes
- the BM25 branch does not yet push filters into SQL because it operates on the 39-document in-memory corpus
- there is no natural-language query parser; callers must construct filters structurally

## Stage 8 — Reranking

Build:

- reranker interface
- configurable cross-encoder/reranker
- hybrid candidate reduction to the final top results
- evaluation proving whether reranking improves retrieval

Status: COMPLETE (2026-10-09)

Implemented:

- provider-independent `Reranker` interface for query-document relevance scoring
- configurable FastEmbed ONNX cross-encoder adapter with lazy model loading
- default `Xenova/ms-marco-MiniLM-L-6-v2` model and explicit model/version provenance
- `RerankedRetriever` that reduces 20 hybrid candidates to a requested top result set
- preserved hybrid, semantic, and lexical ranks/scores in every reranked result
- deterministic handling of equal reranker scores and validation of malformed provider output
- 100-candidate semantic/lexical source windows for the Stage 8 pipeline
- reranked CLI with all Stage 7 structured filters
- side-by-side Recall@K/MRR evaluation and recorded adoption decision
- live reranked-query verification against the 39-document corpus

Evaluation decision:

- hybrid baseline: Recall@3 = 1.00, MRR = 1.00
- reranked baseline: Recall@3 = 1.00, MRR = 1.00
- measured delta: zero on both metrics
- reranking is retained as an opt-in experimental path but is not adopted as the default because the current evaluation does not demonstrate improvement

Known Stage 8 limitations:

- the five-query dataset has a perfect hybrid ceiling and cannot reveal possible ranking gains
- the default cross-encoder is trained for general passage relevance rather than music-specific recommendation quality
- first use downloads an approximately 80 MB model; local inference adds latency compared with hybrid retrieval alone
- reranking scores are model-specific relevance logits and are not calibrated probabilities
- only document text is scored; structured metadata influences candidate filtering but is not separately serialized into the cross-encoder input
- there is no context construction, answer generation, RAG, or agent behavior

## Stage 9 — Basic RAG

Status: COMPLETE (2026-10-15)

Implemented:

- bounded structured evidence construction from reranked results
- stable evidence IDs with canonical document metadata and ranking provenance
- required, separate retrieved-fact, recommendation-reasoning, and uncertainty sections
- Groq Chat Completions generation using a strict closed JSON schema
- Pydantic validation of generated output and rejection of unknown evidence citations
- deterministic insufficient-evidence response that skips the LLM when retrieval is empty
- lazily initialized reusable runtime for semantic retrieval, BM25, reranking, and generation
- `POST /answer` with optional typed Stage 7 filters and semantic score threshold
- provider/configuration failure translation without leaking provider responses or secrets
- focused generator, context, service, runtime, settings, and API tests

Architecture decisions:

- RAG generation is isolated behind a callable boundary that Stage 10 replaces with the provider-independent LLM interface
- strict structured output is used so every expected response section is present before Pydantic validation
- generated factual and recommendation statements must cite supplied evidence; the free-text synthesis may not introduce additional claims
- the runtime is lazy so health and music-search endpoints do not depend on Qdrant, local model loading, or Groq
- embedding and reranking models are reused instead of loaded for every request

Known Stage 9 limitations:

- evidence sufficiency is deterministic only for an empty result set; adaptive evaluation and re-retrieval come later
- citation validation proves that IDs exist, not that the generated wording is semantically entailed by a source
- the first RAG request snapshots the in-process BM25 corpus, so database ingestion currently requires an API restart
- automated tests mock Groq and do not consume external API quota
- retrieval quality remains limited by the small, sparse MusicBrainz-derived development corpus

## Stage 10 — LLM Provider Abstraction

Status: COMPLETE (2026-10-15)

Implemented:

- provider-independent `LLMProvider` interface for chat and typed structured completion
- typed system, user, assistant, and tool messages with role-specific validation
- typed JSON-schema tool definitions, assistant tool calls, and tool-result round trips
- neutral completion responses with content, tool calls, model, finish reason, and token usage
- Groq adapter as the only external LLM HTTP call site
- strict Pydantic JSON-schema conversion and response validation
- per-request model, temperature, and completion-token overrides
- configurable provider, default model, timeout, output cap, retries, and retry backoff
- bounded exponential retry for transport errors and transient HTTP status codes
- numeric `Retry-After` support and distinct configuration, provider, rate-limit, and response errors
- LLM factory used by the RAG runtime
- provider/model/finish-reason/token metadata in successful RAG API responses
- compatibility aliases for the Stage 9 `GROQ_*` model and generation settings

Architecture decisions:

- application services depend on `LLMProvider`; provider wire formats remain in adapters
- structured output accepts a Pydantic response type and returns both validated output and raw completion metadata
- tool calling is represented at the LLM boundary, but tools are not registered or executed before Stage 12
- retries are limited to network failures and transient statuses; deterministic client and validation errors fail immediately
- provider error bodies and API keys are not propagated through public API responses

Known Stage 10 limitations:

- Groq is the only provider implementation, although consumers are provider-independent
- chat completion is synchronous and non-streaming
- there is no fallback provider, circuit breaker, or persistent request ledger
- structured-output validation failures are not automatically regenerated
- token usage is provider-reported and is not yet persisted or converted into cost estimates
- tool definitions and calls are supported, but safe application tools and execution belong to Stage 12
- the default model remains provider-hosted and can be renamed or retired externally

## Stage 11 — Query Understanding

Status: COMPLETE (2026-10-15)

Implemented:

- frozen Pydantic models for music intent, semantic query, hard filters, and preferences
- bounded intents for discovery, research, similarity, and unknown requests
- strict LLM structured extraction through the Stage 10 provider interface
- whitespace normalization and rejection of blank input before any provider call
- explicit decade-to-inclusive-year guidance
- separate qualitative popularity preference instead of invented numeric thresholds
- natural-language filter schema that excludes database IDs and provider internals
- canonical normalization and validation of string and range constraints
- explicit conversion to the existing `VectorSearchFilter` boundary
- provider/model/finish-reason/token provenance on understanding results
- development CLI for direct inspection
- live verification of the master-plan example against configured Groq
- focused schema, service, and malformed-output tests

Architecture decisions:

- query understanding produces data but does not execute retrieval or tools
- semantic descriptions remain search text unless the user explicitly requests a hard filter
- caller-supplied language cannot directly construct database IDs or provider constraints
- strict provider schemas are simplified to Groq's supported JSON Schema subset, then full Pydantic business validation runs locally
- nullable primitive and enum unions are converted to Groq's disambiguated type-array form
- malformed or constraint-invalid output fails as a normalized LLM response error rather than propagating a partial query

Known Stage 11 limitations:

- extraction quality still depends on the configured model and prompt
- only four high-level intents are currently represented
- qualitative popularity preferences are captured but not applied to ranking yet
- query understanding is available as a Python service and CLI, not a public endpoint
- it is not automatically composed with RAG until a later orchestration stage
- there is no conversation context or reference resolution
- each extraction consumes an additional LLM request and tokens

## Stage 12 — Tool Layer

Status: COMPLETE (2026-10-15)

Implemented:

- generic typed `BaseTool` contract with Pydantic input validation and output checking
- LLM-compatible JSON Schema definitions generated from each tool's input model
- explicit `ToolRegistry` allow-list with duplicate-name and unknown-tool rejection
- stable error categories for invalid input, missing tools, duplicate registration, missing entities, and execution failures
- local `search_music` tool that returns canonical database IDs and normalized track data
- canonical `get_track`, `get_artist`, and `get_album` lookup tools
- `semantic_search` tool with the existing typed retrieval filters and ranked evidence output
- optional `search_external_music_provider` tool that returns normalized provider data without persistence
- registry composition factory with external-provider registration controlled by explicit dependency injection
- canonical repository lookups and model-returning search support required by the tools
- focused tests for contracts, registry behavior, database operations, retrieval forwarding, provider isolation, and error normalization

Architecture decisions:

- each tool performs one operation; local search and external-provider search remain separate instead of hiding fallback and persistence behavior
- the registry is an explicit allow-list and never discovers or imports executable functions dynamically
- tool schemas reuse existing music, retrieval, and filter models rather than creating agent-specific data shapes
- local search returns canonical database IDs, while unpersisted external results expose only provider identities
- known tool errors remain specific and stable; unexpected internal failures cross the boundary only as a generic execution error
- tool construction uses dependency injection so tests make no network calls and Stage 13 can own resource lifetime

Known Stage 12 limitations:

- there is no LLM-driven tool selection, execution loop, planning, retry policy, or evidence evaluation; those belong to Stage 13
- preference tools are deferred until Stage 15 defines intentional, inspectable user memory
- similar-track tools are deferred until Stage 19 provides an evaluated similarity engine
- no web-search tool is included because no bounded web knowledge provider exists yet
- the external provider tool is synchronous and inherits MusicBrainz availability and rate limits
- tools are Python application boundaries rather than public HTTP endpoints
- the registry does not yet emit execution traces or metrics; observability belongs to Stage 21

## Stage 13 — Agentic Orchestration

Status: COMPLETE (2026-10-15)

Implemented:

- pinned LangGraph 1.2.15 dependency and compiled `StateGraph` workflow
- explicit graph state for query understanding, planning, results, evidence, limits, traces, and response
- query-understanding node using the Stage 11 structured service
- deterministic retrieval planner that maps safe understood filters into tool inputs
- focused hybrid/reranked `retrieve_music_evidence` tool preserving ranking provenance
- allow-listed primary tool execution through the Stage 12 registry
- conditional external-provider fallback for discovery and similarity requests only
- evidence collection with separate canonical-knowledge and external-provider provenance
- explicit sufficiency gate before generation or additional retrieval
- configurable maximum iterations, tool calls, elapsed time, and evidence count
- one-primary-plus-one-fallback topology that cannot loop indefinitely
- stable high-level tool execution traces without provider or internal exception details
- citation validation for facts and recommendation reasoning
- deterministic insufficient-evidence and generation-failure responses
- reusable lazy runtime with complete provider/client shutdown
- direct development CLI and `POST /agent` endpoint
- focused graph, tool, runtime, settings, limit, failure, and API tests

Architecture decisions:

- LangGraph coordinates existing services and typed tools rather than replacing their business logic
- planning is deterministic after structured query understanding, avoiding an additional unconstrained planner call
- the primary agent path preserves the established semantic + BM25 + metadata filter + cross-encoder pipeline
- external provider search is a single fallback for discovery/similarity, not a source for factual research answers
- external results are not silently persisted or assigned canonical database identities
- graph traces expose node outcomes and stable error categories, never private reasoning or raw provider errors
- evidence survives generation failure so callers can still inspect what was retrieved
- conversation state and checkpoints are intentionally absent until Stage 14

Known Stage 13 limitations:

- elapsed timeout is enforced between synchronous graph nodes; an individual blocking dependency relies on its own client timeout
- the topology allows only one primary retrieval and one external fallback rather than adaptive query reformulation
- the deterministic planner supports the four Stage 11 intents and does not perform open-ended LLM tool selection
- qualitative popularity preferences are retained but are not yet applied to ranking
- external fallback results contain provider metadata rather than indexed research documents
- the agent and legacy RAG endpoints currently initialize separate runtime stacks if both are used in one process
- graph state is request-local with no follow-up reference resolution or persistence
- automated tests use fakes; a live Groq run was attempted but outbound access was blocked by the execution sandbox

## Stage 14 — Conversational State

Status: COMPLETE (2026-10-15)

Implemented:

- frozen typed short-term state for current intent, previous query, resolved semantic query, filters, entities, and recommendations
- ordered entity/recommendation references with explicit positions for ordinal follow-ups
- compact typed conversation context passed into structured query understanding
- follow-up instructions for referential and elliptical constraints with explicit-current-turn precedence
- automatic conversation creation and UUID return on the first agent request
- continuation through an optional conversation ID on later requests
- thread-safe in-memory conversation-store abstraction
- configurable TTL expiration and maximum active-conversation capacity
- least-recently-used capacity eviction and immutable snapshots
- optimistic turn-count checks that reject concurrent lost updates
- state advancement after successful, fallback, and insufficient-evidence turns
- preservation of prior entities/recommendations when a follow-up retrieves no replacement evidence
- inspect and delete endpoints for short-term conversation state
- multi-turn CLI verification path through repeatable `--follow-up` arguments
- focused store, context payload, two-turn resolution, runtime, settings, and API tests

Architecture decisions:

- conversation state wraps the Stage 13 graph instead of placing arbitrary message history in graph state
- only the compact previous-turn context needed for reference resolution is retained
- query understanding resolves context into a standalone semantic query and complete current filters before planning
- explicit current constraints override prior constraints; clearly independent requests should not inherit them
- ordered results carry an explicit position for references such as "the third result"
- optimistic concurrency prevents silent overwrites without holding a lock across network and model calls
- short-term state is anonymous and isolated from future durable user preferences

Known Stage 14 limitations:

- state is in memory, so it is lost on restart and is not shared across API workers or instances
- reference-resolution quality still depends on the configured LLM following the context instructions
- only the latest compact turn state is retained, not a full transcript or multi-turn summary
- expired or capacity-evicted conversations cannot be resumed
- concurrent requests for the same conversation may perform work before one is rejected with a 409 conflict
- no authenticated ownership is attached to conversation IDs yet; authentication and security hardening are later stages
- no long-term preferences are learned or stored; that work begins in Stage 15

## Stage 15 — User Memory

Status: COMPLETE (2026-10-16)

Implemented:

- durable SQLAlchemy `user_preferences` model registered with database initialization
- user-scoped category/value identity with normalized duplicate prevention
- bounded categories for artist, genre, popularity, context, and mood preferences
- bounded stances for like, dislike, prefer, avoid, and frequent behavior
- provenance fields for explicit user input, future provider imports, and inferred behavior
- confidence, creation timestamp, update timestamp, and optimistic version on every record
- transactional repository and service boundaries for list, create, update, and delete
- deterministic user/category filtering and profile ordering
- explicit public preference CRUD endpoints
- server-controlled `explicit_user` source and confidence `1.0` for public writes
- conditional version checks for update/delete conflict detection
- user ownership scoping on every preference lookup and mutation
- read-only typed `get_user_preferences` tool
- validation for normalized text and constrained popularity preferences
- focused persistence, conflict, isolation, tool, validation, and API tests

Architecture decisions:

- durable memory stores normalized preferences, never arbitrary chat transcripts
- no conversation or agent path automatically writes memory
- public callers cannot select provenance or confidence; those fields are assigned by trusted service boundaries
- preference category and normalized value form the stable identity, while stance can be updated
- updates and deletes require an expected version to prevent silent lost writes
- the agent-facing tool is read-only; durable mutation remains an explicit API action
- memory storage is complete here, while applying it to ranking is reserved for Stage 17

Known Stage 15 limitations:

- user UUIDs are caller-provided and are not yet backed by authentication or a user-account model
- the service defines provider/inferred provenance for future trusted integrations, but no automated importer or inference workflow exists
- categories and stances are intentionally small and may need measured expansion
- preference values are normalized strings rather than canonical artist/genre foreign keys
- no audit-history table retains superseded preference versions
- preferences are not yet included in query planning, retrieval, reranking, novelty, or diversity scoring
- schema creation still uses SQLAlchemy `create_all`; production migrations are not yet configured

## Stage 16 — Spotify Integration

Status: COMPLETE (2026-10-16)

Implemented:

- optional Spotify integration that does not affect application startup or core routes
- Authorization Code with PKCE, exact configured redirect URI, and no client secret
- cryptographically random, one-use OAuth state and verifier with bounded expiration
- minimal `user-top-read` and `user-read-recently-played` scopes
- stable account linkage using Spotify's immutable pseudonymous `account_id`
- encrypted access/refresh-token persistence using an operator-supplied Fernet key
- automatic access-token refresh and retention of the prior refresh token when Spotify does not rotate it
- credential removal and explicit reauthorization state after an invalid refresh grant
- connection start, callback, status, disconnect, top-track, and recent-playback endpoints
- typed Spotify profile, track, album, artist, and playback boundaries
- provider IDs explicitly exposed as `spotify_id`, never as canonical Muse IDs
- generic external provider mappings to existing canonical artist, album, and track IDs
- no automatic persistence of Spotify listening content or provider mappings
- normalized configuration, authorization, provider, rate-limit, conflict, and not-found errors
- bounded provider limits and propagation of numeric `Retry-After` guidance
- provider, PKCE state, persistence, encryption, refresh, mapping, API, and optionality tests

Architecture decisions:

- Spotify lives behind `SpotifyIntegration` and `SpotifyService`; core retrieval, memory, and agent modules do not import it
- PKCE avoids introducing a client secret and is compatible with a future browser frontend
- OAuth state stores the Muse user association server-side and is consumed exactly once
- only tokens and minimum account-link metadata are persisted; listening responses are fetched on demand
- tokens are never stored in plaintext and the encryption key is required only when Spotify is enabled
- stable `account_id` is used for account linkage while public Spotify IDs remain provider-local
- provider mappings require a pre-existing canonical Muse entity and are never inferred from a Spotify ID alone
- Stage 16 exposes permitted signals but does not turn them into preferences or ranking features

Known Stage 16 limitations:

- Muse user UUIDs are still caller-provided and are not authenticated
- OAuth state is process-local, so callback routing must reach the same process and pending flows do not survive restarts
- Spotify Development Mode has current Premium-owner, authorized-user, and quota constraints and is not a general public-production entitlement
- Spotify refresh tokens currently have a finite provider-controlled lifetime and eventually require reauthorization
- there is no playlist mutation, playback control, library mutation, or bulk Spotify ingestion
- Spotify top/recent data is not cached, stored as memory, or used for personalization until Stage 17
- external entity mappings exist but require an explicit future reconciliation workflow
- schema creation still uses `create_all`; production migrations remain outstanding

## Stage 17 — Personalized Retrieval

Status: COMPLETE (2026-10-16)

Implemented:

- optional `PersonalizedRetriever` after hybrid retrieval and cross-encoder reranking
- exact preservation of the existing retrieval path when no personalization input exists
- optional `user_id` on `POST /agent`
- request-local loading of durable explicit preferences
- optional, bounded Spotify top-track and recent-playback signal collection
- provider failure isolation with fallback to explicit-only or ordinary retrieval
- canonical known-track resolution through provider mappings and ISRC matching
- normalized familiar-artist matching without making Spotify IDs canonical
- bounded explicit preference scoring for artist, genre, popularity, mood, and context
- current-query qualitative popularity scoring without a durable memory write
- separate provider affinity and novelty components
- deterministic greedy artist/genre diversity adjustment
- normalized relevance based on rank rather than uncalibrated cross-encoder logits
- inspectable per-result score components and concise personalization reasons
- personalization provenance included in agent evidence and grounded-generation context
- configurable global scoring weights validated to sum to one
- deterministic evaluation metrics for preference hit, avoidance, novelty, artist diversity, and relevance retention
- reproducible comparison script and recorded Stage 17 baseline
- focused ranker, profile provider, Spotify signal, retriever, agent, API, configuration, and evaluation tests

Architecture decisions:

- personalization is a post-reranking layer, so candidate generation and hard filters remain deterministic and provider-independent
- explicit preferences remain the authoritative inspectable memory; provider listening data is transient and never copied into memory automatically
- provider failures remove that signal source rather than failing the recommendation request
- provider mappings and ISRCs identify known canonical tracks; normalized artist names provide weaker familiarity only
- preference affinity and novelty remain separate so familiarity does not automatically mean repetition
- diversity is a greedy selection adjustment over artist and genre overlap, not an LLM judgment
- score breakdowns are public reasoning summaries, not private chain-of-thought
- the balanced weights are configurable development defaults supported by a small benchmark, not permanently fixed production values

Recorded Stage 17 baseline (`k=3`, three deterministic cases):

- preference hit rate: `0.0 -> 1.0`
- avoidance success rate: `0.0 -> 1.0`
- mean artist diversity: `0.7778 -> 0.8889`
- mean novelty: unchanged at `0.7778`
- mean relevance retention: `0.9506` relative to relevance-only ranking

Known Stage 17 limitations:

- the benchmark is small and synthetic; it does not establish real-user recommendation quality
- weights are global rather than learned or adapted per user/query
- Spotify signals require two on-demand provider reads per personalized request and are not cached yet
- only canonical knowledge candidates are personalized; external-provider fallback results are not reranked with user signals
- context and mood matches use conservative normalized text/token matching
- explicit avoids are soft ranking penalties rather than hard exclusions
- missing popularity/ISRC/artist metadata limits the applicable components
- diversity considers artist and genre overlap only and uses greedy selection
- external mappings are still populated explicitly rather than by an automated reconciliation workflow
- user UUIDs remain caller-provided without authentication
- personalization reads memory and provider signals but never infers or writes new preferences

## Stage 18 — Audio Feature Layer

Introduce normalized audio features only after metadata-based retrieval is stable.

## Stage 19 — Music Similarity Engine

Build and evaluate similarity using the available metadata and audio representations.

## Stage 20 — Evaluation Framework

Expand repeatable retrieval, RAG, agent, recommendation, and engineering evaluations.

## Stage 21 — Observability

Add structured traces, latency and token measurements, and high-level execution events.

## Stage 22 — Caching / Performance

Add measured caching and performance improvements at justified bottlenecks.

## Stage 23 — Backend API Hardening

Harden public endpoints, validation, errors, rate limits, and API documentation.

## Stage 24 — Frontend

Build the Next.js chat, recommendation, research-source, and integration interfaces.

## Stage 25 — Feedback System

Capture explicit user feedback for later evaluation and recommendation improvement.

## Stage 26 — Security

Review authentication, secrets, data boundaries, abuse controls, and dependency risks.

## Stage 27 — Production Containerization

Create production images, health checks, configuration, and persistent-service topology.

## Stage 28 — CI/CD

Automate verification and controlled build/deployment workflows.

## Stage 29 — Deployment

Deploy the complete application and document operations, backups, and recovery.

## Checkpoint Rule

After each stage:

1. Run tests.
2. Verify existing functionality.
3. Update documentation.
4. Report changes and known issues.
5. Stop for explicit approval before the next major stage.
