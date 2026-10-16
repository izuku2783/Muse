# Muse — System Architecture

## 1. Product Definition

Muse is a web application for AI-powered music research and discovery.

Users can use Muse without connecting a music account. An optional Spotify connection enables personalization using permitted Spotify user data.

Spotify is an integration/provider, not the core recommendation engine.

## 2. Architectural Goals

Muse is designed to demonstrate:

- Agentic AI
- RAG
- LLM tool calling
- Semantic search
- Hybrid retrieval
- Metadata filtering
- Candidate generation and reranking
- User memory and conversational state
- Evaluation
- Observability
- Music metadata and later audio analysis
- Provider-independent architecture

The system must remain modular and should not be tightly coupled to a particular LLM provider, music provider, vector database, or embedding model.

## 3. High-Level Architecture

```text
                           ┌──────────────────────┐
                           │      Next.js Web UI  │
                           │                      │
                           │ Chat / Discovery /   │
                           │ Research / Spotify   │
                           │ Connection           │
                           └──────────┬───────────┘
                                      │
                                      ▼
                           ┌──────────────────────┐
                           │      FastAPI API     │
                           └──────────┬───────────┘
                                      │
                                      ▼
                           ┌──────────────────────┐
                           │    Muse Agent        │
                           │      LangGraph       │
                           │                      │
                           │ Intent → Plan →      │
                           │ Tools → Evaluate →   │
                           │ Retrieve → Rank →    │
                           │ Respond              │
                           └──────────┬───────────┘
                                      │
             ┌────────────────────────┼─────────────────────────┐
             │                        │                         │
             ▼                        ▼                         ▼
   ┌──────────────────┐    ┌──────────────────┐      ┌──────────────────┐
   │ Music Provider    │    │ Knowledge RAG    │      │ User Memory      │
   │ Abstraction       │    │                  │      │                  │
   └────────┬─────────┘    └────────┬─────────┘      └────────┬─────────┘
            │                       │                         │
     ┌──────┴──────┐                ▼                         ▼
     │             │         ┌──────────────┐        ┌────────────────┐
     ▼             ▼         │   Qdrant     │        │  PostgreSQL    │
  Spotify      Other/Local   └──────────────┘        └────────────────┘
  Provider      Provider

                           Retrieval Pipeline
                                  │
                     ┌────────────┴────────────┐
                     ▼                         ▼
               Semantic Retrieval       Structured Filters
                     │                         │
                     └────────────┬────────────┘
                                  ▼
                           Candidate Set
                                  │
                                  ▼
                              Reranker
                                  │
                                  ▼
                         Personalized Results
                                  │
                                  ▼
                         Grounded LLM Response
```

## 4. Core Components

### 4.1 Web Application

Next.js, React, and Tailwind CSS provide the user-facing application.

Initial UI priorities:

1. Natural-language chat/discovery interface
2. Recommendation result cards
3. Research answers with sources
4. Optional Spotify connection
5. High-level development trace

The frontend is intentionally deferred until the backend AI pipeline is stable.

### 4.2 FastAPI Backend

FastAPI exposes the application API while keeping business logic outside route handlers.

The agent must also be callable directly as a Python service without HTTP.

### 4.3 Agent Orchestrator

LangGraph manages explicit agent state and workflow transitions.

Conceptual flow:

```text
START
  → intent analysis
  → planning
  → tool selection
  → tool execution
  → result evaluation
  → additional retrieval if necessary
  → filtering/reranking
  → response generation
  → END
```

Private chain-of-thought is never exposed. Only high-level execution events may be surfaced for observability.

### 4.4 Music Provider Abstraction

The application communicates with music services through an internal provider interface.

Example interface:

```python
class MusicProvider:
    def search_tracks(...): ...
    def get_track(...): ...
    def get_artist(...): ...
    def get_album(...): ...
    def get_related_artists(...): ...
```

Potential implementations include:

- Spotify provider
- Local/database provider
- Future additional provider(s)

The exact provider implementation can change without changing the agent or retrieval layers.

#### Stage 2 implementation

The active music-data path is:

```text
POST /search
  -> MusicService
  -> TrackRepository (database-first lexical lookup)
  -> MusicProvider fallback when no stored result exists
  -> normalized Pydantic records
  -> MusicIngestionService
  -> Artist / Album / Track / Genre repositories
  -> PostgreSQL
```

Route handlers depend on the service rather than containing provider or persistence
logic. The provider can be replaced in tests or by a future implementation without
changing the repositories or API response shape.

MusicBrainz recordings can reference many releases. The adapter selects a release
deterministically: it requires an official, artist-consistent release; excludes live,
remix, compilation, DJ-mix, and mixtape secondary types; prefers album, then single,
then EP; and uses the earliest dated candidate as a stable tie-breaker. If no suitable
release exists, album metadata remains absent.

Deduplication is deliberately conservative. Repositories check provider plus provider
ID first, then ISRC, then normalized title and artist. When both album identities are
known, different album titles are not merged. Version labels such as live, remix,
acoustic, and remaster remain part of normalized titles, preventing those variants
from being collapsed into the base recording.

Ingestion is transactional and idempotent. It upserts the entity graph, persists genre
relationships, supports bounded pagination and retries, and rolls back failed batches.
Missing optional metadata does not erase richer data already stored.

### 4.5 Spotify Integration

Spotify is an optional integration. Stage 16 implements it as an isolated boundary:

```text
caller-provided Muse user UUID
  -> one-use, expiring OAuth state + PKCE verifier
  -> SpotifyIntegration (accounts + Web API wire adapter)
  -> SpotifyService (linkage, scope checks, token refresh)
  -> encrypted spotify_connections record
```

The core application starts and all non-Spotify features remain available when Spotify
configuration is absent. Spotify routes return a scoped configuration error in that
case rather than making application startup depend on Spotify.

It has two conceptual responsibilities:

#### Public/provider data

Where permitted, Spotify can provide external music metadata and identifiers. Stage 16
reads a connected user's top tracks and recent playback on demand. These provider
responses are typed, attributed to Spotify, and are not automatically persisted in the
catalog, memory, agent context, or recommendation pipeline.

#### User personalization

OAuth uses Authorization Code with PKCE, exact redirect-URI matching, one-use state, and
the minimum implemented scopes (`user-top-read` and `user-read-recently-played`). Muse
links the connection with Spotify's immutable pseudonymous `account_id`; the display ID
is informational only. Access and refresh tokens are encrypted with an operator-supplied
Fernet key before PostgreSQL persistence. Expiring access tokens are refreshed, rotated
refresh tokens are retained, and an invalid grant discards stored credentials and marks
the connection as requiring authorization again.

Spotify item responses expose `spotify_id`, not a generic canonical ID. The
`music_provider_mappings` table links an external `(entity_type, provider, provider_id)`
to an already-existing internal Muse entity ID. Mappings validate the canonical entity
before insertion and are not created implicitly from listening history.

Spotify data must not become the only source of truth for Muse's recommendation intelligence.

Muse's own candidate generation, filtering, semantic retrieval, reranking, and personalization pipeline remains responsible for recommendations.

OAuth state remains process-local and should move to a shared expiring store before
multi-worker deployment. User UUIDs are still caller-provided because authentication is
a later hardening stage. Spotify Development Mode quotas and authorized-user rules make
this suitable for a controlled demo rather than unrestricted public onboarding.

### 4.6 PostgreSQL

PostgreSQL stores structured application data, including:

- artists
- albums
- tracks
- genres
- users
- preferences
- conversations
- recommendation history
- external identifiers
- metadata
- integration/account state where appropriate

### 4.7 Qdrant

Qdrant stores embeddings and supports semantic retrieval for:

- music knowledge documents
- semantic music representations
- later multimodal representations

Qdrant access is hidden behind a vector-store abstraction.

#### Stage 4 embedding and indexing implementation

The indexing path is explicit and independently testable:

```text
PostgreSQL
  -> KnowledgeCorpusService
  -> deterministic KnowledgeDocument records
  -> EmbeddingProvider
  -> versioned VectorRecord batches
  -> VectorStore
  -> Qdrant
```

`EmbeddingProvider` exposes document, query, and single-text embedding operations plus
an explicit model name, version, and vector dimension. The initial hashing provider is
a deterministic development baseline that requires no external model runtime. It is
replaceable without changing corpus generation or vector storage.

`VectorStore` defines collection initialization, upsert, raw-vector search, metadata
filtering, document-ID listing, deletion, and counting. The Qdrant implementation uses
cosine distance and deterministic UUID point IDs derived from knowledge-document IDs.
Payloads retain document content, canonical entity metadata, and embedding provenance.

Every Qdrant collection stores its embedding model, version, and dimension as collection
metadata. Writes and searches validate all three values. Incompatible changes fail
instead of silently mixing vectors; the indexing command requires `--recreate` to replace
the collection deliberately.

Indexing is repeatable and idempotent. Existing document IDs are compared with the new
corpus, current documents are upserted, and stale documents are deleted. An interrupted
multi-batch write can be safely repaired by rerunning the command.

Stage 4 exposes raw vector search only as a storage capability. Query interpretation,
semantic retrieval behavior, result quality evaluation, and application-facing search
remain Stage 5 responsibilities.

#### Stage 5 semantic retrieval implementation

Stage 5 adds a learned local embedding provider and a retrieval service without adding
answer generation or agent behavior:

```text
Natural-language query
  -> FastEmbed query embedding
  -> Qdrant cosine similarity search
  -> optional structured metadata filters
  -> ranked canonical entities + scores + metadata
```

FastEmbed runs `BAAI/bge-small-en-v1.5` locally through ONNX Runtime. Documents use the
provider's passage-embedding path and searches use its query-embedding path. The model
produces 384-dimensional vectors and is tracked as
`BAAI/bge-small-en-v1.5:fastembed-0.8.1-query-passage-v1` so incompatible indexes cannot
be queried accidentally.

`SemanticRetriever` accepts a validated non-empty query, embeds it, delegates similarity
search and filters to `VectorStore`, and returns ranked results containing the canonical
entity type and ID, similarity score, document content, and structured metadata. It is
callable directly from Python and currently has a CLI rather than a public HTTP route.

The Stage 5 evaluation harness indexes a small curated corpus in Qdrant local mode and
calculates Recall@K and mean reciprocal rank deterministically. The initial five-query
baseline achieved Recall@3 of 1.00 and MRR of 1.00. This is a regression smoke test;
larger and more difficult evaluation sets are still required before production claims.

No lexical retrieval, score fusion, reranking, prompt construction, generation, RAG, or
agent orchestration is part of Stage 5.

#### Stage 6 hybrid retrieval implementation

Stage 6 adds lexical evidence and deterministic rank fusion while preserving the Stage 5
semantic boundary:

```text
Query
  +-> FastEmbed + Qdrant semantic candidates --+
  +-> in-process BM25 lexical candidates -------+-> Reciprocal Rank Fusion -> candidate set
       (current structured filters constrain both branches)
```

`BM25LexicalRetriever` tokenizes normalized Unicode text and ranks the deterministic
Stage 3 knowledge documents with standard BM25 term-frequency, inverse-document-frequency,
and length-normalization components. It runs in process because the development corpus
currently contains 39 documents. This avoids adding PostgreSQL full-text schema or an
OpenSearch service before corpus size and measured latency justify either choice.

`HybridRetriever` requests up to 50 candidates from each source, deduplicates by stable
document ID, and applies unweighted Reciprocal Rank Fusion. For each source rank `r`, the
contribution is `1 / (60 + r)`. Raw cosine and BM25 values are retained for diagnostics
but never added or normalized together. Final ties use the best source rank and document
ID, making output repeatable.

The existing document type, entity ID, genre, release-year, and provider filters are
applied inside both retrieval branches. They therefore constrain candidate generation
instead of becoming a separate unranked list. Stage 7 owns the broader filter vocabulary
such as artist exclusion, duration, popularity, country, and album type.

The hybrid evaluator reuses the curated Stage 5 corpus and reports semantic-only and
hybrid Recall@K/MRR side by side. Both currently score Recall@3 = 1.00 and MRR = 1.00;
because the five-query dataset is deliberately small, this establishes regression safety
but does not demonstrate production relevance or a measurable hybrid improvement.

No reranking, context construction, generation, RAG, query parsing, or agent orchestration
is part of Stage 6.

#### Stage 7 metadata filtering implementation

Stage 7 expands the structured filter object without introducing an LLM query parser:

```text
Validated VectorSearchFilter
  +-> Qdrant must/must_not payload conditions -> semantic candidates
  +-> deterministic document predicates ------> BM25 candidates
                                                   |
                                                   +-> Stage 6 RRF
```

`VectorSearchFilter` is a frozen Pydantic value object. It supports document and entity
selection, genres, release-year ranges, artist exclusions by ID or exact normalized name,
duration and popularity ranges, instrumental status, country, album type, and provider.
It validates numeric bounds and ordered ranges, removes duplicate values, and normalizes
whitespace and case for textual constraints.

Knowledge documents now carry the minimum canonical metadata required by these filters:
artist name, country, album type, popularity, and a nullable instrumental flag in addition
to the existing IDs, genres, year, duration, and provider. Qdrant payloads retain display
metadata and separately store normalized string facets. This makes vector filtering
case-insensitive without changing the content returned to callers.

Positive constraints use strict unknown-value semantics: a document with no popularity,
country, or other requested field does not match. Artist exclusions use Qdrant `must_not`
conditions for semantic retrieval and identical predicates before BM25 scoring. Existing
provider instrumentalness values at or above 0.5 map to `instrumental=true`; missing
values remain unknown. This is metadata classification only, not audio analysis.

The semantic branch pushes all constraints into Qdrant. The BM25 branch currently applies
them to the deterministic in-memory corpus before scoring; SQL pushdown is deferred until
the corpus size warrants replacing or restructuring that lexical implementation.

Stage 7 does not parse natural language into filters and does not add reranking, RAG,
generation, agents, Spotify integration, or audio processing.

#### Stage 8 reranking implementation

Stage 8 introduces a learned second-stage scorer without changing the default hybrid path:

```text
up to 100 semantic candidates + up to 100 lexical candidates
  -> Reciprocal Rank Fusion
  -> top 20 hybrid candidates
  -> configurable cross-encoder
  -> top 5-10 reranked results
```

`Reranker` is a provider-independent interface that scores a query against an ordered
sequence of document texts. `FastEmbedCrossEncoderReranker` implements it with local ONNX
inference and lazy model loading. The configured default is
`Xenova/ms-marco-MiniLM-L-6-v2:fastembed-0.8.1`; its files share the ignored
`data/models` cache with the embedding model.

`RerankedRetriever` requests 20 candidates from `HybridRetriever`, validates score count
and finiteness, sorts by descending cross-encoder score, and uses hybrid rank plus stable
document ID as deterministic tie-breakers. Results retain reranker score, prior hybrid
rank and fusion score, semantic and lexical provenance, content, and canonical metadata.
Stage 7 filters are applied before reranking, so the cross-encoder cannot reintroduce a
candidate that violates a deterministic constraint.

The Stage 8 evaluation compares hybrid and reranked Recall@K/MRR on identical queries.
Both scored Recall@3 = 1.00 and MRR = 1.00 on the existing five-query dataset, producing
zero measured improvement. Because that dataset is ceiling-limited, reranking remains an
explicit experimental path and has not replaced hybrid retrieval as the application
default. A larger, harder judged dataset is required before that decision changes.

Stage 8 does not add context assembly, generation, RAG, LLMs, agent orchestration,
Spotify integration, or audio processing.

#### Stage 9 basic RAG implementation

Stage 9 adds answer generation without introducing agent behavior:

```text
user query + explicit filters
  -> hybrid retrieval
  -> cross-encoder reranking
  -> bounded structured evidence (E1...En)
  -> Groq strict JSON-schema generation
  -> Pydantic validation and citation validation
  -> grounded response with sources
```

`ContextBuilder` caps both evidence count and total document characters. Each evidence
item contains the canonical document and entity identity, content, metadata, retrieval
rank, and reranker score. The generator therefore receives data rather than a single
unstructured prompt blob.

`GeneratedGroundedAnswer` requires distinct retrieved-fact, recommendation-reasoning,
and uncertainty arrays. Factual and recommendation statements require evidence IDs,
and `RAGService` rejects any ID that was not supplied to the model. When retrieval is
empty, the service returns an explicit insufficient-evidence response without making
an LLM request.

`LLMRAGGenerator` translates RAG evidence into the provider-independent LLM request
described below. It requests a closed, strict JSON schema and validates the returned
JSON with Pydantic. Provider credentials remain server-side.

The API exposes `POST /answer`. Its RAG runtime is initialized lazily so application
startup and non-RAG endpoints do not require Qdrant, model loading, or a Groq key.
Embedding and reranking models plus external clients are reused across answer requests
and closed during application shutdown. The in-process BM25 corpus is a first-request
snapshot, so the API must currently restart after database ingestion.

#### Stage 10 LLM provider abstraction

All external model access now crosses `LLMProvider`:

```text
RAG / future query understanding / future agent
  -> ChatCompletionRequest
  -> LLMProvider
  -> configured provider adapter (currently Groq)
  -> ChatCompletionResponse or StructuredCompletion[T]
```

The contract supports ordered system, user, assistant, and tool messages; per-request
model and generation overrides; typed function definitions and tool calls; normal chat
completion; and Pydantic-validated structured completion. Tool calls are represented
but not executed—tool registration and execution belong to Stage 12.

`GroqLLMProvider` is the only module that knows the external endpoint or wire format.
It translates neutral messages and tools to the provider payload, converts strict
Pydantic schemas to the provider's closed-object requirements, parses tool arguments,
and normalizes configuration, provider, rate-limit, and response failures.

Transport failures and HTTP 408, 409, 429, and 5xx responses use bounded exponential
backoff and honor numeric `Retry-After` values. Non-transient client errors and invalid
model output fail immediately. Provider-reported prompt, completion, and total token
counts are retained on completion responses and surfaced in RAG generation metadata.

Generic `LLM_*` settings choose the provider, model, timeouts, output cap, and retry
policy. Provider-specific settings contain only credentials and endpoint details. The
factory is the single composition point for selecting an implementation.

#### Stage 11 query understanding

Natural-language interpretation is isolated from retrieval and agent behavior:

```text
raw user query
  -> QueryUnderstandingService
  -> LLMProvider.complete_structured(UnderstoodMusicQuery)
  -> Pydantic intent/filter/preference validation
  -> optional conversion to VectorSearchFilter
```

`UnderstoodMusicQuery` distinguishes four bounded intents: music discovery, music
research, similarity search, and unknown. It also separates the semantic search text,
hard retrieval constraints, and qualitative popularity preference. The extraction
schema deliberately omits database entity IDs and provider internals because those
cannot be safely inferred from natural language.

The query-understanding filter model uses user-facing `year_min` and `year_max` fields
and converts explicitly to the storage-facing `VectorSearchFilter`. Construction of
that conversion validates ranges, identifiers, normalized strings, and other existing
retrieval invariants before any future caller can send the result to retrieval.

Descriptive phrases remain in the semantic query unless the user explicitly asks for
a hard genre, entity-type, album-type, or numeric constraint. Qualitative popularity
is retained as a preference rather than converted into an invented numeric threshold.
Strict structured decoding provides the expected shape, and Pydantic remains the final
authority for business validation. Invalid output raises an LLM response error and no
partially parsed filter is returned.

Stage 11 exposes a Python service and development CLI. It does not execute retrieval,
register tools, or choose an agent workflow; those concerns begin in later stages.

#### Stage 12 tool layer

Stage 12 creates the safe application boundary that future orchestration can call:

```text
future bounded orchestrator
  -> ToolRegistry explicit allow-list
  -> BaseTool input validation
  -> one focused operation
  -> typed Pydantic output or stable ToolError
```

Every tool declares a unique name, purpose, Pydantic input type, Pydantic output type,
and an LLM-compatible JSON Schema definition. `ToolRegistry` rejects duplicate names,
does not dynamically import tools, and refuses calls to anything not registered.
Unexpected implementation failures are translated to a generic execution error so
database or provider details are not exposed across the tool boundary.

The initial registry includes local catalog search, canonical track/artist/album lookup,
and semantic knowledge search. External-provider search is registered only when a
provider is explicitly configured, and it returns normalized results without implicitly
writing them to the database. Local catalog results include canonical database IDs so a
later workflow can perform an unambiguous follow-up lookup.

Preference tools are deferred until Stage 15 establishes an intentional user-memory
model. True similar-track tooling is deferred until the evaluated similarity engine in
Stage 19. The Stage 12 layer does not let the LLM select, chain, retry, or evaluate tools;
those workflow responsibilities belong to Stage 13.

#### Stage 13 agentic orchestration

Stage 13 introduces LangGraph as a bounded workflow coordinator:

```text
START
  -> understand_query
  -> plan_retrieval
  -> retrieve_music_evidence
  -> evaluate_evidence
       | enough ----------------------> generate_response
       | insufficient discovery query
       v
     search_external_music_provider
       -> evaluate_evidence
       -> generate_response
  -> END
```

The graph carries explicit typed state for the original query, understood intent,
retrieval plan, tool counts, iteration counts, safe execution traces, evidence, and final
response. Query understanding remains LLM-backed, while retrieval planning is
deterministic: all answerable intents first use filtered hybrid retrieval plus
cross-encoder reranking. Only discovery and similarity intents may use the track provider
as a fallback; research requests never treat sparse provider metadata as factual research
evidence.

Execution is bounded independently by maximum graph retrieval iterations, maximum tool
calls, maximum evidence count, and an elapsed-time budget. The current synchronous
clients retain their own request timeouts; the graph checks its elapsed budget at node
boundaries. It performs at most one primary retrieval and one external fallback, so it
cannot enter an open-ended loop. Tool failures become safe trace records instead of
uncaught provider details.

Knowledge and external-provider results are converted to a shared evidence envelope with
distinct provenance. Generated facts and recommendation reasoning must cite supplied
evidence IDs. Unknown citations or generation failures return an evidence-preserving,
deterministic fallback; empty retrieval returns an insufficient-evidence response without
calling the answer generator.

The agent runtime lazily reuses embedding, Qdrant, reranker, LLM, and music-provider
resources. It is callable directly through Python or `scripts/run_agent.py` and is also
exposed as `POST /agent`. No conversation messages or persistent memory are placed in
graph state because those belong to Stages 14 and 15.

#### Stage 14 conversational state

Stage 14 adds a short-term conversation boundary around the existing request-local graph:

```text
POST /agent (query, optional conversation_id)
  -> bounded ConversationStore lookup/create
  -> compact prior-turn QueryConversationContext
  -> context-aware structured query understanding
  -> unchanged Stage 13 retrieval/generation graph
  -> advance compact conversation snapshot
  -> optimistic save and return conversation_id
```

`ConversationState` stores only the current intent and previous query, resolved semantic
query, active user-facing filters, ordered entity references, and ordered recommendation
references. Entity references retain explicit result positions so follow-ups such as
"the third result" do not depend on implicit list interpretation. Full prompts, answers,
tool payloads, arbitrary chat history, and inferred user preferences are not stored.

The query-understanding call receives the compact state as structured context. Its prompt
must resolve only genuinely referential or elliptical turns, produce a standalone current
semantic query and complete active filters, let explicit new constraints override old
ones, and ignore context for clearly independent requests. The resulting plan and tool
path remain bounded exactly as in Stage 13.

The initial `InMemoryConversationStore` is thread-safe, capped by active-conversation
count, and expires inactive state by TTL. Immutable snapshots prevent callers from
mutating stored state. Saves use the previous turn count as an optimistic version so two
requests cannot silently overwrite the same conversation turn. Unknown/expired IDs return
404 and conflicting updates return 409. State can be inspected and explicitly deleted via
the agent API.

This storage is intentionally separate from Stage 15 user memory: it is anonymous,
ephemeral, process-local, and contains no durable preferences. A process restart or a
request routed to another worker loses access to the conversation, which is acceptable
for the current development runtime but requires a shared store before multi-instance
deployment.

#### Stage 15 user memory

Stage 15 introduces durable preference memory as a separate PostgreSQL-backed domain:

```text
explicit preference API / future trusted importer
  -> UserMemoryService validation
  -> normalized identity + optimistic version check
  -> UserPreferenceRepository
  -> user_preferences table
  -> inspectable UserPreferenceRecord
```

Each record is scoped by an external user UUID and stores a bounded category, display
value, normalized identity, stance, source, optional source detail, confidence, creation
and update timestamps, and an optimistic version. The initial categories cover artists,
genres, popularity, listening contexts, and moods. Stances distinguish likes, dislikes,
preferences, avoidance, and frequently requested contexts.

Public writes are intentionally explicit. They always receive `explicit_user` provenance
and confidence `1.0`; callers cannot spoof provider or inferred sources. The underlying
service supports future trusted importers with explicit source and confidence arguments,
but no chat message, agent turn, or conversation state automatically creates durable
memory. Case-insensitive normalized identities prevent duplicate category/value records
while preserving display text.

Updates and deletes require the last observed version. Conditional SQL mutations reject
stale writers, and uniqueness conflicts return a stable conflict error. Memory is
inspectable and deletable through the user-preference API. The tool layer exposes only
`get_user_preferences`; there is no agent-writable memory tool.

Stage 15 stores preferences but deliberately does not feed them into retrieval or
reranking by itself. Stage 17 reads this same inspectable profile without adding an
implicit memory-write path. User UUIDs are not yet backed by authentication or a
user-account table, which remains a deployment and security limitation rather than
implicit authorization.

#### Stage 17 personalized retrieval

Stage 17 wraps the established cross-encoder output instead of changing semantic, BM25,
filtering, fusion, or relevance-model behavior:

```text
hybrid retrieval -> cross-encoder candidates
  + explicit UserPreference profile
  + optional on-demand Spotify listening signals
  + current-query popularity preference
  -> deterministic component scoring
  -> greedy diversity-aware selection
  -> evidence with inspectable personalization provenance
```

Requests without a user UUID preserve the existing retriever path. A current query's
explicit qualitative popularity request may activate the same post-ranker without
durable user memory. The agent accepts an optional `user_id`, passes only typed signals
through the retrieval tool, and includes the selected candidate's score breakdown in
knowledge evidence supplied for grounded explanation.

Base relevance is rank-normalized because cross-encoder logits are not calibrated
probabilities. Explicit artist, genre, popularity, mood, and context records contribute
bounded positive or negative preference values using their stored confidence. Provider
signals identify known tracks through explicit mappings or ISRC and familiar artists by
normalized name. Familiarity contributes taste affinity while a separate novelty score
prevents the architecture from equating "already heard" with "always recommend."

Selection is greedy and deterministic. After relevance, preference, and novelty are
combined, repeated artists and overlapping genre sets receive a bounded diversity
penalty relative to already selected results. Every component and concise matching
reason remains in `PersonalizationBreakdown`; no hidden model reasoning is exposed.

Default weights are configuration, not permanent product truth. The Stage 17 synthetic
benchmark compares relevance-only ranking with the balanced preset and records
preference hit rate, avoidance success, novelty, artist diversity, and relevance
retention. The adopted development preset is `0.50 relevance + 0.20 preference + 0.15
novelty + 0.15 diversity`; the three-case baseline retained 95.06% of top-k relevance
while improving its preference, avoidance, and diversity checks. Larger offline and
real-user evaluation remains Stage 20 work.

### 4.8 RAG Knowledge Base

The knowledge base can contain:

- artist biographies
- album information
- genre information
- music history
- artist influences
- album reviews
- public music research
- curated music information

Copyrighted lyrics are not used as a core dataset.

#### Stage 3 document corpus

Stage 3 creates a storage-neutral document boundary before any embedding or vector
database is introduced:

```text
PostgreSQL music entities
  -> eager-loaded canonical entity graph
  -> deterministic text cleaning
  -> entity-specific document builder
  -> content + typed metadata
  -> deterministic JSONL corpus
```

Artist, album, track, and genre builders create concise natural-language content rather
than embedding serialized database rows. Metadata remains separate and includes the
document type, canonical entity IDs, provider provenance, genres, release year, and
other entity-specific filtering fields where available.

Document IDs use `<entity-type>:<database-id>`, making regeneration idempotent within a
database. Documents are ordered by entity type and numeric ID, metadata values are
sorted deterministically, and JSON keys are sorted during export. Rebuilding an
unchanged database therefore produces byte-for-byte identical corpus output.

Each music entity is an atomic document at this stage. Embeddings, vector indexing,
semantic retrieval, and chunking of future long-form external sources are deliberately
deferred beyond Stage 3.

### 4.9 Retrieval and Recommendation Pipeline

```text
User Query
   ↓
Query Understanding
   ↓
Candidate Generation
   ├── semantic retrieval
   └── structured music database search
   ↓
Metadata Filtering
   ↓
Reranking
   ↓
Personalization
   ↓
Top Recommendations
   ↓
LLM Explanation
```

The LLM does not arbitrarily choose final recommendations. Deterministic constraints and ranking logic remain outside the LLM wherever practical.

### 4.10 Memory

Muse has two memory layers.

Short-term memory:

- current conversation
- referenced recommendations
- current task state

Long-term memory:

- stable favorite artists
- favorite/disliked genres
- popularity preference
- energy preference
- discovery/novelty preference
- liked/rejected recommendations

Long-term memory is intentionally updated rather than saving every user statement.

## 5. Agent Tools

The initial tool contract includes:

- `search_music(query)`
- `get_song_details(song_id)`
- `get_artist_details(artist_id)`
- `find_similar_songs(song_id, filters)`
- `search_music_knowledge(query)`
- `search_web(query)`
- `get_user_preferences(user_id)`
- `save_user_preference(user_id, preference)`

Spotify-specific operations should be added behind a Spotify integration/service layer rather than leaking Spotify-specific behavior throughout the agent.

Every tool requires:

- input schema
- output schema
- validation
- error handling
- unit tests
- clear agent-facing description

## 6. Data Ownership and Source-of-Truth Principles

Muse distinguishes between:

1. Internal normalized music data
2. External provider identifiers/data
3. Knowledge documents
4. User preferences
5. Derived recommendation signals

External provider data should be normalized into Muse's internal schemas. Provider-specific fields should not dictate the entire internal architecture.

## 7. Configuration and Provider Independence

Provider/model choices are configuration-driven.

Examples:

- LLM provider/model
- embedding model
- reranker model
- music provider
- database URLs
- Qdrant configuration
- Spotify credentials

Secrets are stored in environment variables and never committed to source control.

## 8. Evaluation Architecture

Evaluation is a first-class subsystem rather than an afterthought.

Metrics include:

- Recall@K
- Precision@K
- MRR
- NDCG@K
- context relevance
- context recall
- answer relevance
- faithfulness
- citation correctness
- tool selection accuracy
- task completion rate
- unnecessary tool calls
- recommendation relevance
- diversity
- novelty
- personalization
- constraint satisfaction
- latency
- token usage
- estimated cost
- API calls
- error rate

## 9. Observability

Each request should produce a structured high-level execution trace containing, where appropriate:

- request ID
- timestamp
- detected intent
- agent steps
- tool calls
- retrieval results
- reranking results
- model calls
- token usage
- latency
- errors
- final response

Sensitive information should not be logged unnecessarily.

## 10. Deployment Architecture

Local development uses Docker Compose with:

- backend
- PostgreSQL
- Qdrant
- frontend

Production deployment can be decided later. Infrastructure complexity should be introduced only when it serves a concrete requirement.

## 11. Version Strategy

### v0.1
Foundation and music data

### v0.2
RAG

### v0.3
Hybrid retrieval and reranking

### v0.4
LangGraph agent and tools

### v0.5
Memory and personalization

### v0.6
Evaluation and failure analysis

### v0.7
Production web application and observability

### v1.0
Stable agentic music research/discovery platform with optional Spotify integration

### v2.0
Multimodal audio intelligence

## 12. Key Architectural Decision

Muse is a music intelligence application with an optional Spotify integration, not a Spotify recommendation wrapper.

This preserves the project's core objective: demonstrating a modular, evaluated, observable agentic RAG and recommendation system.
