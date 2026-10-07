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

### 4.5 Spotify Integration

Spotify is an optional integration.

It has two conceptual responsibilities:

#### Public/provider data

Where permitted, Spotify can provide external music metadata and identifiers.

#### User personalization

After OAuth authorization, Muse can use permitted user data to improve personalization, such as understanding a user's listening-related preferences and supporting Spotify-oriented workflows.

Spotify data must not become the only source of truth for Muse's recommendation intelligence.

Muse's own candidate generation, filtering, semantic retrieval, reranking, and personalization pipeline remains responsible for recommendations.

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
