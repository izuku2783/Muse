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