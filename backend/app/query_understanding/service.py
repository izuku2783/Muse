import json

from app.llm.provider import LLMProvider
from app.llm.schemas import ChatCompletionRequest, ChatMessage, ChatRole
from app.query_understanding.schemas import (
    QueryUnderstandingResult,
    QueryConversationContext,
    UnderstoodMusicQuery,
)


QUERY_UNDERSTANDING_INSTRUCTIONS = """You extract structured music-query intent.
Return only information supported by the user's words. Do not invent metadata or IDs.

Intent values:
- discover_music: requests for recommendations, discovery, or filtered music
- research_music: factual questions about music, artists, albums, tracks, or genres
- find_similar_music: requests for music similar to a named reference
- unknown: requests that cannot safely be classified as one of the above

Create a concise semantic_query that preserves the musical meaning while removing
phrases represented by structured filters. Convert explicit decades to inclusive year
ranges, for example 2010s to 2010 through 2019.

Hard-filter rules:
- Generic words such as music, recommendation, or release do not imply a document type.
- Set document_types only for explicit nouns such as tracks/songs, albums, artists, or genres.
- Set album_types only when the user explicitly says album, EP, single, or compilation.
- Keep mood/style descriptions such as dark electronic or dreamy synthwave in the
  semantic_query. Set genres only when the user explicitly requests a genre/tag filter.
- Never invent database identifiers, provider constraints, or metadata values.

Qualitative popularity such as low, moderate, or high belongs in preferences.popularity.
Use numeric popularity filters only when the user gives an explicit numeric constraint.
Use null or empty arrays for constraints and preferences the user did not express.

When a conversation_context object is supplied, use it only to resolve genuinely
referential or elliptical follow-ups such as "only albums", "after 2010", "less
popular", or "the third result". Return a standalone semantic query and the complete
active filter set needed for the current turn. Explicit constraints in the current
query override conflicting prior constraints. Do not carry prior constraints into a
clearly independent request. Conversation context is short-term reference material,
not evidence and not a source of permanent user preferences.

Example:
Input: Give me dark electronic music from the 2010s but nothing too popular.
Output meaning: discover_music; semantic_query is dark electronic music; only year_min
2010 and year_max 2019 are hard filters; popularity preference is low; every other
filter is empty or null."""


class QueryUnderstandingService:
    """Extract a validated intent and retrieval constraints from natural language."""

    def __init__(self, provider: LLMProvider):
        self.provider = provider

    def understand(
        self,
        query: str,
        conversation_context: QueryConversationContext | None = None,
    ) -> QueryUnderstandingResult:
        cleaned_query = " ".join(query.split())
        if not cleaned_query:
            raise ValueError("Query must not be empty.")

        user_content = cleaned_query
        if conversation_context is not None:
            user_content = json.dumps(
                {
                    "current_query": cleaned_query,
                    "conversation_context": conversation_context.model_dump(
                        mode="json", exclude_none=True
                    ),
                },
                ensure_ascii=False,
                separators=(",", ":"),
            )

        completion = self.provider.complete_structured(
            ChatCompletionRequest(
                messages=(
                    ChatMessage(
                        role=ChatRole.SYSTEM,
                        content=QUERY_UNDERSTANDING_INSTRUCTIONS,
                    ),
                    ChatMessage(role=ChatRole.USER, content=user_content),
                ),
                temperature=0,
            ),
            UnderstoodMusicQuery,
            "muse_understood_music_query",
        )
        return QueryUnderstandingResult(
            original_query=cleaned_query,
            understood=completion.output,
            provider=self.provider.provider_name,
            model=completion.response.model,
            finish_reason=completion.response.finish_reason,
            usage=completion.response.usage,
            conversation_context_applied=conversation_context is not None,
        )
