from time import sleep
from uuid import UUID

import pytest

from app.agent.schemas import AgentLimits, AgentStatus, EvidenceSource
from app.agent.service import AgentService
from app.knowledge.schemas import DocumentMetadata, DocumentType
from app.query_understanding.schemas import (
    MusicQueryIntent,
    QueryPreferences,
    QueryUnderstandingFilters,
    QueryUnderstandingResult,
    UnderstoodMusicQuery,
)
from app.rag.schemas import (
    GeneratedGroundedAnswer,
    GroundedStatement,
    RAGGenerationMetadata,
    RAGGenerationResult,
)
from app.retrieval.schemas import RerankedSearchResponse, RerankedSearchResult
from app.retrieval.schemas import PersonalizationBreakdown
from app.services.music.schemas import ArtistData, TrackData
from app.tools.base import BaseTool
from app.tools.errors import ToolExecutionError
from app.tools.registry import ToolRegistry
from app.tools.schemas import (
    RerankedSearchInput,
    SearchExternalMusicProviderInput,
    SearchExternalMusicProviderOutput,
)


def understanding(intent=MusicQueryIntent.DISCOVER_MUSIC):
    return QueryUnderstandingResult(
        original_query="dark synthwave",
        understood=UnderstoodMusicQuery(
            intent=intent,
            semantic_query="dark synthwave",
            filters=QueryUnderstandingFilters(
                document_types=(DocumentType.TRACK,),
                genres=("synthwave",),
                exclude_artists=(),
                year_min=None,
                year_max=None,
                duration_ms_min=None,
                duration_ms_max=None,
                popularity_min=None,
                popularity_max=None,
                instrumental=None,
                countries=(),
                album_types=(),
            ),
            preferences=QueryPreferences(popularity=None),
        ),
        provider="test",
        model="test-model",
    )


class StubUnderstanding:
    def __init__(self, result, delay=0):
        self.result = result
        self.delay = delay
        self.calls = []

    def understand(self, query, conversation_context=None):
        self.calls.append((query, conversation_context))
        if self.delay:
            sleep(self.delay)
        return self.result


def reranked_response(results=True):
    metadata = DocumentMetadata(
        document_type=DocumentType.TRACK,
        entity_id=1,
        track_id=1,
        artist_name="Neon Waves",
        genres=("synthwave",),
    )
    items = (
        RerankedSearchResult(
            rank=1,
            hybrid_rank=1,
            document_id="track:1",
            entity_type=DocumentType.TRACK,
            entity_id=1,
            reranker_score=0.9,
            fusion_score=0.03,
            semantic_rank=1,
            semantic_score=0.8,
            content="Night Drive is a dark synthwave track by Neon Waves.",
            metadata=metadata,
        ),
    ) if results else ()
    return RerankedSearchResponse(
        query="dark synthwave",
        candidate_count=len(items),
        reranker_model="test-reranker",
        reranker_version="1",
        results=items,
    )


class EvidenceTool(BaseTool[RerankedSearchInput, RerankedSearchResponse]):
    name = "retrieve_music_evidence"
    description = "Retrieve test evidence."
    input_type = RerankedSearchInput
    output_type = RerankedSearchResponse

    def __init__(self, response=None, error=None):
        self.response = response or reranked_response()
        self.error = error
        self.calls = []

    def run(self, tool_input):
        self.calls.append(tool_input)
        if self.error:
            raise self.error
        return self.response


class ExternalTool(
    BaseTool[SearchExternalMusicProviderInput, SearchExternalMusicProviderOutput]
):
    name = "search_external_music_provider"
    description = "Search a test provider."
    input_type = SearchExternalMusicProviderInput
    output_type = SearchExternalMusicProviderOutput

    def __init__(self):
        self.calls = []

    def run(self, tool_input):
        self.calls.append(tool_input)
        return SearchExternalMusicProviderOutput(
            query=tool_input.query,
            results=(
                TrackData(
                    provider_id="external-1",
                    provider="musicbrainz",
                    title="Remote Signal",
                    artist=ArtistData(provider_id="artist-1", name="Remote Artist"),
                    genres=["synthwave"],
                ),
            ),
        )


class StubGenerator:
    def __init__(self, error=None, citation="E1"):
        self.error = error
        self.citation = citation
        self.calls = []

    def __call__(self, query, evidence):
        self.calls.append((query, evidence))
        if self.error:
            raise self.error
        statement = GroundedStatement(
            text="Night Drive matches the request.",
            evidence_ids=(self.citation,),
        )
        return RAGGenerationResult(
            output=GeneratedGroundedAnswer(
                answer="Try Night Drive.",
                retrieved_facts=(statement,),
                recommendation_reasoning=(statement,),
                uncertainty=(),
            ),
            metadata=RAGGenerationMetadata(provider="test", model="test-model"),
        )


def service(intent=MusicQueryIntent.DISCOVER_MUSIC, evidence_tool=None, external=None, generator=None, limits=None, delay=0, clock=None):
    evidence_tool = evidence_tool or EvidenceTool()
    tools = [evidence_tool]
    if external:
        tools.append(external)
    return AgentService(
        StubUnderstanding(understanding(intent), delay=delay),
        ToolRegistry(tools),
        generator or StubGenerator(),
        limits=limits,
        **({"clock": clock} if clock else {}),
    )


def test_graph_understands_plans_retrieves_evaluates_and_generates():
    evidence_tool = EvidenceTool()
    external = ExternalTool()
    agent = service(evidence_tool=evidence_tool, external=external)

    response = agent.run("  dark   synthwave ")

    assert response.status == AgentStatus.COMPLETED
    assert response.plan.filters.genres == ("synthwave",)
    assert response.sources[0].source == EvidenceSource.KNOWLEDGE
    assert [trace.tool_name for trace in response.tool_executions] == [
        "retrieve_music_evidence"
    ]
    assert evidence_tool.calls[0].query == "dark synthwave"
    assert external.calls == []


def test_graph_forwards_user_id_and_exposes_personalization_provenance():
    evidence_tool = EvidenceTool()
    personalized = evidence_tool.response.results[0].model_copy(
        update={
            "personalization": PersonalizationBreakdown(
                relevance_score=1,
                explicit_preference_score=1,
                provider_affinity_score=0,
                preference_score=1,
                novelty_score=1,
                diversity_penalty=0,
                final_score=0.9,
                reasons=("explicit genre like: synthwave",),
            )
        }
    )
    evidence_tool.response = evidence_tool.response.model_copy(
        update={
            "results": (personalized,),
            "personalization_applied": True,
            "personalization_sources": ("explicit_preferences",),
        }
    )
    agent = service(evidence_tool=evidence_tool)

    response = agent.run("dark synthwave", user_id=UUID(int=1))

    assert evidence_tool.calls[0].user_id == UUID(int=1)
    assert response.sources[0].personalization.preference_score == 1


def test_graph_uses_external_provider_once_when_discovery_evidence_is_empty():
    external = ExternalTool()
    agent = service(
        evidence_tool=EvidenceTool(response=reranked_response(results=False)),
        external=external,
    )

    response = agent.run("dark synthwave")

    assert response.status == AgentStatus.COMPLETED
    assert response.sources[0].source == EvidenceSource.EXTERNAL_PROVIDER
    assert response.sources[0].track.title == "Remote Signal"
    assert [trace.tool_name for trace in response.tool_executions] == [
        "retrieve_music_evidence",
        "search_external_music_provider",
    ]
    assert len(external.calls) == 1


def test_research_query_does_not_use_track_provider_as_factual_fallback():
    external = ExternalTool()
    agent = service(
        intent=MusicQueryIntent.RESEARCH_MUSIC,
        evidence_tool=EvidenceTool(response=reranked_response(results=False)),
        external=external,
    )

    response = agent.run("dark synthwave")

    assert response.status == AgentStatus.INSUFFICIENT_EVIDENCE
    assert external.calls == []
    assert response.sources == ()


@pytest.mark.parametrize(
    "limits",
    [
        AgentLimits(max_iterations=1),
        AgentLimits(max_tool_calls=1),
    ],
)
def test_execution_limits_prevent_additional_retrieval(limits):
    external = ExternalTool()
    response = service(
        evidence_tool=EvidenceTool(response=reranked_response(results=False)),
        external=external,
        limits=limits,
    ).run("dark synthwave")

    assert response.status == AgentStatus.INSUFFICIENT_EVIDENCE
    assert external.calls == []
    assert len(response.tool_executions) == 1


def test_tool_failure_is_recorded_without_leaking_an_exception():
    agent = service(
        intent=MusicQueryIntent.RESEARCH_MUSIC,
        evidence_tool=EvidenceTool(error=ToolExecutionError("safe failure")),
    )

    response = agent.run("dark synthwave")

    assert response.status == AgentStatus.INSUFFICIENT_EVIDENCE
    assert response.tool_executions[0].error_code == "tool_execution_error"
    assert response.uncertainty == ("safe failure",)


def test_generation_failure_returns_evidence_preserving_fallback():
    response = service(generator=StubGenerator(error=RuntimeError("secret"))).run(
        "dark synthwave"
    )

    assert response.status == AgentStatus.FALLBACK
    assert response.sources[0].document_id == "track:1"
    assert "secret" not in response.uncertainty[0]
    assert response.generation is None


def test_invalid_generated_citation_returns_safe_fallback():
    response = service(generator=StubGenerator(citation="E99")).run("dark synthwave")
    assert response.status == AgentStatus.FALLBACK
    assert response.retrieved_facts == ()


def test_timeout_prevents_tool_execution_and_returns_a_bounded_result():
    evidence_tool = EvidenceTool()
    times = iter((0.0, 1.0))
    response = service(
        intent=MusicQueryIntent.RESEARCH_MUSIC,
        evidence_tool=evidence_tool,
        limits=AgentLimits(timeout_seconds=0.5),
        clock=times.__next__,
    ).run("dark synthwave")

    assert response.status == AgentStatus.INSUFFICIENT_EVIDENCE
    assert evidence_tool.calls == []
    assert response.tool_executions[0].error_code == "agent_timeout"


def test_unknown_intent_finishes_without_calling_tools_or_generator():
    evidence_tool = EvidenceTool()
    generator = StubGenerator()
    response = service(
        intent=MusicQueryIntent.UNKNOWN,
        evidence_tool=evidence_tool,
        generator=generator,
    ).run("what is this")

    assert response.status == AgentStatus.INSUFFICIENT_EVIDENCE
    assert evidence_tool.calls == []
    assert generator.calls == []
    assert response.tool_executions == ()


def test_follow_up_uses_previous_domain_filters_entities_and_recommendations():
    initial = understanding(MusicQueryIntent.FIND_SIMILAR_MUSIC).model_copy(
        update={
            "understood": understanding().understood.model_copy(
                update={
                    "intent": MusicQueryIntent.FIND_SIMILAR_MUSIC,
                    "semantic_query": "albums similar to Radiohead",
                    "filters": understanding().understood.filters.model_copy(
                        update={
                            "document_types": (DocumentType.ALBUM,),
                            "genres": (),
                        }
                    ),
                }
            )
        }
    )
    follow_up = initial.model_copy(
        update={
            "original_query": "Only ones after 2010.",
            "understood": initial.understood.model_copy(
                update={
                    "filters": initial.understood.filters.model_copy(
                        update={"year_min": 2010}
                    )
                }
            ),
        }
    )

    class SequentialUnderstanding:
        def __init__(self):
            self.outputs = iter((initial, follow_up))
            self.contexts = []

        def understand(self, query, conversation_context=None):
            self.contexts.append(conversation_context)
            return next(self.outputs).model_copy(update={"original_query": query})

    query_service = SequentialUnderstanding()
    evidence_tool = EvidenceTool()
    agent = AgentService(
        query_service,
        ToolRegistry((evidence_tool,)),
        StubGenerator(),
    )

    first = agent.run("Give me Radiohead-like albums.")
    second = agent.run(
        "Only ones after 2010.",
        conversation_id=first.conversation.conversation_id,
    )

    assert query_service.contexts[0] is None
    context = query_service.contexts[1]
    assert context.current_intent == MusicQueryIntent.FIND_SIMILAR_MUSIC
    assert context.previous_filters.document_types == (DocumentType.ALBUM,)
    assert len(context.previous_entities) == 1
    assert len(context.previous_recommendations) == 1
    assert evidence_tool.calls[1].filters.release_year_min == 2010
    assert evidence_tool.calls[1].filters.document_types == (DocumentType.ALBUM,)
    assert second.conversation.turn_count == 2
    assert second.conversation.previous_query == "Only ones after 2010."
