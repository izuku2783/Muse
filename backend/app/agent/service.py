from time import monotonic
from collections.abc import Callable
from typing import NotRequired, TypedDict
from uuid import UUID

from langgraph.graph import END, START, StateGraph

from app.agent.generator import AgentGenerator
from app.agent.schemas import (
    AgentEvidence,
    AgentLimits,
    AgentResponse,
    AgentStatus,
    EvidenceSource,
    RetrievalPlan,
    ToolExecutionStatus,
    ToolExecutionTrace,
)
from app.conversation.errors import ConversationNotFoundError
from app.conversation.schemas import (
    ConversationEntityReference,
    ConversationEntitySource,
    ConversationState,
)
from app.conversation.store import ConversationStore, InMemoryConversationStore
from app.query_understanding.schemas import (
    MusicQueryIntent,
    QueryUnderstandingResult,
)
from app.query_understanding.service import QueryUnderstandingService
from app.rag.schemas import GeneratedGroundedAnswer, RAGGenerationMetadata
from app.retrieval.schemas import RerankedSearchResponse
from app.services.music.schemas import TrackData
from app.tools.errors import ToolError
from app.tools.registry import ToolRegistry
from app.tools.schemas import SearchExternalMusicProviderOutput


class AgentGraphState(TypedDict):
    query: str
    started_at: float
    iterations: int
    tool_calls: int
    tool_executions: tuple[ToolExecutionTrace, ...]
    conversation: ConversationState
    user_id: UUID | None
    understood_query: NotRequired[QueryUnderstandingResult]
    plan: NotRequired[RetrievalPlan]
    primary_result: NotRequired[RerankedSearchResponse | None]
    external_result: NotRequired[SearchExternalMusicProviderOutput | None]
    external_attempted: NotRequired[bool]
    evidence: NotRequired[tuple[AgentEvidence, ...]]
    response: NotRequired[AgentResponse]


class AgentService:
    """Bounded LangGraph orchestration over query understanding and safe tools."""

    def __init__(
        self,
        query_understanding: QueryUnderstandingService,
        tools: ToolRegistry,
        generator: AgentGenerator,
        limits: AgentLimits | None = None,
        clock: Callable[[], float] = monotonic,
        conversation_store: ConversationStore | None = None,
    ):
        self.query_understanding = query_understanding
        self.tools = tools
        self.generator = generator
        self.limits = limits or AgentLimits()
        self.clock = clock
        self.conversation_store = conversation_store or InMemoryConversationStore()
        self.graph = self._build_graph()

    def _build_graph(self):
        builder = StateGraph(AgentGraphState)
        builder.add_node("understand_query", self._understand_query)
        builder.add_node("plan_retrieval", self._plan_retrieval)
        builder.add_node("execute_primary", self._execute_primary)
        builder.add_node("evaluate_evidence", self._evaluate_evidence)
        builder.add_node("retrieve_more", self._retrieve_more)
        builder.add_node("generate_response", self._generate_response)
        builder.add_edge(START, "understand_query")
        builder.add_edge("understand_query", "plan_retrieval")
        builder.add_edge("plan_retrieval", "execute_primary")
        builder.add_edge("execute_primary", "evaluate_evidence")
        builder.add_conditional_edges(
            "evaluate_evidence",
            self._route_after_evaluation,
            {
                "retrieve_more": "retrieve_more",
                "generate": "generate_response",
            },
        )
        builder.add_edge("retrieve_more", "evaluate_evidence")
        builder.add_edge("generate_response", END)
        return builder.compile()

    def run(
        self,
        query: str,
        conversation_id: UUID | str | None = None,
        user_id: UUID | str | None = None,
    ) -> AgentResponse:
        cleaned_query = " ".join(query.split())
        if not cleaned_query:
            raise ValueError("Query must not be empty.")
        if conversation_id is None:
            conversation = self.conversation_store.create()
        else:
            identifier = (
                conversation_id
                if isinstance(conversation_id, UUID)
                else UUID(str(conversation_id))
            )
            conversation = self.conversation_store.get(identifier)
            if conversation is None:
                raise ConversationNotFoundError(
                    "Conversation was not found or expired."
                )
        final_state = self.graph.invoke(
            {
                "query": cleaned_query,
                "started_at": self.clock(),
                "iterations": 0,
                "tool_calls": 0,
                "tool_executions": (),
                "conversation": conversation,
                "user_id": UUID(str(user_id)) if user_id is not None else None,
                "external_attempted": False,
                "evidence": (),
            },
            config={"recursion_limit": 20},
        )
        response = final_state["response"]
        updated = self._advance_conversation(conversation, response)
        saved = self.conversation_store.save(
            updated,
            expected_turn_count=conversation.turn_count,
        )
        return response.model_copy(update={"conversation": saved})

    def get_conversation(self, conversation_id: UUID) -> ConversationState:
        conversation = self.conversation_store.get(conversation_id)
        if conversation is None:
            raise ConversationNotFoundError("Conversation was not found or expired.")
        return conversation

    def delete_conversation(self, conversation_id: UUID) -> None:
        if not self.conversation_store.delete(conversation_id):
            raise ConversationNotFoundError("Conversation was not found or expired.")

    def _timed_out(self, state: AgentGraphState) -> bool:
        return self.clock() - state["started_at"] >= self.limits.timeout_seconds

    def _understand_query(self, state: AgentGraphState):
        conversation = state["conversation"]
        context = (
            conversation.to_query_context() if conversation.turn_count > 0 else None
        )
        return {
            "understood_query": self.query_understanding.understand(
                state["query"],
                conversation_context=context,
            )
        }

    def _plan_retrieval(self, state: AgentGraphState):
        understood = state["understood_query"].understood
        allow_external = understood.intent in {
            MusicQueryIntent.DISCOVER_MUSIC,
            MusicQueryIntent.FIND_SIMILAR_MUSIC,
        }
        return {
            "plan": RetrievalPlan(
                semantic_query=understood.semantic_query,
                filters=understood.filters.to_vector_filter(),
                allow_external_fallback=allow_external,
            )
        }

    @staticmethod
    def _result_count(result) -> int:
        return len(result.results) if result is not None else 0

    def _failure_trace(
        self, state: AgentGraphState, tool_name: str, code: str, message: str
    ) -> ToolExecutionTrace:
        return ToolExecutionTrace(
            sequence=state["tool_calls"] + 1,
            tool_name=tool_name,
            status=ToolExecutionStatus.FAILED,
            error_code=code,
            error_message=message,
        )

    def _invoke(self, state: AgentGraphState, tool_name: str, arguments: dict):
        if self._timed_out(state):
            return None, self._failure_trace(
                state, tool_name, "agent_timeout", "The agent time budget was exhausted."
            )
        if state["tool_calls"] >= self.limits.max_tool_calls:
            return None, self._failure_trace(
                state, tool_name, "tool_limit", "The agent tool-call limit was reached."
            )
        try:
            result = self.tools.invoke(tool_name, arguments)
            trace = ToolExecutionTrace(
                sequence=state["tool_calls"] + 1,
                tool_name=tool_name,
                status=ToolExecutionStatus.SUCCEEDED,
                result_count=self._result_count(result),
            )
            return result, trace
        except ToolError as exc:
            return None, self._failure_trace(state, tool_name, exc.code, str(exc))

    def _execute_primary(self, state: AgentGraphState):
        plan = state["plan"]
        if state["understood_query"].understood.intent == MusicQueryIntent.UNKNOWN:
            return {"primary_result": None}
        arguments = {
            "query": plan.semantic_query,
            "limit": self.limits.max_evidence,
            "filters": plan.filters,
        }
        if state["user_id"] is not None:
            arguments["user_id"] = state["user_id"]
        query_popularity = state["understood_query"].understood.preferences.popularity
        if query_popularity is not None:
            arguments["query_popularity"] = query_popularity
        result, trace = self._invoke(
            state,
            plan.primary_tool,
            arguments,
        )
        return {
            "primary_result": result if isinstance(result, RerankedSearchResponse) else None,
            "iterations": state["iterations"] + 1,
            "tool_calls": state["tool_calls"] + 1,
            "tool_executions": (*state["tool_executions"], trace),
        }

    def _retrieve_more(self, state: AgentGraphState):
        plan = state["plan"]
        result, trace = self._invoke(
            state,
            plan.external_tool,
            {"query": plan.semantic_query, "limit": self.limits.max_evidence},
        )
        return {
            "external_result": (
                result if isinstance(result, SearchExternalMusicProviderOutput) else None
            ),
            "external_attempted": True,
            "iterations": state["iterations"] + 1,
            "tool_calls": state["tool_calls"] + 1,
            "tool_executions": (*state["tool_executions"], trace),
        }

    @staticmethod
    def _external_content(track: TrackData) -> str:
        facts = [f"{track.title}"]
        if track.artist:
            facts.append(f"by {track.artist.name}")
        if track.album:
            facts.append(f"from {track.album.title}")
        if track.release_date:
            facts.append(f"released {track.release_date.isoformat()}")
        if track.genres:
            facts.append(f"genres: {', '.join(track.genres)}")
        facts.append(f"provider: {track.provider}")
        return "; ".join(facts) + "."

    def _evaluate_evidence(self, state: AgentGraphState):
        evidence: list[AgentEvidence] = []
        primary = state.get("primary_result")
        if primary:
            for result in primary.results[: self.limits.max_evidence]:
                evidence.append(
                    AgentEvidence(
                        evidence_id=f"E{len(evidence) + 1}",
                        source=EvidenceSource.KNOWLEDGE,
                        content=result.content,
                        document_id=result.document_id,
                        entity_type=result.entity_type,
                        entity_id=result.entity_id,
                        metadata=result.metadata,
                        rank=result.rank,
                        score=result.reranker_score,
                        personalization=result.personalization,
                    )
                )
        external = state.get("external_result")
        if external:
            for rank, track in enumerate(external.results, start=1):
                if len(evidence) >= self.limits.max_evidence:
                    break
                evidence.append(
                    AgentEvidence(
                        evidence_id=f"E{len(evidence) + 1}",
                        source=EvidenceSource.EXTERNAL_PROVIDER,
                        content=self._external_content(track),
                        rank=rank,
                        provider=track.provider,
                        provider_id=track.provider_id,
                        track=track,
                    )
                )
        return {"evidence": tuple(evidence)}

    def _route_after_evaluation(self, state: AgentGraphState) -> str:
        enough = len(state.get("evidence", ())) >= self.limits.minimum_evidence
        plan = state["plan"]
        can_retrieve_more = (
            not enough
            and plan.allow_external_fallback
            and not state.get("external_attempted", False)
            and state["iterations"] < self.limits.max_iterations
            and state["tool_calls"] < self.limits.max_tool_calls
            and not self._timed_out(state)
        )
        return "retrieve_more" if can_retrieve_more else "generate"

    @staticmethod
    def _validate_citations(
        generated: GeneratedGroundedAnswer, evidence: tuple[AgentEvidence, ...]
    ) -> None:
        valid_ids = {item.evidence_id for item in evidence}
        cited_ids = {
            evidence_id
            for statement in (
                *generated.retrieved_facts,
                *generated.recommendation_reasoning,
            )
            for evidence_id in statement.evidence_ids
        }
        unknown = cited_ids - valid_ids
        if unknown:
            raise ValueError(f"Generated answer cited unknown evidence IDs: {sorted(unknown)}")

    def _generate_response(self, state: AgentGraphState):
        evidence = state.get("evidence", ())
        common = {
            "understood_query": state["understood_query"],
            "plan": state["plan"],
            "tool_executions": state["tool_executions"],
            "sources": evidence,
        }
        if not evidence:
            errors = tuple(
                trace.error_message
                for trace in state["tool_executions"]
                if trace.status == ToolExecutionStatus.FAILED and trace.error_message
            )
            uncertainty = errors or ("No supporting music evidence was retrieved.",)
            return {
                "response": AgentResponse(
                    status=AgentStatus.INSUFFICIENT_EVIDENCE,
                    answer="I do not have enough evidence to answer that reliably.",
                    retrieved_facts=(),
                    recommendation_reasoning=(),
                    uncertainty=uncertainty,
                    **common,
                )
            }
        if self._timed_out(state):
            return {"response": self._fallback_response(common, "The agent time budget was exhausted.")}
        try:
            generation = self.generator(state["plan"].semantic_query, evidence)
            self._validate_citations(generation.output, evidence)
            return {
                "response": AgentResponse(
                    status=AgentStatus.COMPLETED,
                    **generation.output.model_dump(),
                    generation=generation.metadata,
                    **common,
                )
            }
        except Exception:
            return {
                "response": self._fallback_response(
                    common,
                    "Evidence was retrieved, but grounded response generation failed.",
                )
            }

    @staticmethod
    def _fallback_response(common: dict, reason: str) -> AgentResponse:
        return AgentResponse(
            status=AgentStatus.FALLBACK,
            answer="I found relevant evidence but could not generate a reliable synthesis.",
            retrieved_facts=(),
            recommendation_reasoning=(),
            uncertainty=(reason,),
            generation=None,
            **common,
        )

    @staticmethod
    def _evidence_references(
        evidence: tuple[AgentEvidence, ...]
    ) -> tuple[ConversationEntityReference, ...]:
        references = []
        seen = set()
        for position, item in enumerate(evidence, start=1):
            if item.source == EvidenceSource.KNOWLEDGE:
                key = ("knowledge", item.entity_type, item.entity_id)
                reference = ConversationEntityReference(
                    source=ConversationEntitySource.KNOWLEDGE,
                    label=item.content[:500],
                    position=position,
                    entity_type=item.entity_type,
                    entity_id=item.entity_id,
                )
            else:
                key = ("external", item.provider, item.provider_id)
                reference = ConversationEntityReference(
                    source=ConversationEntitySource.EXTERNAL_PROVIDER,
                    label=item.track.title,
                    position=position,
                    provider=item.provider,
                    provider_id=item.provider_id,
                )
            if key not in seen:
                seen.add(key)
                references.append(reference)
        return tuple(references)

    def _advance_conversation(
        self,
        previous: ConversationState,
        response: AgentResponse,
    ) -> ConversationState:
        understood = response.understood_query.understood
        references = self._evidence_references(response.sources)
        resolved_intent = (
            understood.intent
            if understood.intent != MusicQueryIntent.UNKNOWN
            else previous.current_intent
        )
        filters = (
            understood.filters
            if understood.intent != MusicQueryIntent.UNKNOWN
            else previous.previous_filters
        )
        semantic_query = (
            understood.semantic_query
            if understood.intent != MusicQueryIntent.UNKNOWN
            else previous.previous_semantic_query
        )
        recommendations = previous.previous_recommendations
        if references and resolved_intent in {
            MusicQueryIntent.DISCOVER_MUSIC,
            MusicQueryIntent.FIND_SIMILAR_MUSIC,
        }:
            recommendations = references
        return previous.model_copy(
            update={
                "turn_count": previous.turn_count + 1,
                "current_intent": resolved_intent,
                "previous_query": response.understood_query.original_query,
                "previous_semantic_query": semantic_query,
                "previous_filters": filters,
                "previous_entities": references or previous.previous_entities,
                "previous_recommendations": recommendations,
            }
        )
