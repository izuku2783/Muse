from app.rag.context import ContextBuilder
from app.rag.generator import RAGGenerator
from app.rag.schemas import (
    GeneratedGroundedAnswer,
    GroundedResponse,
    RAGGenerationRequest,
)
from app.retrieval.reranked import RerankedRetriever
from app.vector_store.schemas import VectorSearchFilter


SYSTEM_INSTRUCTIONS = """You are Muse, a grounded music research assistant.
Use only the supplied structured evidence. Never invent music metadata.
Keep retrieved facts, recommendation reasoning, and uncertainty separate.
Every factual or recommendation statement must cite one or more supplied evidence IDs.
The answer field may summarize cited statements but must introduce no additional claims.
If evidence is insufficient, say so explicitly in the answer and uncertainty fields."""


class GroundingError(RuntimeError):
    pass


class RAGService:
    """Retrieve evidence, request a structured answer, and validate grounding."""

    def __init__(
        self,
        retriever: RerankedRetriever,
        generator: RAGGenerator,
        context_builder: ContextBuilder | None = None,
    ):
        self.retriever = retriever
        self.generator = generator
        self.context_builder = context_builder or ContextBuilder()

    @staticmethod
    def _validate_citations(
        generated: GeneratedGroundedAnswer,
        evidence_ids: set[str],
    ) -> None:
        statements = (
            *generated.retrieved_facts,
            *generated.recommendation_reasoning,
        )
        invalid = {
            evidence_id
            for statement in statements
            for evidence_id in statement.evidence_ids
            if evidence_id not in evidence_ids
        }
        if invalid:
            raise GroundingError(
                f"Generated answer cited unknown evidence IDs: {sorted(invalid)}"
            )

    def answer(
        self,
        query: str,
        filters: VectorSearchFilter | None = None,
        semantic_score_threshold: float | None = None,
    ) -> GroundedResponse:
        retrieved = self.retriever.search(
            query,
            limit=self.context_builder.max_evidence,
            filters=filters,
            semantic_score_threshold=semantic_score_threshold,
        )
        context = self.context_builder.build(retrieved)
        if not context.evidence:
            return GroundedResponse(
                answer="I do not have enough evidence to answer that reliably.",
                retrieved_facts=(),
                recommendation_reasoning=(),
                uncertainty=("No supporting music documents were retrieved.",),
                sources=(),
            )
        generation = self.generator(
            RAGGenerationRequest(
                system_instructions=SYSTEM_INSTRUCTIONS,
                query=context.query,
                evidence=context.evidence,
            )
        )
        self._validate_citations(
            generation.output,
            {item.evidence_id for item in context.evidence},
        )
        return GroundedResponse(
            **generation.output.model_dump(),
            sources=context.evidence,
            generation=generation.metadata,
        )
