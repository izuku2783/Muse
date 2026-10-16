from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from app.llm.errors import LLMConfigurationError, LLMError
from app.rag.runtime import get_rag_service
from app.rag.schemas import GroundedResponse
from app.rag.service import GroundingError, RAGService
from app.vector_store.schemas import VectorSearchFilter


router = APIRouter(
    prefix="/answer",
    tags=["rag"],
)


class RAGAnswerRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, pattern=r".*\S.*")
    filters: VectorSearchFilter | None = None
    semantic_score_threshold: float | None = Field(default=None, ge=-1, le=1)


@router.post("", response_model=GroundedResponse)
def answer_with_evidence(
    request: RAGAnswerRequest,
    service: RAGService = Depends(get_rag_service),
) -> GroundedResponse:
    try:
        return service.answer(
            request.query,
            filters=request.filters,
            semantic_score_threshold=request.semantic_score_threshold,
        )
    except LLMConfigurationError as exc:
        raise HTTPException(
            status_code=503,
            detail="RAG generation is not configured.",
        ) from exc
    except (LLMError, GroundingError) as exc:
        raise HTTPException(
            status_code=502,
            detail="RAG generation failed.",
        ) from exc
