from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Response
from pydantic import BaseModel, ConfigDict, Field

from app.agent.runtime import get_agent_service
from app.agent.schemas import AgentResponse
from app.agent.service import AgentService
from app.conversation.errors import (
    ConversationConflictError,
    ConversationNotFoundError,
)
from app.conversation.schemas import ConversationState
from app.llm.errors import LLMConfigurationError, LLMError


router = APIRouter(prefix="/agent", tags=["agent"])


class AgentRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, pattern=r".*\S.*")
    conversation_id: UUID | None = None
    user_id: UUID | None = None


@router.post("", response_model=AgentResponse)
def run_agent(
    request: AgentRequest,
    service: AgentService = Depends(get_agent_service),
) -> AgentResponse:
    try:
        arguments = {"conversation_id": request.conversation_id}
        if request.user_id is not None:
            arguments["user_id"] = request.user_id
        return service.run(request.query, **arguments)
    except ConversationNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ConversationConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except LLMConfigurationError as exc:
        raise HTTPException(
            status_code=503,
            detail="Agent reasoning is not configured.",
        ) from exc
    except LLMError as exc:
        raise HTTPException(
            status_code=502,
            detail="Agent reasoning failed.",
        ) from exc


@router.get("/conversations/{conversation_id}", response_model=ConversationState)
def get_conversation(
    conversation_id: UUID,
    service: AgentService = Depends(get_agent_service),
) -> ConversationState:
    try:
        return service.get_conversation(conversation_id)
    except ConversationNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.delete("/conversations/{conversation_id}", status_code=204)
def delete_conversation(
    conversation_id: UUID,
    service: AgentService = Depends(get_agent_service),
) -> Response:
    try:
        service.delete_conversation(conversation_id)
    except ConversationNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return Response(status_code=204)
