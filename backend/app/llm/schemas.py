from dataclasses import dataclass
from enum import Enum
from typing import Any, Generic, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ChatRole(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class ToolCall(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    arguments: dict[str, Any]


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    role: ChatRole
    content: str | None = None
    tool_call_id: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()

    @model_validator(mode="after")
    def validate_role_payload(self):
        has_content = self.content is not None and bool(self.content.strip())
        if self.role in {ChatRole.SYSTEM, ChatRole.USER} and not has_content:
            raise ValueError(f"{self.role.value} messages require content")
        if self.role == ChatRole.TOOL:
            if not has_content or not self.tool_call_id:
                raise ValueError("tool messages require content and tool_call_id")
        elif self.tool_call_id is not None:
            raise ValueError("tool_call_id is only valid for tool messages")
        if self.tool_calls and self.role != ChatRole.ASSISTANT:
            raise ValueError("tool_calls are only valid for assistant messages")
        if self.role == ChatRole.ASSISTANT and not has_content and not self.tool_calls:
            raise ValueError("assistant messages require content or tool_calls")
        return self


class ToolDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1, pattern=r"^[A-Za-z0-9_-]+$")
    description: str = Field(min_length=1)
    parameters: dict[str, Any]
    strict: bool = True

    @model_validator(mode="after")
    def parameters_must_be_object_schema(self):
        if self.parameters.get("type") != "object":
            raise ValueError("tool parameters must be a JSON object schema")
        return self


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    messages: tuple[ChatMessage, ...] = Field(min_length=1)
    model: str | None = Field(default=None, min_length=1)
    temperature: float = Field(default=0, ge=0, le=2)
    max_completion_tokens: int | None = Field(default=None, ge=1)
    tools: tuple[ToolDefinition, ...] = ()
    tool_choice: Literal["auto", "none", "required"] = "auto"

    @model_validator(mode="after")
    def tool_choice_requires_tools(self):
        if self.tool_choice == "required" and not self.tools:
            raise ValueError("required tool choice needs at least one tool")
        return self


class TokenUsage(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)


class ChatCompletionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    content: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()
    model: str = Field(min_length=1)
    finish_reason: str | None = None
    usage: TokenUsage | None = None

    @model_validator(mode="after")
    def require_content_or_tool_calls(self):
        if not self.content and not self.tool_calls:
            raise ValueError("completion requires content or tool calls")
        return self


StructuredOutputT = TypeVar("StructuredOutputT", bound=BaseModel)


@dataclass(frozen=True)
class StructuredCompletion(Generic[StructuredOutputT]):
    output: StructuredOutputT
    response: ChatCompletionResponse
