import pytest
from pydantic import ValidationError

from app.llm.schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    ChatRole,
    TokenUsage,
    ToolCall,
    ToolDefinition,
)


def tool_call() -> ToolCall:
    return ToolCall(id="call-1", name="search_music", arguments={"query": "jazz"})


def test_chat_messages_support_a_complete_tool_round_trip():
    messages = (
        ChatMessage(role=ChatRole.USER, content="Find jazz"),
        ChatMessage(role=ChatRole.ASSISTANT, tool_calls=(tool_call(),)),
        ChatMessage(
            role=ChatRole.TOOL,
            content='{"results": []}',
            tool_call_id="call-1",
        ),
    )
    request = ChatCompletionRequest(messages=messages)
    assert request.messages[1].tool_calls[0].name == "search_music"


@pytest.mark.parametrize(
    "message",
    [
        {"role": "user"},
        {"role": "user", "content": "   "},
        {"role": "tool", "content": "result"},
        {"role": "user", "content": "hello", "tool_call_id": "call-1"},
        {"role": "system", "content": "hello", "tool_calls": [tool_call()]},
        {"role": "assistant"},
    ],
)
def test_chat_messages_reject_invalid_role_payloads(message):
    with pytest.raises(ValidationError):
        ChatMessage.model_validate(message)


def test_required_tool_choice_requires_a_tool_definition():
    with pytest.raises(ValidationError, match="at least one tool"):
        ChatCompletionRequest(
            messages=(ChatMessage(role=ChatRole.USER, content="Find jazz"),),
            tool_choice="required",
        )


def test_completion_response_exposes_tool_calls_and_usage():
    response = ChatCompletionResponse(
        tool_calls=(tool_call(),),
        model="test-model",
        finish_reason="tool_calls",
        usage=TokenUsage(
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
        ),
    )
    assert response.usage.total_tokens == 15
    assert response.tool_calls[0].arguments == {"query": "jazz"}


def test_tool_definition_rejects_unsafe_name():
    with pytest.raises(ValidationError):
        ToolDefinition(
            name="search music",
            description="Search music",
            parameters={"type": "object"},
        )


def test_tool_definition_requires_object_parameter_schema():
    with pytest.raises(ValidationError, match="object schema"):
        ToolDefinition(
            name="search_music",
            description="Search music",
            parameters={"type": "array"},
        )
