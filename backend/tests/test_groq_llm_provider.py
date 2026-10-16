import json
from enum import Enum

import httpx
import pytest
from pydantic import BaseModel, ConfigDict

from app.llm.errors import (
    LLMConfigurationError,
    LLMProviderError,
    LLMRateLimitError,
    LLMResponseError,
)
from app.llm.groq import GroqLLMProvider, _strict_json_schema
from app.llm.schemas import (
    ChatCompletionRequest,
    ChatMessage,
    ChatRole,
    ToolCall,
    ToolDefinition,
)


class StructuredAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: str
    tags: tuple[str, ...] = ()


class OptionalTone(str, Enum):
    DARK = "dark"
    BRIGHT = "bright"


class NullableStructuredAnswer(BaseModel):
    tone: OptionalTone | None = None
    year: int | None = None


def request(**kwargs) -> ChatCompletionRequest:
    return ChatCompletionRequest(
        messages=(ChatMessage(role=ChatRole.USER, content="Find jazz"),),
        **kwargs,
    )


def response_body(content="Try Kind of Blue", **overrides):
    body = {
        "model": "returned-model",
        "choices": [
            {
                "message": {"content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        },
    }
    body.update(overrides)
    return body


def provider_with_handler(handler, **kwargs):
    return GroqLLMProvider(
        api_key="test-key",
        model="default-model",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        sleeper=lambda _: None,
        **kwargs,
    )


def test_chat_completion_uses_model_override_and_tracks_usage():
    captured = {}

    def handler(http_request):
        captured["auth"] = http_request.headers["Authorization"]
        captured["payload"] = json.loads(http_request.content)
        return httpx.Response(200, json=response_body(), request=http_request)

    provider = provider_with_handler(handler)
    result = provider.complete(request(model="request-model"))

    assert result.content == "Try Kind of Blue"
    assert result.model == "returned-model"
    assert result.usage.total_tokens == 15
    assert captured["auth"] == "Bearer test-key"
    assert captured["payload"]["model"] == "request-model"


def test_tool_definitions_are_sent_and_tool_calls_are_parsed():
    captured = {}

    def handler(http_request):
        captured["payload"] = json.loads(http_request.content)
        body = response_body(content=None)
        body["choices"][0]["message"]["tool_calls"] = [
            {
                "id": "call-1",
                "type": "function",
                "function": {
                    "name": "search_music",
                    "arguments": '{"query":"jazz"}',
                },
            }
        ]
        body["choices"][0]["finish_reason"] = "tool_calls"
        return httpx.Response(200, json=body, request=http_request)

    provider = provider_with_handler(handler)
    result = provider.complete(
        request(
            tools=(
                ToolDefinition(
                    name="search_music",
                    description="Search the music catalog.",
                    parameters={
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                        "additionalProperties": False,
                    },
                ),
            ),
            tool_choice="required",
        )
    )

    assert result.tool_calls == (
        ToolCall(id="call-1", name="search_music", arguments={"query": "jazz"}),
    )
    payload = captured["payload"]
    assert payload["tool_choice"] == "required"
    assert payload["tools"][0]["function"]["name"] == "search_music"


def test_assistant_and_tool_messages_are_serialized_for_follow_up_calls():
    captured = {}

    def handler(http_request):
        captured["payload"] = json.loads(http_request.content)
        return httpx.Response(200, json=response_body(), request=http_request)

    provider = provider_with_handler(handler)
    provider.complete(
        ChatCompletionRequest(
            messages=(
                ChatMessage(role=ChatRole.USER, content="Find jazz"),
                ChatMessage(
                    role=ChatRole.ASSISTANT,
                    tool_calls=(
                        ToolCall(
                            id="call-1",
                            name="search_music",
                            arguments={"query": "jazz"},
                        ),
                    ),
                ),
                ChatMessage(
                    role=ChatRole.TOOL,
                    content='{"results":[]}',
                    tool_call_id="call-1",
                ),
            )
        )
    )
    messages = captured["payload"]["messages"]
    assert messages[1]["tool_calls"][0]["function"]["arguments"] == (
        '{"query":"jazz"}'
    )
    assert messages[2]["tool_call_id"] == "call-1"


def test_structured_completion_uses_strict_schema_and_validates_output():
    captured = {}

    def handler(http_request):
        captured["payload"] = json.loads(http_request.content)
        return httpx.Response(
            200,
            json=response_body(json.dumps({"answer": "Jazz", "tags": ["cool"]})),
            request=http_request,
        )

    provider = provider_with_handler(handler)
    completion = provider.complete_structured(
        request(),
        StructuredAnswer,
        "structured_answer",
    )

    assert completion.output.tags == ("cool",)
    assert completion.response.usage.total_tokens == 15
    schema = captured["payload"]["response_format"]["json_schema"]
    assert schema["strict"] is True
    assert schema["schema"]["required"] == ["answer", "tags"]
    assert "default" not in schema["schema"]["properties"]["tags"]
    assert "title" not in schema["schema"]


def test_strict_schema_collapses_nullable_primitive_and_enum_unions():
    schema = _strict_json_schema(NullableStructuredAnswer.model_json_schema())
    tone = schema["properties"]["tone"]
    year = schema["properties"]["year"]
    assert "anyOf" not in tone
    assert tone["type"] == ["string", "null"]
    assert tone["enum"] == ["dark", "bright", None]
    assert "anyOf" not in year
    assert year["type"] == ["integer", "null"]


def test_retryable_failure_retries_then_returns_success():
    attempts = []
    delays = []

    def handler(http_request):
        attempts.append(http_request)
        if len(attempts) == 1:
            return httpx.Response(
                429,
                headers={"Retry-After": "0.25"},
                request=http_request,
            )
        return httpx.Response(200, json=response_body(), request=http_request)

    provider = GroqLLMProvider(
        api_key="test-key",
        model="default-model",
        max_retries=1,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        sleeper=delays.append,
    )
    assert provider.complete(request()).content == "Try Kind of Blue"
    assert len(attempts) == 2
    assert delays == [0.25]


def test_network_failure_retries_then_returns_success():
    attempts = []

    def handler(http_request):
        attempts.append(http_request)
        if len(attempts) == 1:
            raise httpx.ConnectError("offline", request=http_request)
        return httpx.Response(200, json=response_body(), request=http_request)

    provider = provider_with_handler(handler, max_retries=1)
    assert provider.complete(request()).content == "Try Kind of Blue"
    assert len(attempts) == 2


def test_exhausted_rate_limit_raises_specific_error():
    def handler(http_request):
        return httpx.Response(429, request=http_request)

    provider = provider_with_handler(handler, max_retries=1)
    with pytest.raises(LLMRateLimitError):
        provider.complete(request())


def test_non_retryable_provider_error_is_not_retried():
    attempts = []

    def handler(http_request):
        attempts.append(http_request)
        return httpx.Response(400, request=http_request)

    provider = provider_with_handler(handler, max_retries=2)
    with pytest.raises(LLMProviderError, match="status 400"):
        provider.complete(request())
    assert len(attempts) == 1


def test_missing_api_key_fails_before_network_call():
    provider = GroqLLMProvider(
        api_key=None,
        model="default-model",
        client=httpx.Client(
            transport=httpx.MockTransport(
                lambda request: pytest.fail("network should not be called")
            )
        ),
    )
    with pytest.raises(LLMConfigurationError, match="GROQ_API_KEY"):
        provider.complete(request())


@pytest.mark.parametrize(
    "body",
    [
        {"choices": []},
        response_body(content=None),
        response_body(
            content=None,
            choices=[
                {
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call-1",
                                "function": {
                                    "name": "search_music",
                                    "arguments": "not-json",
                                },
                            }
                        ],
                    }
                }
            ],
        ),
    ],
)
def test_malformed_completion_is_normalized(body):
    provider = provider_with_handler(
        lambda http_request: httpx.Response(
            200,
            json=body,
            request=http_request,
        )
    )
    with pytest.raises(LLMResponseError):
        provider.complete(request())


def test_schema_invalid_structured_output_is_normalized():
    provider = provider_with_handler(
        lambda http_request: httpx.Response(
            200,
            json=response_body('{"answer": 123, "tags": []}'),
            request=http_request,
        )
    )
    with pytest.raises(LLMResponseError):
        provider.complete_structured(request(), StructuredAnswer, "answer")
