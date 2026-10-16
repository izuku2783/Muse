import json
import time
from collections.abc import Callable
from copy import deepcopy
from typing import Any

import httpx
from pydantic import ValidationError

from app.llm.errors import (
    LLMConfigurationError,
    LLMProviderError,
    LLMRateLimitError,
    LLMResponseError,
)
from app.llm.provider import LLMProvider
from app.llm.schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    ChatRole,
    StructuredCompletion,
    StructuredOutputT,
    TokenUsage,
    ToolCall,
)


RETRYABLE_STATUS_CODES = {408, 409, 429, 500, 502, 503, 504}
UNSUPPORTED_STRICT_SCHEMA_KEYWORDS = {
    "default",
    "examples",
    "exclusiveMaximum",
    "exclusiveMinimum",
    "maxItems",
    "maxLength",
    "maximum",
    "minItems",
    "minLength",
    "minimum",
    "pattern",
    "title",
    "uniqueItems",
}


def _strict_json_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Convert Pydantic's object schemas to Groq strict-mode requirements."""
    strict_schema = deepcopy(schema)

    def resolve_local_ref(node: dict[str, Any]) -> dict[str, Any] | None:
        reference = node.get("$ref")
        if not isinstance(reference, str) or not reference.startswith("#/$defs/"):
            return None
        definition_name = reference.removeprefix("#/$defs/")
        definition = strict_schema.get("$defs", {}).get(definition_name)
        return deepcopy(definition) if isinstance(definition, dict) else None

    def collapse_nullable_union(node: dict[str, Any]) -> None:
        branches = node.get("anyOf")
        if not isinstance(branches, list) or len(branches) != 2:
            return
        null_branches = [branch for branch in branches if branch == {"type": "null"}]
        if len(null_branches) != 1:
            return
        other = next(branch for branch in branches if branch != {"type": "null"})
        if not isinstance(other, dict):
            return
        resolved = resolve_local_ref(other) if "$ref" in other else deepcopy(other)
        if not resolved or resolved.get("type") not in {
            "string",
            "number",
            "integer",
            "boolean",
        }:
            return
        resolved["type"] = [resolved["type"], "null"]
        if "enum" in resolved and None not in resolved["enum"]:
            resolved["enum"] = [*resolved["enum"], None]
        node.clear()
        node.update(resolved)

    def visit(node: Any) -> None:
        if isinstance(node, dict):
            collapse_nullable_union(node)
            for keyword in UNSUPPORTED_STRICT_SCHEMA_KEYWORDS:
                node.pop(keyword, None)
            if node.get("type") == "object" or "properties" in node:
                properties = node.get("properties", {})
                node["additionalProperties"] = False
                node["required"] = list(properties)
            for value in node.values():
                visit(value)
        elif isinstance(node, list):
            for value in node:
                visit(value)

    visit(strict_schema)
    return strict_schema


class GroqLLMProvider(LLMProvider):
    """Groq implementation of the provider-independent LLM contract."""

    def __init__(
        self,
        api_key: str | None,
        model: str,
        base_url: str = "https://api.groq.com/openai/v1",
        timeout_seconds: float = 30.0,
        max_completion_tokens: int = 1_024,
        max_retries: int = 2,
        retry_backoff_seconds: float = 0.5,
        client: httpx.Client | None = None,
        sleeper: Callable[[float], None] = time.sleep,
    ):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.max_completion_tokens = max_completion_tokens
        self.max_retries = max_retries
        self.retry_backoff_seconds = retry_backoff_seconds
        self.client = client or httpx.Client(timeout=timeout_seconds)
        self.sleeper = sleeper
        self._owns_client = client is None

    @property
    def provider_name(self) -> str:
        return "groq"

    @property
    def default_model(self) -> str:
        return self.model

    def close(self) -> None:
        if self._owns_client:
            self.client.close()

    @staticmethod
    def _serialize_message(message: ChatMessage) -> dict[str, Any]:
        payload: dict[str, Any] = {"role": message.role.value}
        if message.content is not None:
            payload["content"] = message.content
        if message.tool_call_id is not None:
            payload["tool_call_id"] = message.tool_call_id
        if message.tool_calls:
            payload["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": json.dumps(
                            call.arguments,
                            ensure_ascii=False,
                            separators=(",", ":"),
                        ),
                    },
                }
                for call in message.tool_calls
            ]
        return payload

    def _build_payload(self, request: ChatCompletionRequest) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": request.model or self.model,
            "messages": [
                self._serialize_message(message) for message in request.messages
            ],
            "temperature": request.temperature,
            "max_completion_tokens": (
                request.max_completion_tokens or self.max_completion_tokens
            ),
        }
        if request.tools:
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": (
                            _strict_json_schema(tool.parameters)
                            if tool.strict
                            else tool.parameters
                        ),
                        "strict": tool.strict,
                    },
                }
                for tool in request.tools
            ]
            payload["tool_choice"] = request.tool_choice
        return payload

    @staticmethod
    def _retry_delay(
        response: httpx.Response | None,
        attempt: int,
        backoff_seconds: float,
    ) -> float:
        if response is not None:
            retry_after = response.headers.get("Retry-After")
            if retry_after is not None:
                try:
                    return min(max(float(retry_after), 0), 30)
                except ValueError:
                    pass
        return min(backoff_seconds * (2**attempt), 30)

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not self.api_key:
            raise LLMConfigurationError(
                "GROQ_API_KEY is required for the configured LLM provider."
            )

        for attempt in range(self.max_retries + 1):
            response = None
            try:
                response = self.client.post(
                    f"{self.base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                )
            except httpx.RequestError as exc:
                if attempt == self.max_retries:
                    raise LLMProviderError("Groq request failed.") from exc
            else:
                if response.is_success:
                    try:
                        body = response.json()
                    except json.JSONDecodeError as exc:
                        raise LLMResponseError(
                            "Groq returned a non-JSON response."
                        ) from exc
                    if not isinstance(body, dict):
                        raise LLMResponseError("Groq returned an invalid response body.")
                    return body
                if response.status_code not in RETRYABLE_STATUS_CODES:
                    raise LLMProviderError(
                        f"Groq rejected the request with status {response.status_code}."
                    )
                if attempt == self.max_retries:
                    if response.status_code == 429:
                        raise LLMRateLimitError("Groq rate limit exceeded.")
                    raise LLMProviderError(
                        f"Groq failed after retries with status {response.status_code}."
                    )

            self.sleeper(
                self._retry_delay(response, attempt, self.retry_backoff_seconds)
            )

        raise LLMProviderError("Groq request failed.")

    @staticmethod
    def _parse_tool_calls(raw_calls: Any) -> tuple[ToolCall, ...]:
        if raw_calls is None:
            return ()
        if not isinstance(raw_calls, list):
            raise LLMResponseError("Groq returned malformed tool calls.")
        parsed = []
        try:
            for raw_call in raw_calls:
                function = raw_call["function"]
                arguments = json.loads(function["arguments"])
                if not isinstance(arguments, dict):
                    raise TypeError("tool arguments must be an object")
                parsed.append(
                    ToolCall(
                        id=raw_call["id"],
                        name=function["name"],
                        arguments=arguments,
                    )
                )
        except (KeyError, TypeError, json.JSONDecodeError, ValidationError) as exc:
            raise LLMResponseError("Groq returned malformed tool calls.") from exc
        return tuple(parsed)

    @staticmethod
    def _parse_usage(raw_usage: Any) -> TokenUsage | None:
        if raw_usage is None:
            return None
        try:
            return TokenUsage(
                prompt_tokens=raw_usage["prompt_tokens"],
                completion_tokens=raw_usage["completion_tokens"],
                total_tokens=raw_usage["total_tokens"],
            )
        except (KeyError, TypeError, ValidationError) as exc:
            raise LLMResponseError("Groq returned malformed token usage.") from exc

    def _parse_response(
        self,
        body: dict[str, Any],
        fallback_model: str,
    ) -> ChatCompletionResponse:
        try:
            choice = body["choices"][0]
            message = choice["message"]
            return ChatCompletionResponse(
                content=message.get("content"),
                tool_calls=self._parse_tool_calls(message.get("tool_calls")),
                model=body.get("model") or fallback_model,
                finish_reason=choice.get("finish_reason"),
                usage=self._parse_usage(body.get("usage")),
            )
        except (KeyError, IndexError, TypeError, ValidationError) as exc:
            raise LLMResponseError("Groq returned a malformed completion.") from exc

    def _complete(
        self,
        request: ChatCompletionRequest,
        response_format: dict[str, Any] | None = None,
    ) -> ChatCompletionResponse:
        payload = self._build_payload(request)
        if response_format is not None:
            payload["response_format"] = response_format
        body = self._post(payload)
        return self._parse_response(body, payload["model"])

    def complete(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        return self._complete(request)

    def complete_structured(
        self,
        request: ChatCompletionRequest,
        response_model: type[StructuredOutputT],
        schema_name: str,
    ) -> StructuredCompletion[StructuredOutputT]:
        response = self._complete(
            request,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name,
                    "strict": True,
                    "schema": _strict_json_schema(response_model.model_json_schema()),
                },
            },
        )
        if response.content is None:
            raise LLMResponseError("Groq returned no structured output content.")
        try:
            output = response_model.model_validate_json(response.content)
        except (ValueError, ValidationError) as exc:
            raise LLMResponseError(
                "Groq returned schema-invalid structured output."
            ) from exc
        return StructuredCompletion(output=output, response=response)
