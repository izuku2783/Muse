from abc import ABC, abstractmethod

from app.llm.schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    StructuredCompletion,
    StructuredOutputT,
)


class LLMProvider(ABC):
    """Provider-independent chat, structured-output, and tool-call boundary."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def default_model(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def complete(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        raise NotImplementedError

    @abstractmethod
    def complete_structured(
        self,
        request: ChatCompletionRequest,
        response_model: type[StructuredOutputT],
        schema_name: str,
    ) -> StructuredCompletion[StructuredOutputT]:
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError
