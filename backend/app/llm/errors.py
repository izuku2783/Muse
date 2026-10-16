class LLMError(RuntimeError):
    """Base error raised by the LLM boundary."""


class LLMConfigurationError(LLMError):
    """The configured provider cannot be used."""


class LLMProviderError(LLMError):
    """The provider request failed or was rejected."""


class LLMRateLimitError(LLMProviderError):
    """The provider rejected the request because of a rate limit."""


class LLMResponseError(LLMError):
    """The provider returned a malformed or schema-invalid response."""
