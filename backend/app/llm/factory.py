from app.config.settings import Settings, get_settings
from app.llm.errors import LLMConfigurationError
from app.llm.groq import GroqLLMProvider
from app.llm.provider import LLMProvider


def get_llm_provider(settings: Settings | None = None) -> LLMProvider:
    configured = settings or get_settings()
    provider_name = configured.llm_provider.strip().casefold()
    if provider_name == "groq":
        return GroqLLMProvider(
            api_key=configured.groq_api_key,
            model=configured.llm_model,
            base_url=configured.groq_base_url,
            timeout_seconds=configured.llm_timeout_seconds,
            max_completion_tokens=configured.llm_max_completion_tokens,
            max_retries=configured.llm_max_retries,
            retry_backoff_seconds=configured.llm_retry_backoff_seconds,
        )
    raise LLMConfigurationError(
        f"Unsupported LLM provider: {configured.llm_provider}"
    )
