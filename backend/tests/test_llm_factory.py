import pytest

from app.config.settings import Settings
from app.llm.errors import LLMConfigurationError
from app.llm.factory import get_llm_provider
from app.llm.groq import GroqLLMProvider


def test_factory_builds_configured_groq_provider():
    settings = Settings(
        _env_file=None,
        LLM_PROVIDER="groq",
        LLM_MODEL="test-model",
        LLM_MAX_RETRIES=4,
        LLM_RETRY_BACKOFF_SECONDS=0.25,
        GROQ_API_KEY="test-key",
        GROQ_BASE_URL="https://groq.test/v1",
    )
    provider = get_llm_provider(settings)
    try:
        assert isinstance(provider, GroqLLMProvider)
        assert provider.default_model == "test-model"
        assert provider.max_retries == 4
        assert provider.retry_backoff_seconds == 0.25
        assert provider.base_url == "https://groq.test/v1"
    finally:
        provider.close()


def test_factory_rejects_unknown_provider():
    settings = Settings(_env_file=None, LLM_PROVIDER="unknown")
    with pytest.raises(LLMConfigurationError, match="unknown"):
        get_llm_provider(settings)
