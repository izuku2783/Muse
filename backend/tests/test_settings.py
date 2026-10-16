import pytest
from pydantic import ValidationError

from app.config.settings import Settings


def test_release_debug_environment_value_is_treated_as_disabled(monkeypatch):
    monkeypatch.setenv("DEBUG", "release")
    settings = Settings(_env_file=None)
    assert settings.debug is False


def test_empty_groq_api_key_is_normalized_to_none(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "")
    settings = Settings(_env_file=None)
    assert settings.groq_api_key is None


def test_groq_generation_settings_can_be_configured(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setenv("LLM_MODEL", "test-model")
    monkeypatch.setenv("LLM_TIMEOUT_SECONDS", "12.5")
    monkeypatch.setenv("LLM_MAX_COMPLETION_TOKENS", "512")
    monkeypatch.setenv("LLM_MAX_RETRIES", "3")
    settings = Settings(_env_file=None)
    assert settings.groq_api_key == "test-key"
    assert settings.llm_model == "test-model"
    assert settings.llm_timeout_seconds == 12.5
    assert settings.llm_max_completion_tokens == 512
    assert settings.llm_max_retries == 3


def test_legacy_groq_generation_names_remain_environment_aliases(monkeypatch):
    monkeypatch.setenv("GROQ_MODEL", "legacy-model")
    monkeypatch.setenv("GROQ_TIMEOUT_SECONDS", "9")
    monkeypatch.setenv("GROQ_MAX_COMPLETION_TOKENS", "256")
    settings = Settings(_env_file=None)
    assert settings.llm_model == "legacy-model"
    assert settings.llm_timeout_seconds == 9
    assert settings.llm_max_completion_tokens == 256


def test_agent_execution_limits_can_be_configured(monkeypatch):
    monkeypatch.setenv("AGENT_MAX_ITERATIONS", "3")
    monkeypatch.setenv("AGENT_MAX_TOOL_CALLS", "4")
    monkeypatch.setenv("AGENT_TIMEOUT_SECONDS", "45")
    monkeypatch.setenv("AGENT_MAX_EVIDENCE", "6")
    monkeypatch.setenv("CONVERSATION_TTL_SECONDS", "900")
    monkeypatch.setenv("CONVERSATION_MAX_ACTIVE", "200")
    settings = Settings(_env_file=None)
    assert settings.agent_max_iterations == 3
    assert settings.agent_max_tool_calls == 4
    assert settings.agent_timeout_seconds == 45
    assert settings.agent_max_evidence == 6
    assert settings.conversation_ttl_seconds == 900
    assert settings.conversation_max_active == 200


def test_spotify_is_optional_and_can_be_configured_without_a_client_secret(monkeypatch):
    monkeypatch.setenv("SPOTIFY_CLIENT_ID", "client-id")
    monkeypatch.setenv(
        "SPOTIFY_REDIRECT_URI",
        "https://muse.example.com/integrations/spotify/callback",
    )
    monkeypatch.setenv("SPOTIFY_TOKEN_ENCRYPTION_KEY", "key")
    settings = Settings(_env_file=None)
    assert settings.spotify_client_id == "client-id"
    assert settings.spotify_redirect_uri.startswith("https://")
    assert settings.spotify_token_encryption_key.get_secret_value() == "key"


def test_spotify_redirect_rejects_localhost_and_insecure_remote_http(monkeypatch):
    for value in (
        "http://localhost:8000/callback",
        "http://muse.example.com/callback",
    ):
        monkeypatch.setenv("SPOTIFY_REDIRECT_URI", value)
        with pytest.raises(ValidationError):
            Settings(_env_file=None)


def test_personalization_weights_and_signal_limit_are_configurable(monkeypatch):
    monkeypatch.setenv("PERSONALIZATION_RELEVANCE_WEIGHT", "0.5")
    monkeypatch.setenv("PERSONALIZATION_PREFERENCE_WEIGHT", "0.3")
    monkeypatch.setenv("PERSONALIZATION_NOVELTY_WEIGHT", "0.1")
    monkeypatch.setenv("PERSONALIZATION_DIVERSITY_WEIGHT", "0.1")
    monkeypatch.setenv("PERSONALIZATION_SPOTIFY_SIGNAL_LIMIT", "30")
    settings = Settings(_env_file=None)
    assert settings.personalization_relevance_weight == 0.5
    assert settings.personalization_preference_weight == 0.3
    assert settings.personalization_spotify_signal_limit == 30


def test_personalization_weights_must_sum_to_one(monkeypatch):
    monkeypatch.setenv("PERSONALIZATION_RELEVANCE_WEIGHT", "0.4")
    with pytest.raises(ValidationError, match="sum to 1.0"):
        Settings(_env_file=None)


@pytest.mark.parametrize(
    "name,value",
    [
        ("LLM_TIMEOUT_SECONDS", "0"),
        ("LLM_MAX_COMPLETION_TOKENS", "63"),
        ("LLM_MAX_RETRIES", "6"),
        ("AGENT_MAX_ITERATIONS", "0"),
        ("AGENT_MAX_TOOL_CALLS", "11"),
        ("AGENT_TIMEOUT_SECONDS", "0"),
        ("AGENT_MAX_EVIDENCE", "21"),
        ("CONVERSATION_TTL_SECONDS", "0"),
        ("CONVERSATION_MAX_ACTIVE", "0"),
    ],
)
def test_groq_generation_settings_reject_invalid_limits(monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    with pytest.raises(ValidationError):
        Settings(_env_file=None)
