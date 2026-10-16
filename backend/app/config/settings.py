from functools import lru_cache

from pydantic import AliasChoices, Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Muse"
    app_version: str = "0.1.0"
    environment: str = "development"
    debug: bool = True
    musicbrainz_user_agent: str = Field(
        default="Muse/0.1.0 (development)",
        alias="MUSICBRAINZ_USER_AGENT",
    )

    database_url: str = Field(
        default="postgresql+psycopg://muse:muse_password@localhost:5432/muse",
        alias="DATABASE_URL",
    )
    qdrant_url: str = Field(
        default="http://localhost:6333",
        alias="QDRANT_URL",
    )
    qdrant_collection: str = Field(
        default="muse_knowledge",
        alias="QDRANT_COLLECTION",
    )
    qdrant_api_key: str | None = Field(default=None, alias="QDRANT_API_KEY")
    embedding_provider: str = Field(default="fastembed", alias="EMBEDDING_PROVIDER")
    embedding_model: str = Field(
        default="BAAI/bge-small-en-v1.5",
        alias="EMBEDDING_MODEL",
    )
    embedding_version: str = Field(
        default="fastembed-0.8.1-query-passage-v1",
        alias="EMBEDDING_VERSION",
    )
    embedding_dimension: int = Field(
        default=384,
        ge=8,
        le=4096,
        alias="EMBEDDING_DIMENSION",
    )
    embedding_cache_dir: str = Field(
        default="data/models",
        alias="EMBEDDING_CACHE_DIR",
    )
    reranker_provider: str = Field(default="fastembed", alias="RERANKER_PROVIDER")
    reranker_model: str = Field(
        default="Xenova/ms-marco-MiniLM-L-6-v2",
        alias="RERANKER_MODEL",
    )
    reranker_version: str = Field(
        default="fastembed-0.8.1",
        alias="RERANKER_VERSION",
    )
    reranker_cache_dir: str = Field(
        default="data/models",
        alias="RERANKER_CACHE_DIR",
    )
    reranker_batch_size: int = Field(
        default=32,
        ge=1,
        le=256,
        alias="RERANKER_BATCH_SIZE",
    )
    llm_provider: str = Field(default="groq", alias="LLM_PROVIDER")
    llm_model: str = Field(
        default="qwen/qwen3.8-27b",
        min_length=1,
        validation_alias=AliasChoices("LLM_MODEL", "GROQ_MODEL"),
    )
    groq_api_key: str | None = Field(default=None, alias="GROQ_API_KEY")
    groq_base_url: str = Field(
        default="https://api.groq.com/openai/v1",
        min_length=1,
        alias="GROQ_BASE_URL",
    )
    llm_timeout_seconds: float = Field(
        default=30.0,
        gt=0,
        le=120,
        validation_alias=AliasChoices(
            "LLM_TIMEOUT_SECONDS",
            "GROQ_TIMEOUT_SECONDS",
        ),
    )
    llm_max_completion_tokens: int = Field(
        default=1_024,
        ge=64,
        le=8_192,
        validation_alias=AliasChoices(
            "LLM_MAX_COMPLETION_TOKENS",
            "GROQ_MAX_COMPLETION_TOKENS",
        ),
    )
    llm_max_retries: int = Field(
        default=2,
        ge=0,
        le=5,
        alias="LLM_MAX_RETRIES",
    )
    llm_retry_backoff_seconds: float = Field(
        default=0.5,
        ge=0,
        le=10,
        alias="LLM_RETRY_BACKOFF_SECONDS",
    )
    agent_max_iterations: int = Field(
        default=2,
        ge=1,
        le=5,
        alias="AGENT_MAX_ITERATIONS",
    )
    agent_max_tool_calls: int = Field(
        default=2,
        ge=1,
        le=10,
        alias="AGENT_MAX_TOOL_CALLS",
    )
    agent_timeout_seconds: float = Field(
        default=60.0,
        gt=0,
        le=300,
        alias="AGENT_TIMEOUT_SECONDS",
    )
    agent_max_evidence: int = Field(
        default=5,
        ge=1,
        le=20,
        alias="AGENT_MAX_EVIDENCE",
    )
    conversation_ttl_seconds: float = Field(
        default=1_800,
        gt=0,
        le=86_400,
        alias="CONVERSATION_TTL_SECONDS",
    )
    conversation_max_active: int = Field(
        default=1_000,
        ge=1,
        le=100_000,
        alias="CONVERSATION_MAX_ACTIVE",
    )
    spotify_client_id: str | None = Field(default=None, alias="SPOTIFY_CLIENT_ID")
    spotify_redirect_uri: str = Field(
        default="http://127.0.0.1:8000/integrations/spotify/callback",
        alias="SPOTIFY_REDIRECT_URI",
    )
    spotify_token_encryption_key: SecretStr | None = Field(
        default=None,
        alias="SPOTIFY_TOKEN_ENCRYPTION_KEY",
    )
    spotify_http_timeout_seconds: float = Field(
        default=10.0,
        gt=0,
        le=60,
        alias="SPOTIFY_HTTP_TIMEOUT_SECONDS",
    )
    spotify_oauth_state_ttl_seconds: float = Field(
        default=600,
        gt=0,
        le=1_800,
        alias="SPOTIFY_OAUTH_STATE_TTL_SECONDS",
    )
    personalization_relevance_weight: float = Field(
        default=0.5, ge=0, le=1, alias="PERSONALIZATION_RELEVANCE_WEIGHT"
    )
    personalization_preference_weight: float = Field(
        default=0.2, ge=0, le=1, alias="PERSONALIZATION_PREFERENCE_WEIGHT"
    )
    personalization_novelty_weight: float = Field(
        default=0.15, ge=0, le=1, alias="PERSONALIZATION_NOVELTY_WEIGHT"
    )
    personalization_diversity_weight: float = Field(
        default=0.15, ge=0, le=1, alias="PERSONALIZATION_DIVERSITY_WEIGHT"
    )
    personalization_spotify_signal_limit: int = Field(
        default=20,
        ge=1,
        le=50,
        alias="PERSONALIZATION_SPOTIFY_SIGNAL_LIMIT",
    )

    @field_validator(
        "qdrant_api_key",
        "groq_api_key",
        "spotify_client_id",
        "spotify_token_encryption_key",
        mode="before",
    )
    @classmethod
    def empty_secret_is_none(cls, value):
        return None if value == "" else value

    @field_validator("debug", mode="before")
    @classmethod
    def normalize_debug_mode(cls, value):
        if isinstance(value, str):
            normalized = value.strip().casefold()
            if normalized in {"release", "production", "prod"}:
                return False
            if normalized in {"development", "dev"}:
                return True
        return value

    @field_validator("spotify_redirect_uri")
    @classmethod
    def validate_spotify_redirect_uri(cls, value: str) -> str:
        normalized = value.strip()
        if normalized.startswith("https://"):
            return normalized
        if normalized.startswith("http://127.0.0.1") or normalized.startswith(
            "http://[::1]"
        ):
            return normalized
        raise ValueError(
            "Spotify redirect URI must use HTTPS or an explicit loopback IP."
        )

    @model_validator(mode="after")
    def validate_personalization_weights(self):
        total = (
            self.personalization_relevance_weight
            + self.personalization_preference_weight
            + self.personalization_novelty_weight
            + self.personalization_diversity_weight
        )
        if abs(total - 1.0) > 1e-9:
            raise ValueError("Personalization weights must sum to 1.0.")
        return self

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        populate_by_name=True,
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
