from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Muse"
    app_version: str = "0.1.0"
    environment: str = "development"
    debug: bool = True

    database_url: str = Field(
        default="postgresql+psycopg://muse:muse_password@localhost:5432/muse",
        alias="DATABASE_URL",
    )

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