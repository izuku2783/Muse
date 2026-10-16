from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI

from app.api.routes.health import router as health_router
from app.api.routes.agent import router as agent_router
from app.api.routes.music import router as music_router
from app.api.routes.memory import router as memory_router
from app.api.routes.rag import router as rag_router
from app.api.routes.spotify import router as spotify_router
from app.config.settings import get_settings
from app.database.init_db import initialize_database
from app.logging_config import configure_logging
from app.rag.runtime import close_rag_runtime
from app.agent.runtime import close_agent_runtime
from app.spotify.runtime import close_spotify_runtime


configure_logging()

logger = structlog.get_logger()

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(
        "application_starting",
        app_name=settings.app_name,
        version=settings.app_version,
        environment=settings.environment,
    )

    initialize_database()

    yield

    close_agent_runtime()
    close_rag_runtime()
    close_spotify_runtime()
    logger.info("application_stopping")


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Muse - AI Music Research & Discovery Agent",
    lifespan=lifespan,
)


app.include_router(health_router)
app.include_router(music_router)
app.include_router(rag_router)
app.include_router(agent_router)
app.include_router(memory_router)
app.include_router(spotify_router)
