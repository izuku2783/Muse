from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI

from app.api.routes.health import router as health_router
from app.api.routes.music import router as music_router
from app.config.settings import get_settings
from app.database.init_db import initialize_database
from app.logging_config import configure_logging


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

    logger.info("application_stopping")


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Muse - AI Music Research & Discovery Agent",
    lifespan=lifespan,
)


app.include_router(health_router)
app.include_router(music_router)