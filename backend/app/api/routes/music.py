from collections.abc import Generator

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.database.session import get_db
from app.services.music.errors import MusicProviderError
from app.services.music.service import MusicService


router = APIRouter(
    prefix="/search",
    tags=["music"],
)


class MusicSearchRequest(BaseModel):
    query: str = Field(min_length=1, pattern=r".*\S.*")
    limit: int = Field(
        default=10,
        ge=1,
        le=100,
    )
    offset: int = Field(
        default=0,
        ge=0,
    )


def get_music_service(
    db: Session = Depends(get_db),
) -> Generator[MusicService, None, None]:
    service = MusicService(db)
    try:
        yield service
    finally:
        service.close()


@router.post("")
def search_music(
    request: MusicSearchRequest,
    service: MusicService = Depends(get_music_service),
):
    try:
        results = service.search_music(
            query=request.query,
            limit=request.limit,
            offset=request.offset,
        )

        return {
            "query": request.query,
            "count": len(results),
            "results": [
                result.model_dump()
                for result in results
            ],
        }

    except MusicProviderError as exc:
        raise HTTPException(
            status_code=502,
            detail="Music provider request failed.",
        ) from exc
