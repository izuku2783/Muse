from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.services.music.service import MusicService


router = APIRouter(
    prefix="/search",
    tags=["music"],
)


class MusicSearchRequest(BaseModel):
    query: str = Field(min_length=1)
    limit: int = Field(
        default=10,
        ge=1,
        le=100,
    )
    offset: int = Field(
        default=0,
        ge=0,
    )


@router.post("")
def search_music(request: MusicSearchRequest):
    try:
        service = MusicService()

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

    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail="Music provider request failed.",
        ) from exc