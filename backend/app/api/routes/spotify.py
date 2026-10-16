from typing import Literal
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Response
from sqlalchemy.orm import Session

from app.database.session import get_db
from app.spotify.errors import (
    SpotifyAuthorizationError,
    SpotifyConfigurationError,
    SpotifyConnectionConflictError,
    SpotifyConnectionNotFoundError,
    SpotifyProviderError,
    SpotifyRateLimitError,
)
from app.spotify.runtime import get_spotify_runtime
from app.spotify.schemas import (
    SpotifyAuthorizationStart,
    SpotifyConnectionStatus,
    SpotifyRecentPlayback,
    SpotifyTopTracks,
)
from app.spotify.service import SpotifyService


router = APIRouter(tags=["spotify-integration"])


def get_spotify_service(db: Session = Depends(get_db)) -> SpotifyService:
    try:
        runtime = get_spotify_runtime()
    except SpotifyConfigurationError as exc:
        raise HTTPException(
            status_code=503, detail="Spotify integration is not configured."
        ) from exc
    return SpotifyService(
        db,
        integration=runtime.integration,
        state_store=runtime.state_store,
        token_cipher=runtime.token_cipher,
    )


def _translate_spotify_error(exc: Exception) -> HTTPException:
    if isinstance(exc, SpotifyConnectionNotFoundError):
        return HTTPException(status_code=404, detail=str(exc))
    if isinstance(exc, SpotifyConnectionConflictError):
        return HTTPException(status_code=409, detail=str(exc))
    if isinstance(exc, SpotifyAuthorizationError):
        return HTTPException(status_code=401, detail=str(exc))
    if isinstance(exc, SpotifyRateLimitError):
        headers = (
            {"Retry-After": str(exc.retry_after_seconds)}
            if exc.retry_after_seconds is not None
            else None
        )
        return HTTPException(status_code=429, detail=str(exc), headers=headers)
    return HTTPException(status_code=502, detail="Spotify request failed.")


@router.post(
    "/users/{user_id}/integrations/spotify/authorize",
    response_model=SpotifyAuthorizationStart,
)
def start_spotify_authorization(
    user_id: UUID,
    service: SpotifyService = Depends(get_spotify_service),
) -> SpotifyAuthorizationStart:
    return service.start_authorization(user_id)


@router.get(
    "/integrations/spotify/callback",
    response_model=SpotifyConnectionStatus,
)
def complete_spotify_authorization(
    state: str = Query(min_length=1),
    code: str | None = None,
    error: str | None = None,
    service: SpotifyService = Depends(get_spotify_service),
) -> SpotifyConnectionStatus:
    try:
        return service.complete_authorization(code=code, state=state, error=error)
    except SpotifyAuthorizationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except (SpotifyConnectionConflictError, SpotifyProviderError) as exc:
        raise _translate_spotify_error(exc) from exc


@router.get(
    "/users/{user_id}/integrations/spotify",
    response_model=SpotifyConnectionStatus,
)
def get_spotify_connection(
    user_id: UUID,
    service: SpotifyService = Depends(get_spotify_service),
) -> SpotifyConnectionStatus:
    return service.get_status(user_id)


@router.delete("/users/{user_id}/integrations/spotify", status_code=204)
def disconnect_spotify(
    user_id: UUID,
    service: SpotifyService = Depends(get_spotify_service),
) -> Response:
    try:
        service.disconnect(user_id)
    except SpotifyConnectionNotFoundError as exc:
        raise _translate_spotify_error(exc) from exc
    return Response(status_code=204)


@router.get(
    "/users/{user_id}/integrations/spotify/top-tracks",
    response_model=SpotifyTopTracks,
)
def get_spotify_top_tracks(
    user_id: UUID,
    limit: int = Query(default=20, ge=1, le=50),
    time_range: Literal["short_term", "medium_term", "long_term"] = "medium_term",
    service: SpotifyService = Depends(get_spotify_service),
) -> SpotifyTopTracks:
    try:
        return service.get_top_tracks(user_id, limit=limit, time_range=time_range)
    except (SpotifyConnectionNotFoundError, SpotifyAuthorizationError, SpotifyProviderError) as exc:
        raise _translate_spotify_error(exc) from exc


@router.get(
    "/users/{user_id}/integrations/spotify/recently-played",
    response_model=SpotifyRecentPlayback,
)
def get_spotify_recently_played(
    user_id: UUID,
    limit: int = Query(default=20, ge=1, le=50),
    service: SpotifyService = Depends(get_spotify_service),
) -> SpotifyRecentPlayback:
    try:
        return service.get_recent_playback(user_id, limit=limit)
    except (SpotifyConnectionNotFoundError, SpotifyAuthorizationError, SpotifyProviderError) as exc:
        raise _translate_spotify_error(exc) from exc
