from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Response
from sqlalchemy.orm import Session

from app.database.session import get_db
from app.memory.errors import (
    UserPreferenceConflictError,
    UserPreferenceNotFoundError,
)
from app.memory.schemas import (
    PreferenceCategory,
    UserPreferenceCreate,
    UserPreferenceProfile,
    UserPreferenceRecord,
    UserPreferenceUpdate,
)
from app.memory.service import UserMemoryService


router = APIRouter(prefix="/users/{user_id}/preferences", tags=["user-memory"])


def get_user_memory_service(db: Session = Depends(get_db)) -> UserMemoryService:
    return UserMemoryService(db)


def _translate_memory_error(exc: Exception) -> HTTPException:
    if isinstance(exc, UserPreferenceNotFoundError):
        return HTTPException(status_code=404, detail=str(exc))
    return HTTPException(status_code=409, detail=str(exc))


@router.get("", response_model=UserPreferenceProfile)
def list_user_preferences(
    user_id: UUID,
    category: PreferenceCategory | None = None,
    service: UserMemoryService = Depends(get_user_memory_service),
) -> UserPreferenceProfile:
    return service.get_profile(user_id, category=category)


@router.post("", response_model=UserPreferenceRecord, status_code=201)
def create_user_preference(
    user_id: UUID,
    request: UserPreferenceCreate,
    service: UserMemoryService = Depends(get_user_memory_service),
) -> UserPreferenceRecord:
    try:
        return service.create(user_id, request)
    except UserPreferenceConflictError as exc:
        raise _translate_memory_error(exc) from exc


@router.patch("/{preference_id}", response_model=UserPreferenceRecord)
def update_user_preference(
    user_id: UUID,
    preference_id: int,
    request: UserPreferenceUpdate,
    service: UserMemoryService = Depends(get_user_memory_service),
) -> UserPreferenceRecord:
    try:
        return service.update(user_id, preference_id, request)
    except (UserPreferenceNotFoundError, UserPreferenceConflictError) as exc:
        raise _translate_memory_error(exc) from exc


@router.delete("/{preference_id}", status_code=204)
def delete_user_preference(
    user_id: UUID,
    preference_id: int,
    expected_version: int = Query(ge=1),
    service: UserMemoryService = Depends(get_user_memory_service),
) -> Response:
    try:
        service.delete(user_id, preference_id, expected_version)
    except (UserPreferenceNotFoundError, UserPreferenceConflictError) as exc:
        raise _translate_memory_error(exc) from exc
    return Response(status_code=204)
