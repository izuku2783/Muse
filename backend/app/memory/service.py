import unicodedata
from collections.abc import Callable
from datetime import datetime, timezone
from uuid import UUID

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.memory.errors import (
    UserPreferenceConflictError,
    UserPreferenceNotFoundError,
)
from app.memory.schemas import (
    PreferenceCategory,
    PreferenceSource,
    UserPreferenceCreate,
    UserPreferenceProfile,
    UserPreferenceRecord,
    UserPreferenceUpdate,
)
from app.models.memory import UserPreference
from app.repositories.memory import UserPreferenceRepository


def normalize_preference_value(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).split()).casefold()


class UserMemoryService:
    """Explicit, durable preference operations; no chat is stored implicitly."""

    def __init__(
        self,
        session: Session,
        clock: Callable[[], datetime] | None = None,
    ):
        self.session = session
        self.repository = UserPreferenceRepository(session)
        self.clock = clock or (lambda: datetime.now(timezone.utc))

    @staticmethod
    def _record(preference: UserPreference) -> UserPreferenceRecord:
        return UserPreferenceRecord(
            id=preference.id,
            user_id=UUID(preference.user_id),
            category=preference.category,
            value=preference.value,
            stance=preference.stance,
            source=preference.source,
            source_detail=preference.source_detail,
            confidence=preference.confidence,
            version=preference.version,
            created_at=preference.created_at,
            updated_at=preference.updated_at,
        )

    def get_profile(
        self,
        user_id: UUID,
        category: PreferenceCategory | None = None,
    ) -> UserPreferenceProfile:
        return UserPreferenceProfile(
            user_id=user_id,
            preferences=tuple(
                self._record(item) for item in self.repository.list(user_id, category)
            ),
        )

    def create(
        self,
        user_id: UUID,
        request: UserPreferenceCreate,
        source: PreferenceSource = PreferenceSource.EXPLICIT_USER,
        confidence: float = 1.0,
        source_detail: str | None = None,
    ) -> UserPreferenceRecord:
        if not 0 <= confidence <= 1:
            raise ValueError("Preference confidence must be between 0 and 1.")
        now = self.clock()
        preference = UserPreference(
            user_id=str(user_id),
            category=request.category.value,
            value=request.value,
            normalized_value=normalize_preference_value(request.value),
            stance=request.stance.value,
            source=source.value,
            source_detail=source_detail,
            confidence=confidence,
            version=1,
            created_at=now,
            updated_at=now,
        )
        try:
            self.repository.add(preference)
            self.session.commit()
            self.session.refresh(preference)
        except IntegrityError as exc:
            self.session.rollback()
            raise UserPreferenceConflictError(
                "A preference for this category and value already exists."
            ) from exc
        return self._record(preference)

    def update(
        self,
        user_id: UUID,
        preference_id: int,
        request: UserPreferenceUpdate,
    ) -> UserPreferenceRecord:
        current = self.repository.get(user_id, preference_id)
        if current is None:
            raise UserPreferenceNotFoundError("User preference was not found.")
        category = PreferenceCategory(current.category)
        proposed = UserPreferenceCreate(
            category=category,
            value=request.value if request.value is not None else current.value,
            stance=request.stance if request.stance is not None else current.stance,
        )
        values = {
            "value": proposed.value,
            "normalized_value": normalize_preference_value(proposed.value),
            "stance": proposed.stance.value,
            "updated_at": self.clock(),
        }
        try:
            changed = self.repository.update_if_version(
                user_id,
                preference_id,
                request.expected_version,
                values,
            )
            if not changed:
                self.session.rollback()
                raise UserPreferenceConflictError(
                    "User preference changed before this update was applied."
                )
            self.session.commit()
        except IntegrityError as exc:
            self.session.rollback()
            raise UserPreferenceConflictError(
                "A preference for this category and value already exists."
            ) from exc
        updated = self.repository.get(user_id, preference_id)
        return self._record(updated)

    def delete(
        self,
        user_id: UUID,
        preference_id: int,
        expected_version: int,
    ) -> None:
        if self.repository.get(user_id, preference_id) is None:
            raise UserPreferenceNotFoundError("User preference was not found.")
        if not self.repository.delete_if_version(
            user_id, preference_id, expected_version
        ):
            self.session.rollback()
            raise UserPreferenceConflictError(
                "User preference changed before deletion was applied."
            )
        self.session.commit()
