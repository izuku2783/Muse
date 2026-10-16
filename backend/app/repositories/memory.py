from uuid import UUID

from sqlalchemy import delete, select, update
from sqlalchemy.orm import Session

from app.memory.schemas import PreferenceCategory
from app.models.memory import UserPreference


class UserPreferenceRepository:
    def __init__(self, session: Session):
        self.session = session

    def list(
        self,
        user_id: UUID,
        category: PreferenceCategory | None = None,
    ) -> list[UserPreference]:
        statement = select(UserPreference).where(
            UserPreference.user_id == str(user_id)
        )
        if category is not None:
            statement = statement.where(UserPreference.category == category.value)
        statement = statement.order_by(
            UserPreference.category,
            UserPreference.normalized_value,
            UserPreference.id,
        )
        return list(self.session.scalars(statement).all())

    def get(self, user_id: UUID, preference_id: int) -> UserPreference | None:
        return self.session.scalar(
            select(UserPreference).where(
                UserPreference.id == preference_id,
                UserPreference.user_id == str(user_id),
            )
        )

    def add(self, preference: UserPreference) -> UserPreference:
        self.session.add(preference)
        self.session.flush()
        return preference

    def update_if_version(
        self,
        user_id: UUID,
        preference_id: int,
        expected_version: int,
        values: dict,
    ) -> bool:
        result = self.session.execute(
            update(UserPreference)
            .where(
                UserPreference.id == preference_id,
                UserPreference.user_id == str(user_id),
                UserPreference.version == expected_version,
            )
            .values(**values, version=expected_version + 1)
        )
        return result.rowcount == 1

    def delete_if_version(
        self,
        user_id: UUID,
        preference_id: int,
        expected_version: int,
    ) -> bool:
        result = self.session.execute(
            delete(UserPreference).where(
                UserPreference.id == preference_id,
                UserPreference.user_id == str(user_id),
                UserPreference.version == expected_version,
            )
        )
        return result.rowcount == 1
