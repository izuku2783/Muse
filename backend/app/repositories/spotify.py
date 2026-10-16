from uuid import UUID

from collections.abc import Collection

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.models.spotify import MusicProviderMapping, SpotifyConnection
from app.models.music import Album, Artist, Track


class SpotifyConnectionRepository:
    def __init__(self, session: Session):
        self.session = session

    def get_for_user(self, user_id: UUID) -> SpotifyConnection | None:
        return self.session.scalar(
            select(SpotifyConnection).where(SpotifyConnection.user_id == str(user_id))
        )

    def get_for_account(self, account_id: str) -> SpotifyConnection | None:
        return self.session.scalar(
            select(SpotifyConnection).where(
                SpotifyConnection.spotify_account_id == account_id
            )
        )

    def add(self, connection: SpotifyConnection) -> SpotifyConnection:
        self.session.add(connection)
        self.session.flush()
        return connection

    def delete_for_user(self, user_id: UUID) -> bool:
        result = self.session.execute(
            delete(SpotifyConnection).where(SpotifyConnection.user_id == str(user_id))
        )
        return result.rowcount == 1


class MusicProviderMappingRepository:
    def __init__(self, session: Session):
        self.session = session

    def get_by_provider(
        self, entity_type: str, provider: str, provider_id: str
    ) -> MusicProviderMapping | None:
        return self.session.scalar(
            select(MusicProviderMapping).where(
                MusicProviderMapping.entity_type == entity_type,
                MusicProviderMapping.provider == provider,
                MusicProviderMapping.provider_id == provider_id,
            )
        )

    def resolve_entity_ids(
        self,
        entity_type: str,
        provider: str,
        provider_ids: Collection[str],
    ) -> frozenset[int]:
        if not provider_ids:
            return frozenset()
        return frozenset(
            self.session.scalars(
                select(MusicProviderMapping.entity_id).where(
                    MusicProviderMapping.entity_type == entity_type,
                    MusicProviderMapping.provider == provider,
                    MusicProviderMapping.provider_id.in_(provider_ids),
                )
            ).all()
        )

    def add(
        self,
        entity_type: str,
        entity_id: int,
        provider: str,
        provider_id: str,
    ) -> MusicProviderMapping:
        model_by_type = {"artist": Artist, "album": Album, "track": Track}
        model = model_by_type.get(entity_type)
        if model is None:
            raise ValueError("Provider mapping entity type is invalid.")
        if self.session.get(model, entity_id) is None:
            raise ValueError("Provider mapping requires an existing canonical entity.")
        mapping = MusicProviderMapping(
            entity_type=entity_type,
            entity_id=entity_id,
            provider=provider,
            provider_id=provider_id,
        )
        self.session.add(mapping)
        self.session.flush()
        return mapping
