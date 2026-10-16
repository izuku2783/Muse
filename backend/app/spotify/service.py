from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from uuid import UUID

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.models.spotify import SpotifyConnection
from app.repositories.spotify import SpotifyConnectionRepository
from app.spotify.crypto import SpotifyTokenCipher
from app.spotify.errors import (
    SpotifyAuthorizationError,
    SpotifyConnectionConflictError,
    SpotifyConnectionNotFoundError,
)
from app.spotify.oauth_state import SpotifyOAuthStateStore
from app.spotify.provider import SpotifyIntegration
from app.spotify.schemas import (
    SpotifyAuthorizationStart,
    SpotifyConnectionStatus,
    SpotifyRecentPlayback,
    SpotifyToken,
    SpotifyTopTracks,
)


class SpotifyService:
    def __init__(
        self,
        session: Session,
        integration: SpotifyIntegration,
        state_store: SpotifyOAuthStateStore,
        token_cipher: SpotifyTokenCipher,
        clock: Callable[[], datetime] | None = None,
    ):
        self.session = session
        self.integration = integration
        self.state_store = state_store
        self.token_cipher = token_cipher
        self.repository = SpotifyConnectionRepository(session)
        self.clock = clock or (lambda: datetime.now(timezone.utc))

    def start_authorization(self, user_id: UUID) -> SpotifyAuthorizationStart:
        state, _, challenge, expires_at = self.state_store.create(user_id)
        return SpotifyAuthorizationStart(
            authorization_url=self.integration.build_authorization_url(state, challenge),
            expires_at=expires_at,
        )

    def complete_authorization(
        self,
        *,
        code: str | None,
        state: str,
        error: str | None = None,
    ) -> SpotifyConnectionStatus:
        pending = self.state_store.consume(state)
        if error is not None:
            raise SpotifyAuthorizationError("Spotify authorization was denied or failed.")
        if not code or not code.strip():
            raise SpotifyAuthorizationError("Spotify authorization code is missing.")
        token = self.integration.exchange_code(code, pending.code_verifier)
        if not token.refresh_token:
            raise SpotifyAuthorizationError("Spotify did not return a refresh token.")
        profile = self.integration.get_profile(token.access_token)
        other = self.repository.get_for_account(profile.account_id)
        if other is not None and other.user_id != str(pending.user_id):
            raise SpotifyConnectionConflictError(
                "This Spotify account is already connected to another Muse user."
            )
        now = self.clock()
        connection = self.repository.get_for_user(pending.user_id)
        if connection is None:
            connection = SpotifyConnection(
                user_id=str(pending.user_id),
                spotify_account_id=profile.account_id,
                created_at=now,
            )
        connection.spotify_account_id = profile.account_id
        connection.spotify_user_id = profile.id
        connection.display_name = profile.display_name
        connection.encrypted_access_token = self.token_cipher.encrypt(token.access_token)
        connection.encrypted_refresh_token = self.token_cipher.encrypt(token.refresh_token)
        connection.access_token_expires_at = now + timedelta(seconds=token.expires_in)
        connection.granted_scopes = " ".join(sorted(token.scopes))
        connection.reauthorization_required = False
        connection.updated_at = now
        try:
            if connection.id is None:
                self.repository.add(connection)
            self.session.commit()
            self.session.refresh(connection)
        except IntegrityError as exc:
            self.session.rollback()
            raise SpotifyConnectionConflictError(
                "Spotify account connection conflicts with an existing connection."
            ) from exc
        return self._status(pending.user_id, connection)

    def get_status(self, user_id: UUID) -> SpotifyConnectionStatus:
        return self._status(user_id, self.repository.get_for_user(user_id))

    def disconnect(self, user_id: UUID) -> None:
        if not self.repository.delete_for_user(user_id):
            self.session.rollback()
            raise SpotifyConnectionNotFoundError("Spotify connection was not found.")
        self.session.commit()

    def get_top_tracks(
        self, user_id: UUID, *, limit: int, time_range: str
    ) -> SpotifyTopTracks:
        access_token = self._access_token(user_id, required_scope="user-top-read")
        return SpotifyTopTracks(
            items=self.integration.get_top_tracks(
                access_token, limit=limit, time_range=time_range
            ),
            time_range=time_range,
        )

    def get_recent_playback(
        self, user_id: UUID, *, limit: int
    ) -> SpotifyRecentPlayback:
        access_token = self._access_token(
            user_id, required_scope="user-read-recently-played"
        )
        return SpotifyRecentPlayback(
            items=self.integration.get_recently_played(access_token, limit=limit)
        )

    def _access_token(self, user_id: UUID, required_scope: str) -> str:
        connection = self.repository.get_for_user(user_id)
        if connection is None:
            raise SpotifyConnectionNotFoundError("Spotify connection was not found.")
        if connection.reauthorization_required:
            raise SpotifyAuthorizationError("Spotify reauthorization is required.")
        if required_scope not in connection.granted_scopes.split():
            raise SpotifyAuthorizationError(
                f"Spotify connection does not grant the required {required_scope} scope."
            )
        if not connection.encrypted_access_token or not connection.encrypted_refresh_token:
            raise SpotifyAuthorizationError("Spotify reauthorization is required.")
        expires_at = connection.access_token_expires_at
        if expires_at is None or self._as_utc(expires_at) <= self.clock() + timedelta(seconds=30):
            return self._refresh(connection)
        return self.token_cipher.decrypt(connection.encrypted_access_token)

    def _refresh(self, connection: SpotifyConnection) -> str:
        refresh_token = self.token_cipher.decrypt(connection.encrypted_refresh_token)
        try:
            token = self.integration.refresh_access_token(refresh_token)
        except SpotifyAuthorizationError:
            connection.encrypted_access_token = None
            connection.encrypted_refresh_token = None
            connection.access_token_expires_at = None
            connection.reauthorization_required = True
            connection.updated_at = self.clock()
            self.session.commit()
            raise
        self._apply_refreshed_token(connection, token, refresh_token)
        self.session.commit()
        return token.access_token

    def _apply_refreshed_token(
        self,
        connection: SpotifyConnection,
        token: SpotifyToken,
        previous_refresh_token: str,
    ) -> None:
        now = self.clock()
        connection.encrypted_access_token = self.token_cipher.encrypt(token.access_token)
        connection.encrypted_refresh_token = self.token_cipher.encrypt(
            token.refresh_token or previous_refresh_token
        )
        connection.access_token_expires_at = now + timedelta(seconds=token.expires_in)
        if token.scope:
            connection.granted_scopes = " ".join(sorted(token.scopes))
        connection.updated_at = now

    @staticmethod
    def _as_utc(value: datetime) -> datetime:
        return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value

    @staticmethod
    def _status(
        user_id: UUID, connection: SpotifyConnection | None
    ) -> SpotifyConnectionStatus:
        if connection is None:
            return SpotifyConnectionStatus(user_id=user_id, connected=False)
        return SpotifyConnectionStatus(
            user_id=user_id,
            connected=not connection.reauthorization_required,
            reauthorization_required=connection.reauthorization_required,
            spotify_display_name=connection.display_name,
            granted_scopes=tuple(sorted(connection.granted_scopes.split())),
            connected_at=connection.created_at,
        )
