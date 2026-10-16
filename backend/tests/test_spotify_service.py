from datetime import datetime, timezone
from uuid import UUID

import pytest
from cryptography.fernet import Fernet
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from app.database.session import Base
from app.models.music import Artist
from app.models.spotify import SpotifyConnection
from app.repositories.spotify import MusicProviderMappingRepository
from app.spotify.crypto import SpotifyTokenCipher
from app.spotify.errors import (
    SpotifyAuthorizationError,
    SpotifyConnectionConflictError,
    SpotifyConnectionNotFoundError,
)
from app.spotify.oauth_state import SpotifyOAuthStateStore
from app.spotify.schemas import SpotifyProfile, SpotifyToken
from app.spotify.service import SpotifyService


USER_ID = UUID("00000000-0000-0000-0000-000000000001")
OTHER_USER_ID = UUID("00000000-0000-0000-0000-000000000002")


class FakeSpotifyIntegration:
    def __init__(self):
        self.profile = SpotifyProfile(
            account_id="stable-account-id", id="display-id", display_name="Muse User"
        )
        self.token = SpotifyToken(
            access_token="access-secret",
            refresh_token="refresh-secret",
            expires_in=3600,
            scope="user-top-read user-read-recently-played user-read-private",
        )
        self.refresh_token = SpotifyToken(
            access_token="refreshed-access",
            expires_in=3600,
            scope=self.token.scope,
        )
        self.calls = []

    def build_authorization_url(self, state, challenge):
        self.calls.append(("authorize", state, challenge))
        return f"https://accounts.spotify.com/authorize?state={state}"

    def exchange_code(self, code, verifier):
        self.calls.append(("exchange", code, verifier))
        return self.token

    def get_profile(self, access_token):
        self.calls.append(("profile", access_token))
        return self.profile

    def refresh_access_token(self, refresh_token):
        self.calls.append(("refresh", refresh_token))
        return self.refresh_token

    def get_top_tracks(self, access_token, *, limit, time_range):
        self.calls.append(("top", access_token, limit, time_range))
        return ()

    def get_recently_played(self, access_token, *, limit):
        self.calls.append(("recent", access_token, limit))
        return ()


@pytest.fixture
def session():
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as db:
        yield db


@pytest.fixture
def service(session):
    now = datetime(2026, 10, 16, tzinfo=timezone.utc)
    return SpotifyService(
        session,
        FakeSpotifyIntegration(),
        SpotifyOAuthStateStore(clock=lambda: now),
        SpotifyTokenCipher(Fernet.generate_key().decode()),
        clock=lambda: now,
    )


def connect(service, user_id=USER_ID):
    start = service.start_authorization(user_id)
    state = service.integration.calls[-1][1]
    return service.complete_authorization(code="code", state=state), start


def test_oauth_connection_uses_account_id_and_encrypts_tokens(session, service):
    status, start = connect(service)
    stored = session.scalar(select(SpotifyConnection))

    assert str(start.authorization_url).startswith("https://accounts.spotify.com/")
    assert status.connected is True
    assert stored.spotify_account_id == "stable-account-id"
    assert "access-secret" not in stored.encrypted_access_token
    assert "refresh-secret" not in stored.encrypted_refresh_token


def test_spotify_account_cannot_be_linked_to_two_muse_users(service):
    connect(service)
    state = service.start_authorization(OTHER_USER_ID)
    actual_state = service.integration.calls[-1][1]
    with pytest.raises(SpotifyConnectionConflictError):
        service.complete_authorization(code="code", state=actual_state)


def test_status_and_disconnect_are_user_scoped(service):
    assert service.get_status(USER_ID).connected is False
    connect(service)
    assert service.get_status(USER_ID).spotify_display_name == "Muse User"
    assert service.get_status(OTHER_USER_ID).connected is False
    service.disconnect(USER_ID)
    with pytest.raises(SpotifyConnectionNotFoundError):
        service.disconnect(USER_ID)


def test_expiring_token_is_refreshed_and_rotated_token_is_optional(session, service):
    connect(service)
    connection = session.scalar(select(SpotifyConnection))
    connection.access_token_expires_at = datetime(2020, 1, 1, tzinfo=timezone.utc)
    session.commit()

    service.get_top_tracks(USER_ID, limit=10, time_range="medium_term")

    assert ("refresh", "refresh-secret") in service.integration.calls
    assert ("top", "refreshed-access", 10, "medium_term") in service.integration.calls
    assert service.token_cipher.decrypt(connection.encrypted_refresh_token) == "refresh-secret"


def test_invalid_refresh_discards_credentials_and_requires_reauthorization(session, service):
    connect(service)
    connection = session.scalar(select(SpotifyConnection))
    connection.access_token_expires_at = datetime(2020, 1, 1, tzinfo=timezone.utc)
    session.commit()
    service.integration.refresh_access_token = lambda token: (_ for _ in ()).throw(
        SpotifyAuthorizationError("expired")
    )

    with pytest.raises(SpotifyAuthorizationError):
        service.get_recent_playback(USER_ID, limit=5)

    assert service.get_status(USER_ID).reauthorization_required is True
    assert connection.encrypted_refresh_token is None


def test_missing_scope_is_rejected_before_provider_call(service):
    connect(service)
    connection = service.repository.get_for_user(USER_ID)
    connection.granted_scopes = "user-read-private"
    service.session.commit()
    with pytest.raises(SpotifyAuthorizationError, match="user-top-read"):
        service.get_top_tracks(USER_ID, limit=10, time_range="short_term")


def test_provider_mapping_keeps_internal_id_canonical(session):
    repository = MusicProviderMappingRepository(session)
    artist = Artist(provider="musicbrainz", provider_id="mb-artist", name="Neon Waves")
    session.add(artist)
    session.flush()
    mapping = repository.add("artist", artist.id, "spotify", "spotify-artist-id")
    session.commit()

    loaded = repository.get_by_provider("artist", "spotify", "spotify-artist-id")
    assert loaded.id == mapping.id
    assert loaded.entity_id == artist.id
    assert loaded.provider_id != str(loaded.entity_id)


def test_provider_mapping_rejects_noncanonical_or_invalid_entities(session):
    repository = MusicProviderMappingRepository(session)
    with pytest.raises(ValueError, match="existing canonical"):
        repository.add("track", 999, "spotify", "spotify-track-id")
    with pytest.raises(ValueError, match="type is invalid"):
        repository.add("playlist", 1, "spotify", "spotify-playlist-id")
