import pytest

from app.api.routes.music import get_music_service
from app.main import app
from app.services.music.errors import MusicProviderError
from app.services.music.schemas import ArtistData, TrackData


class FakeMusicService:
    def __init__(self, results=None, error=None):
        self.results = results or []
        self.error = error
        self.calls = []

    def search_music(self, query, limit=10, offset=0):
        self.calls.append((query, limit, offset))
        if self.error:
            raise self.error
        return self.results


@pytest.fixture
def override_music_service():
    services = []

    def install(service):
        services.append(service)
        app.dependency_overrides[get_music_service] = lambda: service
        return service

    yield install
    app.dependency_overrides.pop(get_music_service, None)


def test_search_endpoint_returns_normalized_tracks(client, override_music_service):
    service = override_music_service(
        FakeMusicService(
            [
                TrackData(
                    provider_id="track-1",
                    provider="musicbrainz",
                    title="Dark Signal",
                    artist=ArtistData(provider_id="artist-1", name="Neon Waves"),
                )
            ]
        )
    )
    response = client.post(
        "/search", json={"query": "dark synthwave", "limit": 5, "offset": 10}
    )
    assert response.status_code == 200
    assert response.json()["count"] == 1
    assert response.json()["results"][0]["provider"] == "musicbrainz"
    assert service.calls == [("dark synthwave", 5, 10)]


@pytest.mark.parametrize(
    "payload",
    [
        {"query": ""},
        {"query": "   "},
        {"query": "music", "limit": 0},
        {"query": "music", "limit": 101},
        {"query": "music", "offset": -1},
    ],
)
def test_search_endpoint_validates_request(client, override_music_service, payload):
    override_music_service(FakeMusicService())
    assert client.post("/search", json=payload).status_code == 422


def test_search_endpoint_returns_empty_results(client, override_music_service):
    override_music_service(FakeMusicService())
    response = client.post("/search", json={"query": "unknown"})
    assert response.status_code == 200
    assert response.json() == {"query": "unknown", "count": 0, "results": []}


def test_search_endpoint_translates_provider_failure(client, override_music_service):
    override_music_service(FakeMusicService(error=MusicProviderError("unavailable")))
    response = client.post("/search", json={"query": "music"})
    assert response.status_code == 502
    assert response.json() == {"detail": "Music provider request failed."}
