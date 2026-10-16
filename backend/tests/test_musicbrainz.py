from datetime import date

import httpx
import pytest

from app.services.music.errors import MusicProviderError
from app.services.music.musicbrainz import MusicBrainzProvider


@pytest.fixture
def recording():
    return {
        "id": "recording-1",
        "title": "Night Drive",
        "length": 245000,
        "isrcs": ["USABC1234567"],
        "genres": [{"name": "Synthwave"}, {"name": "electronic"}],
        "artist-credit": [{"artist": {"id": "artist-1", "name": "Neon Waves", "sort-name": "Neon Waves", "type": "Group", "country": "US"}}],
        "releases": [
            {"id": "live-release", "title": "Live at Midnight", "status": "Official", "date": "2019-01-01", "release-group": {"primary-type": "Album", "secondary-types": ["Live"]}, "artist-credit": [{"artist": {"id": "artist-1"}}]},
            {"id": "album-1", "title": "After Dark", "status": "Official", "date": "2020-03-04", "release-group": {"primary-type": "Album"}, "artist-credit": [{"artist": {"id": "artist-1"}}]},
        ],
    }


def test_extract_track_normalizes_complete_recording(recording):
    track = MusicBrainzProvider._extract_track(recording)
    assert track.title == "Night Drive"
    assert track.artist.name == "Neon Waves"
    assert track.artist.country == "US"
    assert track.album.provider_id == "album-1"
    assert track.album.album_type == "Album"
    assert track.release_date == date(2020, 3, 4)
    assert track.duration_ms == 245000
    assert track.isrc == "USABC1234567"
    assert track.genres == ["electronic", "Synthwave"]


def test_release_selection_prefers_earliest_official_canonical_release(recording):
    later = dict(recording["releases"][1], id="album-later", date="2021-01-01")
    recording["releases"].insert(1, later)
    assert MusicBrainzProvider._extract_album(recording).provider_id == "album-1"


@pytest.mark.parametrize("status", ["Bootleg", "Promotion", None])
def test_release_selection_rejects_non_official_releases(recording, status):
    recording["releases"] = [dict(recording["releases"][1], status=status)]
    assert MusicBrainzProvider._extract_album(recording) is None


def test_release_selection_rejects_inconsistent_artist(recording):
    recording["releases"] = [dict(recording["releases"][1], **{"artist-credit": [{"artist": {"id": "someone-else"}}]})]
    assert MusicBrainzProvider._extract_album(recording) is None


def test_extract_track_handles_missing_optional_metadata(recording):
    for key in ("artist-credit", "releases", "isrcs", "genres", "length"):
        recording.pop(key, None)
    track = MusicBrainzProvider._extract_track(recording)
    assert track.artist is None
    assert track.album is None
    assert track.isrc is None
    assert track.genres == []
    assert track.duration_ms is None


def test_extract_track_rejects_missing_recording_id(recording):
    recording.pop("id")
    with pytest.raises(ValueError, match="missing an ID"):
        MusicBrainzProvider._extract_track(recording)


def test_search_tracks_returns_normalized_results_and_pagination(recording, monkeypatch):
    provider = MusicBrainzProvider("Muse tests (test@example.com)")
    captured = {}
    def request(endpoint, params):
        captured.update(endpoint=endpoint, params=params)
        return {"recordings": [recording]}
    monkeypatch.setattr(provider, "_request", request)
    results = provider.search_tracks(" dark synthwave ", limit=200, offset=20)
    assert results[0].provider == "musicbrainz"
    assert captured == {"endpoint": "/recording", "params": {"query": "dark synthwave", "limit": 100, "offset": 20, "fmt": "json"}}


def test_search_tracks_skips_malformed_records(recording, monkeypatch):
    provider = MusicBrainzProvider("Muse tests (test@example.com)")
    monkeypatch.setattr(provider, "_request", lambda *_: {"recordings": [{"title": "Bad"}, recording]})
    assert [item.provider_id for item in provider.search_tracks("query")] == ["recording-1"]


def test_search_tracks_handles_empty_query_without_request(monkeypatch):
    provider = MusicBrainzProvider("Muse tests (test@example.com)")
    monkeypatch.setattr(provider, "_request", lambda *_: pytest.fail("request should not run"))
    assert provider.search_tracks("   ") == []


def test_search_tracks_handles_empty_provider_response(monkeypatch):
    provider = MusicBrainzProvider("Muse tests (test@example.com)")
    monkeypatch.setattr(provider, "_request", lambda *_: {})
    assert provider.search_tracks("unknown") == []


def test_request_translates_http_errors():
    def handler(request):
        return httpx.Response(503, request=request)
    provider = MusicBrainzProvider("Muse tests (test@example.com)", client=httpx.Client(transport=httpx.MockTransport(handler)))
    with pytest.raises(MusicProviderError, match="request failed"):
        provider.search_tracks("query")


def test_request_normalizes_leading_slash_and_sets_user_agent():
    captured = {}
    def handler(request):
        captured["request"] = request
        return httpx.Response(200, json={"recordings": []}, request=request)
    provider = MusicBrainzProvider("Muse tests (test@example.com)", client=httpx.Client(transport=httpx.MockTransport(handler)))
    provider.search_tracks("query")
    assert captured["request"].url.path == "/ws/2/recording"
    assert captured["request"].headers["user-agent"] == "Muse tests (test@example.com)"
