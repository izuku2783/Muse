from uuid import UUID
from types import SimpleNamespace

from app.spotify.errors import SpotifyProviderError
from app.spotify.personalization import SpotifyListeningSignalSource
from app.spotify.schemas import (
    SpotifyArtistSummary,
    SpotifyPlaybackItem,
    SpotifyRecentPlayback,
    SpotifyTopTracks,
    SpotifyTrackSummary,
)


USER_ID = UUID("00000000-0000-0000-0000-000000000001")
RUNTIME = SimpleNamespace(integration=object(), state_store=object(), token_cipher=object())


def track(identifier, isrc, artist):
    return SpotifyTrackSummary(
        spotify_id=identifier,
        name=f"Track {identifier}",
        external_ids={"isrc": isrc},
        artists=(SpotifyArtistSummary(spotify_id=f"artist-{artist}", name=artist),),
    )


def test_spotify_signal_source_normalizes_top_and_recent_data(monkeypatch):
    top_track = track("top", "TOPISRC", "Top Artist")
    recent_track = track("recent", "RECENTISRC", "Recent Artist")

    class FakeService:
        def __init__(self, *args, **kwargs):
            pass

        def get_top_tracks(self, user_id, *, limit, time_range):
            assert (user_id, limit, time_range) == (USER_ID, 10, "medium_term")
            return SpotifyTopTracks(items=(top_track,), time_range=time_range)

        def get_recent_playback(self, user_id, *, limit):
            return SpotifyRecentPlayback(
                items=(
                    SpotifyPlaybackItem(
                        track=recent_track,
                        played_at="2026-10-16T10:00:00Z",
                    ),
                )
            )

    monkeypatch.setattr("app.spotify.personalization.SpotifyService", FakeService)
    signals = SpotifyListeningSignalSource(RUNTIME, limit=10).collect(USER_ID, object())

    assert signals.provider_track_ids == frozenset({"top", "recent"})
    assert signals.isrcs == frozenset({"TOPISRC", "RECENTISRC"})
    assert signals.artist_names == frozenset({"Top Artist", "Recent Artist"})


def test_spotify_signal_failure_is_optional_and_partial_data_is_retained(monkeypatch):
    top_track = track("top", "TOPISRC", "Top Artist")

    class PartialService:
        def __init__(self, *args, **kwargs):
            pass

        def get_top_tracks(self, *args, **kwargs):
            return SpotifyTopTracks(items=(top_track,), time_range="medium_term")

        def get_recent_playback(self, *args, **kwargs):
            raise SpotifyProviderError("unavailable")

    monkeypatch.setattr("app.spotify.personalization.SpotifyService", PartialService)
    signals = SpotifyListeningSignalSource(RUNTIME).collect(USER_ID, object())
    assert signals.provider_track_ids == frozenset({"top"})


def test_spotify_signal_source_returns_none_when_all_provider_reads_fail(monkeypatch):
    class FailedService:
        def __init__(self, *args, **kwargs):
            pass

        def get_top_tracks(self, *args, **kwargs):
            raise SpotifyProviderError("unavailable")

        get_recent_playback = get_top_tracks

    monkeypatch.setattr("app.spotify.personalization.SpotifyService", FailedService)
    assert SpotifyListeningSignalSource(RUNTIME).collect(USER_ID, object()) is None
