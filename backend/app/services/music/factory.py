from app.services.music.musicbrainz import MusicBrainzProvider
from app.services.music.provider import MusicProvider


def get_music_provider() -> MusicProvider:
    return MusicBrainzProvider(
        user_agent=(
            "Muse/0.1.0 "
            "AI Music Research & Discovery Agent "
            "(development)"
        )
    )