from app.config.settings import get_settings
from app.services.music.musicbrainz import MusicBrainzProvider
from app.services.music.provider import MusicProvider


def get_music_provider() -> MusicProvider:
    settings = get_settings()
    return MusicBrainzProvider(
        user_agent=settings.musicbrainz_user_agent,
    )
