from app.database.session import Base, engine

# Import models so their tables are registered with Base.metadata.
from app.models import (  # noqa: F401
    Album,
    Artist,
    Genre,
    MusicProviderMapping,
    SpotifyConnection,
    Track,
    UserPreference,
)


def initialize_database() -> None:
    Base.metadata.create_all(bind=engine)
