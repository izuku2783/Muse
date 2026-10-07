from app.database.session import Base, engine

# Import models so their tables are registered with Base.metadata.
from app.models import Album, Artist, Genre, Track  # noqa: F401


def initialize_database() -> None:
    Base.metadata.create_all(bind=engine)