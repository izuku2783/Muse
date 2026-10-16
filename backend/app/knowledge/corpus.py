import json
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from app.knowledge.builders import KnowledgeDocumentBuilder
from app.knowledge.schemas import KnowledgeDocument
from app.models.music import Album, Artist, Genre, Track


@dataclass(frozen=True)
class CorpusExportSummary:
    document_count: int
    output_path: Path


class KnowledgeCorpusService:
    """Build and export a repeatable corpus from canonical music entities."""

    def __init__(
        self,
        session: Session,
        builder: KnowledgeDocumentBuilder | None = None,
    ):
        self.session = session
        self.builder = builder or KnowledgeDocumentBuilder()

    def build_documents(self) -> list[KnowledgeDocument]:
        documents: list[KnowledgeDocument] = []

        artists = self.session.scalars(
            select(Artist)
            .options(
                selectinload(Artist.genres),
                selectinload(Artist.tracks).selectinload(Track.genres),
            )
            .order_by(Artist.id)
        ).all()
        documents.extend(self.builder.build_artist(artist) for artist in artists)

        albums = self.session.scalars(
            select(Album)
            .options(
                selectinload(Album.artist),
                selectinload(Album.tracks).selectinload(Track.genres),
            )
            .order_by(Album.id)
        ).all()
        documents.extend(self.builder.build_album(album) for album in albums)

        tracks = self.session.scalars(
            select(Track)
            .options(
                selectinload(Track.artist),
                selectinload(Track.album),
                selectinload(Track.genres),
            )
            .order_by(Track.id)
        ).all()
        documents.extend(self.builder.build_track(track) for track in tracks)

        genres = self.session.scalars(
            select(Genre)
            .options(
                selectinload(Genre.artists),
                selectinload(Genre.tracks).selectinload(Track.artist),
            )
            .order_by(Genre.id)
        ).all()
        documents.extend(self.builder.build_genre(genre) for genre in genres)

        return documents

    def export_jsonl(self, output_path: str | Path) -> CorpusExportSummary:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        documents = self.build_documents()
        lines = [
            json.dumps(
                document.model_dump(mode="json", exclude_none=True),
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            for document in documents
        ]
        content = "\n".join(lines)
        if lines:
            content += "\n"

        temporary = destination.with_name(f".{destination.name}.tmp")
        try:
            temporary.write_text(content, encoding="utf-8", newline="\n")
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)

        return CorpusExportSummary(
            document_count=len(documents),
            output_path=destination,
        )
