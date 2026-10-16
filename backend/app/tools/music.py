from app.repositories.music import AlbumRepository, ArtistRepository, TrackRepository
from app.services.music.errors import MusicProviderError
from app.services.music.provider import MusicProvider
from app.tools.base import BaseTool
from app.tools.errors import ToolEntityNotFoundError, ToolExecutionError
from app.tools.schemas import (
    AlbumRecord,
    ArtistRecord,
    EntityIdInput,
    GetAlbumOutput,
    GetArtistOutput,
    GetTrackOutput,
    SearchExternalMusicProviderInput,
    SearchExternalMusicProviderOutput,
    SearchMusicInput,
    SearchMusicOutput,
    TrackRecord,
)


class SearchMusicTool(BaseTool[SearchMusicInput, SearchMusicOutput]):
    name = "search_music"
    description = "Search only the local music catalog for matching tracks."
    input_type = SearchMusicInput
    output_type = SearchMusicOutput

    def __init__(self, repository: TrackRepository):
        self.repository = repository

    def run(self, tool_input: SearchMusicInput) -> SearchMusicOutput:
        tracks = self.repository.search_models(
            tool_input.query,
            limit=tool_input.limit,
            offset=tool_input.offset,
        )
        return SearchMusicOutput(
            query=tool_input.query,
            results=tuple(
                TrackRecord(id=track.id, data=self.repository.to_data(track))
                for track in tracks
            ),
        )


class GetTrackTool(BaseTool[EntityIdInput, GetTrackOutput]):
    name = "get_track"
    description = "Get one local track by its canonical database ID."
    input_type = EntityIdInput
    output_type = GetTrackOutput

    def __init__(self, repository: TrackRepository):
        self.repository = repository

    def run(self, tool_input: EntityIdInput) -> GetTrackOutput:
        track = self.repository.get_by_id(tool_input.entity_id)
        if track is None:
            raise ToolEntityNotFoundError(
                f"Track {tool_input.entity_id} was not found."
            )
        return GetTrackOutput(
            track=TrackRecord(id=track.id, data=self.repository.to_data(track))
        )


class GetArtistTool(BaseTool[EntityIdInput, GetArtistOutput]):
    name = "get_artist"
    description = "Get one local artist by its canonical database ID."
    input_type = EntityIdInput
    output_type = GetArtistOutput

    def __init__(self, repository: ArtistRepository):
        self.repository = repository

    def run(self, tool_input: EntityIdInput) -> GetArtistOutput:
        artist = self.repository.get_by_id(tool_input.entity_id)
        if artist is None:
            raise ToolEntityNotFoundError(
                f"Artist {tool_input.entity_id} was not found."
            )
        return GetArtistOutput(
            artist=ArtistRecord(id=artist.id, data=self.repository.to_data(artist))
        )


class GetAlbumTool(BaseTool[EntityIdInput, GetAlbumOutput]):
    name = "get_album"
    description = "Get one local album by its canonical database ID."
    input_type = EntityIdInput
    output_type = GetAlbumOutput

    def __init__(self, repository: AlbumRepository):
        self.repository = repository

    def run(self, tool_input: EntityIdInput) -> GetAlbumOutput:
        album = self.repository.get_by_id(tool_input.entity_id)
        if album is None:
            raise ToolEntityNotFoundError(
                f"Album {tool_input.entity_id} was not found."
            )
        return GetAlbumOutput(
            album=AlbumRecord(id=album.id, data=self.repository.to_data(album))
        )


class SearchExternalMusicProviderTool(
    BaseTool[SearchExternalMusicProviderInput, SearchExternalMusicProviderOutput]
):
    name = "search_external_music_provider"
    description = "Search the configured external music provider without persisting results."
    input_type = SearchExternalMusicProviderInput
    output_type = SearchExternalMusicProviderOutput

    def __init__(self, provider: MusicProvider):
        self.provider = provider

    def run(
        self, tool_input: SearchExternalMusicProviderInput
    ) -> SearchExternalMusicProviderOutput:
        try:
            tracks = self.provider.search_tracks(
                tool_input.query,
                limit=tool_input.limit,
                offset=tool_input.offset,
            )
        except MusicProviderError as exc:
            raise ToolExecutionError(
                "The external music provider could not complete the search."
            ) from exc
        return SearchExternalMusicProviderOutput(
            query=tool_input.query,
            results=tuple(tracks),
        )

