from sqlalchemy.orm import Session

from app.memory.service import UserMemoryService
from app.repositories.music import AlbumRepository, ArtistRepository, TrackRepository
from app.retrieval.semantic import SemanticRetriever
from app.retrieval.reranked import RerankedRetriever
from app.services.music.provider import MusicProvider
from app.tools.music import (
    GetAlbumTool,
    GetArtistTool,
    GetTrackTool,
    SearchExternalMusicProviderTool,
    SearchMusicTool,
)
from app.tools.memory import GetUserPreferencesTool
from app.tools.registry import ToolRegistry
from app.tools.retrieval import RetrieveMusicEvidenceTool, SemanticSearchTool


def build_tool_registry(
    session: Session,
    semantic_retriever: SemanticRetriever,
    external_provider: MusicProvider | None = None,
    reranked_retriever: RerankedRetriever | None = None,
) -> ToolRegistry:
    tracks = TrackRepository(session)
    tools = [
        SearchMusicTool(tracks),
        GetTrackTool(tracks),
        GetArtistTool(ArtistRepository(session)),
        GetAlbumTool(AlbumRepository(session)),
        GetUserPreferencesTool(UserMemoryService(session)),
        SemanticSearchTool(semantic_retriever),
    ]
    if external_provider is not None:
        tools.append(SearchExternalMusicProviderTool(external_provider))
    if reranked_retriever is not None:
        tools.append(RetrieveMusicEvidenceTool(reranked_retriever))
    return ToolRegistry(tools)

