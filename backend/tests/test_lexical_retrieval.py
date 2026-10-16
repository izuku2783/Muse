import pytest

from app.knowledge.schemas import KnowledgeDocument, DocumentMetadata, DocumentType
from app.retrieval.lexical import BM25LexicalRetriever, tokenize
from app.vector_store.schemas import VectorSearchFilter


def document(
    entity_id: int,
    content: str,
    *,
    document_type: DocumentType = DocumentType.TRACK,
    genres: tuple[str, ...] = (),
    release_year: int | None = None,
    provider: str | None = None,
    artist_id: int | None = None,
    artist_name: str | None = None,
    duration_ms: int | None = None,
    popularity: float | None = None,
    instrumental: bool | None = None,
    country: str | None = None,
    album_type: str | None = None,
) -> KnowledgeDocument:
    entity_field = f"{document_type.value}_id"
    return KnowledgeDocument(
        document_id=f"{document_type.value}:{entity_id}",
        document_type=document_type,
        content=content,
        metadata=DocumentMetadata(
            document_type=document_type,
            entity_id=entity_id,
            artist_id=artist_id,
            artist_name=artist_name,
            genres=genres,
            release_year=release_year,
            provider=provider,
            duration_ms=duration_ms,
            popularity=popularity,
            instrumental=instrumental,
            country=country,
            album_type=album_type,
            **{entity_field: entity_id},
        ),
    )


def test_tokenize_normalizes_case_whitespace_and_unicode():
    assert tokenize("  DARK synth-wave Café  ") == ["dark", "synth", "wave", "café"]


def test_bm25_ranks_more_relevant_exact_term_match_first():
    retriever = BM25LexicalRetriever(
        [
            document(1, "dark ambient textures"),
            document(2, "dark dark ambient"),
            document(3, "bright acoustic pop"),
        ]
    )

    response = retriever.search("dark ambient")

    assert [result.document_id for result in response.results] == ["track:2", "track:1"]
    assert all(result.score > 0 for result in response.results)
    assert [result.rank for result in response.results] == [1, 2]


def test_bm25_returns_no_results_when_terms_do_not_match():
    response = BM25LexicalRetriever([document(1, "ambient music")]).search("jazz")
    assert response.results == ()


def test_bm25_breaks_equal_score_ties_by_document_id():
    response = BM25LexicalRetriever(
        [document(2, "ambient"), document(1, "ambient")]
    ).search("ambient")
    assert [result.document_id for result in response.results] == ["track:1", "track:2"]


def test_bm25_applies_current_structured_filters_before_ranking():
    retriever = BM25LexicalRetriever(
        [
            document(
                1,
                "ambient music",
                genres=("Ambient",),
                release_year=1998,
                provider="local",
            ),
            document(
                2,
                "ambient music",
                genres=("ambient",),
                release_year=2012,
                provider="musicbrainz",
            ),
            document(
                3,
                "ambient album",
                document_type=DocumentType.ALBUM,
                release_year=2010,
                provider="musicbrainz",
            ),
        ]
    )
    filters = VectorSearchFilter(
        document_types=(DocumentType.TRACK,),
        genres=("AMBIENT",),
        release_year_min=2000,
        release_year_max=2020,
        provider="musicbrainz",
    )

    response = retriever.search("ambient", filters=filters)

    assert [result.document_id for result in response.results] == ["track:2"]


def test_bm25_applies_expanded_stage_7_filters_and_artist_exclusions():
    retriever = BM25LexicalRetriever(
        [
            document(
                1,
                "instrumental synthwave",
                artist_id=10,
                artist_name="Neon Waves",
                duration_ms=240_000,
                popularity=80,
                instrumental=True,
                country="US",
                album_type="Album",
            ),
            document(
                2,
                "instrumental synthwave",
                artist_id=20,
                artist_name="Other Artist",
                duration_ms=360_000,
                popularity=40,
                instrumental=False,
                country="GB",
                album_type="Single",
            ),
        ]
    )
    filters = VectorSearchFilter(
        duration_ms_max=300_000,
        popularity_min=70,
        instrumental=True,
        countries=("us",),
        album_types=("album",),
    )
    assert [
        result.document_id
        for result in retriever.search("synthwave", filters=filters).results
    ] == ["track:1"]
    assert retriever.search(
        "synthwave",
        filters=VectorSearchFilter(exclude_artist_ids=(10,)),
    ).results[0].document_id == "track:2"
    assert retriever.search(
        "synthwave",
        filters=VectorSearchFilter(exclude_artists=("NEON WAVES",)),
    ).results[0].document_id == "track:2"


@pytest.mark.parametrize("query", ["", " ", "\n\t"])
def test_bm25_rejects_empty_query(query):
    with pytest.raises(ValueError, match="must not be empty"):
        BM25LexicalRetriever([]).search(query)


@pytest.mark.parametrize("limit", [0, 101])
def test_bm25_validates_limit(limit):
    with pytest.raises(ValueError, match="between 1 and 100"):
        BM25LexicalRetriever([]).search("ambient", limit=limit)


@pytest.mark.parametrize(
    ("k1", "b"),
    [(0, 0.75), (-1, 0.75), (1.5, -0.1), (1.5, 1.1)],
)
def test_bm25_validates_configuration(k1, b):
    with pytest.raises(ValueError, match="BM25 requires"):
        BM25LexicalRetriever([], k1=k1, b=b)


def test_bm25_rejects_duplicate_document_ids():
    with pytest.raises(ValueError, match="unique document IDs"):
        BM25LexicalRetriever([document(1, "first"), document(1, "second")])
