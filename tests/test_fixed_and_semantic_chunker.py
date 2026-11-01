from __future__ import annotations

from backend.ingestion.chunker.factory import get_chunker
from backend.ingestion.chunker.fixed import FixedChunker
from backend.ingestion.chunker.semantic import SemanticChunker
from backend.ingestion.parsers.base import ParsedDocument


def _doc(content: str) -> ParsedDocument:
    return ParsedDocument(
        document_id="d1",
        source="s.txt",
        filename="s.txt",
        file_type="txt",
        content=content,
        metadata={"title": "T"},
    )


def test_fixed_chunk_size_and_overlap() -> None:
    chunker = FixedChunker(chunk_size=20, chunk_overlap=5)
    text = "abcdefghijklmnopqrstuvwxyz"  # 26 chars
    chunks = chunker.chunk(_doc(text))

    assert len(chunks) >= 2
    assert all(len(c.content) <= 20 for c in chunks)
    assert chunks[0].metadata["chunker"] == "fixed"
    assert chunks[0].title == "T"
    assert chunks[0].total_chunks == len(chunks)


def test_fixed_empty() -> None:
    assert FixedChunker(chunk_size=10, chunk_overlap=2).chunk(_doc("")) == []


def test_semantic_with_mock_embeddings() -> None:
    sentences_doc = (
        "Cats sleep a lot. Cats like milk. Cats are soft. "
        "Quantum physics is hard. Quantum fields fluctuate. Entanglement is weird."
    )

    def fake_embed(texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for t in texts:
            if "Cat" in t or "cat" in t.lower():
                vectors.append([1.0, 0.0, 0.0])
            elif "Quantum" in t or "Entanglement" in t:
                vectors.append([0.0, 1.0, 0.0])
            else:
                vectors.append([0.0, 0.0, 1.0])
        return vectors

    chunker = SemanticChunker(
        embed_fn=fake_embed,
        similarity_threshold=0.5,
        chunk_size=500,
    )
    chunks = chunker.chunk(_doc(sentences_doc))

    assert len(chunks) >= 2
    assert chunks[0].metadata["chunker"] == "semantic"
    assert all(c.document_id == "d1" for c in chunks)


def test_semantic_fallback_without_embed_fn() -> None:
    chunker = SemanticChunker(embed_fn=None, chunk_size=40)
    text = "One sentence here. Another sentence follows. Third one appears now. Fourth ends."
    chunks = chunker.chunk(_doc(text))
    assert len(chunks) >= 1
    assert all(c.total_chunks == len(chunks) for c in chunks)


def test_factory_switch() -> None:
    assert isinstance(get_chunker("fixed"), FixedChunker)
    assert isinstance(get_chunker("semantic"), SemanticChunker)
    # recursive already exists
    from backend.ingestion.chunker.recursive import RecursiveChunker

    assert isinstance(get_chunker("recursive"), RecursiveChunker)
