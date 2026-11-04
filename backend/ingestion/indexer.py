from __future__ import annotations

from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

from backend.config import settings
from backend.ingestion.chunker.base import DocumentChunk
from backend.ingestion.chunker.factory import get_chunker
from backend.ingestion.cleaner import CleanerConfig, clean
from backend.ingestion.embedder import BaseEmbedder, OllamaEmbedder
from backend.ingestion.parsers.factory import ParserFactory, UnknownFormatError
from backend.ingestion.parsers.markdown_parser import MarkdownParser
from backend.ingestion.parsers.pdf_parser import PdfParser
from backend.ingestion.parsers.text_parser import TextParser
from backend.observability.logging import get_logger
from backend.retrieval.bm25_index import BM25Index
from backend.stores.base import BaseVectorStore
from backend.stores.chroma_store import ChromaVectorStore
from backend.stores.metadata_store import DocumentRecord, MetadataStore

logger = get_logger(__name__)

SUPPORTED_SUFFIXES = {".md", ".markdown", ".txt", ".pdf"}


def default_parser_factory() -> ParserFactory:
    return ParserFactory(parsers=[MarkdownParser(), PdfParser(), TextParser()])


class Indexer:
    """
    Offline pipeline:
    scan -> parse -> clean -> chunk -> embed -> vector + BM25 + metadata
    """

    def __init__(
        self,
        documents_dir: str | Path | None = None,
        *,
        parser_factory: ParserFactory | None = None,
        embedder: BaseEmbedder | None = None,
        vector_store: BaseVectorStore | None = None,
        bm25_path: str | Path | None = None,
        metadata_path: str | Path | None = None,
        cleaner_config: CleanerConfig | None = None,
        batch_size: int = 32,
    ) -> None:
        self.documents_dir = Path(documents_dir or settings.paths.documents)
        self.parser_factory = parser_factory or default_parser_factory()
        self.embedder = embedder or OllamaEmbedder()
        self.vector_store = vector_store or ChromaVectorStore()
        self.bm25_path = Path(bm25_path or Path(settings.paths.index) / "bm25.pkl")
        self.metadata_store = MetadataStore(
            metadata_path or Path(settings.paths.index) / "metadata.json"
        )
        self.cleaner_config = cleaner_config or CleanerConfig(remove_headers_footers=True)
        self.batch_size = batch_size
        self.chunker = get_chunker()

        self._all_chunks: list[DocumentChunk] = []

    def run(self, *, rebuild_bm25: bool = True) -> dict[str, int]:
        files = list(self._scan())
        logger.info("index_scan", directory=str(self.documents_dir), files=len(files))

        indexed_docs = 0
        indexed_chunks = 0
        skipped = 0
        errors = 0

        new_chunks: list[DocumentChunk] = []

        for path in files:
            try:
                n_chunks, skipped_doc = self._index_file(path, new_chunks)
                if skipped_doc:
                    skipped += 1
                else:
                    indexed_docs += 1
                    indexed_chunks += n_chunks
            except UnknownFormatError:
                logger.warning("index_skip_unknown_format", path=str(path))
                skipped += 1
            except Exception as exc:  # noqa: BLE001
                errors += 1
                logger.exception("index_file_failed", path=str(path), error=str(exc))

        if new_chunks:
            self._embed_and_store(new_chunks)

        if rebuild_bm25:
            self._rebuild_bm25()

        self.metadata_store.save()

        stats = {
            "files_seen": len(files),
            "documents_indexed": indexed_docs,
            "chunks_indexed": indexed_chunks,
            "skipped": skipped,
            "errors": errors,
        }
        logger.info("index_complete", **stats)
        return stats

    def _scan(self) -> Iterable[Path]:
        if not self.documents_dir.exists():
            logger.warning("documents_dir_missing", path=str(self.documents_dir))
            return
        for path in sorted(self.documents_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
                yield path

    def _index_file(
        self,
        path: Path,
        out_chunks: list[DocumentChunk],
    ) -> tuple[int, bool]:
        parser = self.parser_factory.get_parser(path)
        logger.info("index_parse_start", path=str(path), parser=type(parser).__name__)

        parsed = parser.parse(path)

        if self.metadata_store.has_document(parsed.document_id):
            logger.info(
                "index_skip_duplicate",
                document_id=parsed.document_id,
                path=str(path),
            )
            return 0, True

        cleaned = clean(parsed.content, self.cleaner_config)
        parsed.content = cleaned
        logger.info(
            "index_cleaned",
            document_id=parsed.document_id,
            chars=len(cleaned),
        )

        chunks = self.chunker.chunk(parsed)
        logger.info(
            "index_chunked",
            document_id=parsed.document_id,
            chunks=len(chunks),
        )

        out_chunks.extend(chunks)
        self._all_chunks.extend(chunks)

        self.metadata_store.upsert(
            DocumentRecord(
                document_id=parsed.document_id,
                source=parsed.source,
                filename=parsed.filename,
                file_type=parsed.file_type,
                chunk_ids=[c.chunk_id for c in chunks],
                metadata={
                    "title": parsed.metadata.get("title"),
                    "page_count": parsed.metadata.get("page_count"),
                },
                indexed_at=datetime.now(UTC).isoformat(),
            )
        )
        return len(chunks), False

    def _embed_and_store(self, chunks: list[DocumentChunk]) -> None:
        logger.info("index_embed_start", chunks=len(chunks))
        texts = [c.content for c in chunks]
        embeddings: list[list[float]] = []

        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            vectors = self.embedder.embed_documents(batch)
            embeddings.extend(vectors)
            logger.info(
                "index_embed_batch",
                batch_start=start,
                batch_size=len(batch),
            )

        self.vector_store.add_documents(chunks, embeddings)
        logger.info("index_vector_stored", chunks=len(chunks))

    def _rebuild_bm25(self) -> None:
        """
        полная пересборка BM25 по всем известным чанкам из metadata + vector store
        упрощённо: по self._all_chunks текущего прогона + уже известные id
        """
        all_chunks = self._reparse_all_indexed()
        index = BM25Index()
        index.build(all_chunks)
        index.save(self.bm25_path)
        logger.info("index_bm25_saved", path=str(self.bm25_path), documents=len(index))

    def _reparse_all_indexed(self) -> list[DocumentChunk]:
        chunks: list[DocumentChunk] = []
        for path in self._scan():
            try:
                parser = self.parser_factory.get_parser(path)
                parsed = parser.parse(path)
                if not self.metadata_store.has_document(parsed.document_id):
                    continue
                parsed.content = clean(parsed.content, self.cleaner_config)
                chunks.extend(self.chunker.chunk(parsed))
            except Exception as exc:  # noqa: BLE001
                logger.warning("bm25_reparse_failed", path=str(path), error=str(exc))
        return chunks
