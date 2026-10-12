from __future__ import annotations

from pathlib import Path

from backend.ingestion.parsers.base import BaseParser
from backend.observability.logging import get_logger

logger = get_logger(__name__)


class UnknownFormatError(ValueError):
    """неподдерживаемый формат файла"""


class ParserFactory:
    """выбирает реализацию по расширению файла"""

    def __init__(self, parsers: list[BaseParser] | None = None) -> None:
        # пока конкретных парсеров может не быть - фабрика всё равно работает
        # реализации MarkdownParser, PdfParser, TextParser добавятся отдельно
        self._parsers: list[BaseParser] = parsers or []

    def register(self, parser: BaseParser) -> None:
        self._parsers.append(parser)

    def get_parser(self, path: Path | str) -> BaseParser:
        path = Path(path)

        for parser in self._parsers:
            if parser.can_parse(path):
                logger.info(
                    "parser_selected",
                    parser=parser.__class__.__name__,
                    path=str(path),
                    suffix=path.suffix.lower(),
                )
                return parser

        logger.warning(
            "parser_not_found",
            path=str(path),
            suffix=path.suffix.lower(),
        )
        raise UnknownFormatError(
            f"Неизвестный или неподдерживаемый формат файла: {path.suffix or path.name}"
        )

    def supported_extensions(self) -> list[str]:
        """список расширений, которые сейчас умеем парсить (для диагностики)"""
        # пока парсеры не зарегистрированы - пустой список
        # после добавления конкретных парсеров можно расширить
        return []
