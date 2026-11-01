from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

_ALLOWED_EXTRA = r"""\s\-–—.,;:!?()\[\]{}<>"'«»“”‘’/`~@#$%^&*+=_|\\"""


@dataclass
class CleanerConfig:
    normalize_unicode: bool = True
    strip_trailing_whitespace: bool = True
    collapse_blank_lines: bool = True
    remove_headers_footers: bool = False
    remove_junk_chars: bool = True
    # эвристика колонтитулов
    header_footer_min_repeats: int = 3
    max_header_footer_lines: int = 2


def clean(text: str, config: CleanerConfig | None = None) -> str:
    cfg = config or CleanerConfig()
    if not text:
        return ""

    result = text

    if cfg.normalize_unicode:
        result = _normalize_unicode(result)

    if cfg.strip_trailing_whitespace:
        result = _strip_trailing_whitespace(result)

    if cfg.remove_headers_footers:
        result = _remove_headers_footers(
            result,
            min_repeats=cfg.header_footer_min_repeats,
            max_lines=cfg.max_header_footer_lines,
        )

    if cfg.remove_junk_chars:
        result = _remove_junk_chars(result)

    if cfg.collapse_blank_lines:
        result = _collapse_blank_lines(result)

    if cfg.strip_trailing_whitespace:
        result = _strip_trailing_whitespace(result)

    return result.strip() + ("\n" if result.strip() else "")


def _normalize_unicode(text: str) -> str:
    return unicodedata.normalize("NFKC", text)


def _strip_trailing_whitespace(text: str) -> str:
    lines = [line.rstrip() for line in text.splitlines()]
    return "\n".join(lines)


def _collapse_blank_lines(text: str) -> str:
    # 3+ пустых строк подряд = одна пустая строка (два \n)
    return re.sub(r"\n{3,}", "\n\n", text)


def _remove_junk_chars(text: str) -> str:
    # \w в UNICODE = буквы + цифры + _; плюс явный whitelist
    pattern = rf"[^\w{_ALLOWED_EXTRA}]"
    cleaned = re.sub(pattern, "", text, flags=re.UNICODE)
    # убираем одиночные управляющие, если остались
    cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", cleaned)
    return cleaned


def _remove_headers_footers(
    text: str,
    min_repeats: int = 3,
    max_lines: int = 2,
) -> str:
    """
    Эвристика: строки, которые часто встречаются в начале/конце
    «траничных блоков, считаем колонтитулами

    Ожидаем разделители вида '--- Page N ---' (как в PdfParser)
    Если разделителей нет - пытаемся по частым коротким строкам
    """
    page_split = re.split(r"(?m)^--- Page \d+ ---\s*$", text)
    pages = [p for p in page_split if p.strip()]

    if len(pages) >= min_repeats:
        return _strip_repeated_edge_lines(pages, min_repeats, max_lines)

    # fallback без page-маркеров: убираем очень частые короткие строки
    return _strip_frequent_short_lines(text, min_repeats=max(min_repeats, 5))


def _strip_repeated_edge_lines(
    pages: list[str],
    min_repeats: int,
    max_lines: int,
) -> str:
    from collections import Counter

    head_counter: Counter[str] = Counter()
    tail_counter: Counter[str] = Counter()

    parsed_pages: list[list[str]] = []
    for page in pages:
        lines = [ln.rstrip() for ln in page.splitlines()]
        # не учитываем ведущие/хвостовые пустые
        while lines and not lines[0].strip():
            lines.pop(0)
        while lines and not lines[-1].strip():
            lines.pop()
        parsed_pages.append(lines)
        for ln in lines[:max_lines]:
            if ln.strip():
                head_counter[ln.strip()] += 1
        for ln in lines[-max_lines:]:
            if ln.strip():
                tail_counter[ln.strip()] += 1

    headers = {s for s, c in head_counter.items() if c >= min_repeats}
    footers = {s for s, c in tail_counter.items() if c >= min_repeats}

    cleaned_pages: list[str] = []
    for lines in parsed_pages:
        filtered = [ln for ln in lines if ln.strip() not in headers and ln.strip() not in footers]
        cleaned_pages.append("\n".join(filtered).strip())

    return "\n\n".join(p for p in cleaned_pages if p)


def _strip_frequent_short_lines(text: str, min_repeats: int = 5) -> str:
    from collections import Counter

    lines = text.splitlines()
    counter = Counter(ln.strip() for ln in lines if ln.strip() and len(ln.strip()) <= 80)
    frequent = {s for s, c in counter.items() if c >= min_repeats}
    if not frequent:
        return text
    return "\n".join(ln for ln in lines if ln.strip() not in frequent)
