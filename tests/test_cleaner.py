from __future__ import annotations

from backend.ingestion.cleaner import CleanerConfig, clean


def test_collapse_blank_lines() -> None:
    text = "line1\n\n\n\nline2\n\n\nline3"
    result = clean(text, CleanerConfig(remove_junk_chars=False))
    assert "\n\n\n" not in result
    assert "line1" in result and "line2" in result


def test_strip_trailing_whitespace() -> None:
    text = "hello   \nworld\t\t\n"
    result = clean(text, CleanerConfig(remove_junk_chars=False))
    for line in result.splitlines():
        assert line == line.rstrip()


def test_normalize_unicode_nfkc() -> None:
    text = "ＡＢＣ１２３"
    result = clean(text, CleanerConfig(remove_junk_chars=False))
    assert "ABC123" in result


def test_remove_junk_chars() -> None:
    text = "Hello\x00World\x07!"
    result = clean(text)
    assert "\x00" not in result
    assert "\x07" not in result
    assert "Hello" in result
    assert "World" in result


def test_remove_headers_footers_with_page_markers() -> None:
    pages = []
    for i in range(1, 5):
        pages.append(
            f"--- Page {i} ---\n"
            f"Confidential Company Header\n"
            f"Unique content on page {i}\n"
            f"Page footer {i % 1 + 1}\n"
            f"Confidential Footer\n"
        )
    text = "\n".join(
        f"--- Page {i} ---\n"
        f"Confidential Company Header\n"
        f"Unique content on page {i}\n"
        f"Confidential Footer\n"
        for i in range(1, 5)
    )
    result = clean(
        text,
        CleanerConfig(
            remove_headers_footers=True,
            remove_junk_chars=False,
            header_footer_min_repeats=3,
        ),
    )
    assert "Confidential Company Header" not in result
    assert "Confidential Footer" not in result
    assert "Unique content on page 1" in result
    assert "Unique content on page 4" in result


def test_config_can_disable_steps() -> None:
    text = "Ａ\n\n\n\nＢ"
    # unicode off, blank collapse off
    result = clean(
        text,
        CleanerConfig(
            normalize_unicode=False,
            collapse_blank_lines=False,
            remove_junk_chars=False,
            strip_trailing_whitespace=True,
        ),
    )
    assert "Ａ" in result  # not normalized
    assert "\n\n\n" in result  # blank lines kept


def test_empty_input() -> None:
    assert clean("") == ""
    assert clean("   \n\n  ") == ""
