from __future__ import annotations

import time
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field


@dataclass
class Timings:
    """Замеры времени этапов запроса (мс)"""

    embed_ms: float | None = None
    retrieve_ms: float | None = None
    generate_ms: float | None = None
    total_ms: float | None = None
    _start: float = field(default_factory=time.perf_counter, repr=False)

    def mark_total(self) -> None:
        self.total_ms = round((time.perf_counter() - self._start) * 1000, 2)

    def to_dict(self) -> dict[str, float | None]:
        return {
            "embed_ms": self.embed_ms,
            "retrieve_ms": self.retrieve_ms,
            "generate_ms": self.generate_ms,
            "total_ms": self.total_ms,
        }


@contextmanager
def measure() -> Generator[list[float], None, None]:
    bucket: list[float] = []
    start = time.perf_counter()
    try:
        yield bucket
    finally:
        elapsed = round((time.perf_counter() - start) * 1000, 2)
        bucket.append(elapsed)


def set_stage(timings: Timings, stage: str, value_ms: float) -> None:
    if stage == "embed":
        timings.embed_ms = value_ms
    elif stage == "retrieve":
        timings.retrieve_ms = value_ms
    elif stage == "generate":
        timings.generate_ms = value_ms
    else:
        raise ValueError(f"Unknown stage: {stage}")
