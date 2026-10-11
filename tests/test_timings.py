from __future__ import annotations

import time

from backend.observability.timing import Timings, measure, set_stage


def test_measure_context_manager() -> None:
    with measure() as bucket:
        time.sleep(0.01)

    assert len(bucket) == 1
    assert bucket[0] >= 10  # min ~10 мс


def test_timings_stages() -> None:
    timings = Timings()

    with measure() as bucket:
        time.sleep(0.005)
    set_stage(timings, "embed", bucket[0])

    with measure() as bucket:
        time.sleep(0.005)
    set_stage(timings, "retrieve", bucket[0])

    with measure() as bucket:
        time.sleep(0.005)
    set_stage(timings, "generate", bucket[0])

    timings.mark_total()

    data = timings.to_dict()
    assert data["embed_ms"] is not None and data["embed_ms"] >= 5
    assert data["retrieve_ms"] is not None and data["retrieve_ms"] >= 5
    assert data["generate_ms"] is not None and data["generate_ms"] >= 5
    assert data["total_ms"] is not None and data["total_ms"] >= 15


def test_timings_to_dict_keys() -> None:
    timings = Timings()
    timings.mark_total()
    keys = set(timings.to_dict().keys())
    assert keys == {"embed_ms", "retrieve_ms", "generate_ms", "total_ms"}
