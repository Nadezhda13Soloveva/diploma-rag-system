from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from backend.api.main import app
from backend.stores.metadata_store import DocumentRecord, MetadataStore

client = TestClient(app)


def test_list_documents_pagination(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    meta_path = tmp_path / "metadata.json"
    store = MetadataStore(meta_path)
    for i in range(5):
        store.upsert(
            DocumentRecord(
                document_id=f"id-{i}",
                source=f"docs/f{i}.md",
                filename=f"f{i}.md",
                file_type="md",
                chunk_ids=[f"c{i}-0", f"c{i}-1"],
                indexed_at=f"2026-01-0{i+1}T00:00:00+00:00",
            )
        )
    store.save()

    monkeypatch.setattr(
        "backend.api.routes.documents.settings.paths.index",
        str(tmp_path),
    )

    resp = client.get("/documents?limit=2&offset=0")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 5
    assert data["limit"] == 2
    assert data["offset"] == 0
    assert len(data["items"]) == 2
    item = data["items"][0]
    assert "filename" in item
    assert "chunk_count" in item
    assert item["chunk_count"] == 2
    assert "date_indexed" in item

    resp2 = client.get("/documents?limit=2&offset=4")
    assert resp2.status_code == 200
    assert len(resp2.json()["items"]) == 1


def test_list_documents_empty(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(
        "backend.api.routes.documents.settings.paths.index",
        str(tmp_path),
    )
    resp = client.get("/documents")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 0
    assert data["items"] == []
