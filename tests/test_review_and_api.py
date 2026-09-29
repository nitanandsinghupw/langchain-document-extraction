import pytest
from conftest import DATA
from fastapi.testclient import TestClient

from doc_extraction.api import create_app
from doc_extraction.pipeline import Status
from doc_extraction.review_store import NotReviewableError, ReviewAction, ReviewStore


@pytest.fixture
def store(tmp_path) -> ReviewStore:
    return ReviewStore(tmp_path / "test.db")


@pytest.fixture
def client(pipeline, store) -> TestClient:
    return TestClient(create_app(pipeline, store))


def upload(client, name):
    with open(DATA / "samples" / name, "rb") as f:
        return client.post("/extract", files={"file": (name, f, "text/plain")}).json()


def test_a_reviewer_can_fix_a_missing_due_date_and_approve(pipeline, store, sample):
    queued = store.save(pipeline.run(sample("inv-005.txt"), "inv-005.txt"), sample("inv-005.txt"))
    assert queued.status is Status.NEEDS_REVIEW

    done = store.review(queued.id, ReviewAction.APPROVE, "maria", corrections={"due_date": "2025-10-25"})

    assert done.status is Status.APPROVED
    assert done.data["due_date"] == "2025-10-25"
    assert done.reviewed_by == "maria"
    assert "due_date_missing" not in {i.code for i in done.issues}


def test_documents_can_only_be_reviewed_once(pipeline, store, sample):
    queued = store.save(pipeline.run(sample("inv-003.txt")), sample("inv-003.txt"))
    store.review(queued.id, ReviewAction.REJECT, "maria")

    with pytest.raises(NotReviewableError):
        store.review(queued.id, ReviewAction.APPROVE, "maria")


def test_api_upload_queue_review_and_export(client):
    assert upload(client, "inv-001.txt")["status"] == "auto_approved"
    queued = upload(client, "inv-004.txt")
    assert queued["status"] == "needs_review"

    queue = client.get("/documents", params={"status": "needs_review"}).json()
    assert [d["id"] for d in queue] == [queued["id"]]

    fixed = client.post(
        f"/documents/{queued['id']}/review",
        json={"action": "approve", "reviewer": "maria", "corrections": {"iban": "DE89370400440532013000"}},
    ).json()
    assert fixed["status"] == "approved"
    assert not fixed["review_reasons"]

    csv_text = client.get("/export/invoices.csv").text
    assert csv_text.count("\n") == 3  # header + two approved invoices
    assert "SH-2025-112" in csv_text and "BL-2025-0417" in csv_text


def test_api_rejects_bad_input(client):
    assert client.post("/extract", files={"file": ("x.xlsx", b"data", "application/octet-stream")}).status_code == 415
    assert client.get("/documents/nope").status_code == 404
    queued = upload(client, "inv-005.txt")
    bad = client.post(
        f"/documents/{queued['id']}/review",
        json={"action": "approve", "reviewer": "maria", "corrections": {"due_date": "not a date"}},
    )
    assert bad.status_code == 422


def test_the_same_invoice_uploaded_twice_is_held_for_review(client):
    first = upload(client, "inv-001.txt")
    with open(DATA / "pdf" / "inv-001.pdf", "rb") as f:
        second = client.post("/extract", files={"file": ("inv-001.pdf", f, "application/pdf")}).json()

    assert first["status"] == "auto_approved"
    assert second["status"] == "needs_review"
    assert f"Possible duplicate of document {first['id']}" in second["review_reasons"][0]
