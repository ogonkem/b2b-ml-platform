"""
tests/unit/test_rag_ingest.py
Tests for rag_service/ingest_task.py.

Two layers, matching this project's convention of testing pure logic
directly and mocking every external service:
  - Chunking/token-bound logic is run for real against the three fixture
    policy docs (tests/fixtures/policy_docs/) — no mocking needed, since
    none of it touches MinIO, Postgres, or the network.
  - The full ingest_document task is run with MinIO, Postgres (via
    rag_service.db.get_cursor), and the OpenAI embedding call all mocked —
    same style as tests/unit/test_celery_tasks.py's process_batch tests.
"""
import base64
import re
import sys
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pytest

from rag_service.ingest_task import (
    Chunk,
    _enforce_token_bounds,
    _extract_ext,
    _extract_text,
    _token_count,
    structure_aware_chunks,
)

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures" / "policy_docs"


def _load(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


def _final_chunks(text: str):
    return _enforce_token_bounds(structure_aware_chunks(text))


# ── Doc 1: commercial bank lending policy — clean numbered-clause section_ref ──

class TestDoc1CommercialBankPolicy:

    def test_produces_multiple_structural_chunks(self):
        chunks = _final_chunks(_load("1_commercial_bank_lending_policy.md"))
        assert len(chunks) >= 8

    def test_section_refs_are_clean_numeric_clause_identifiers(self):
        """Every chunk that has a section_ref at all must be a bare
        hierarchical clause number like "1.1" or "4.2" — not a header
        sentence, not the "§" prefix, nothing else riding along."""
        chunks = _final_chunks(_load("1_commercial_bank_lending_policy.md"))
        refs = [c.section_ref for c in chunks if c.section_ref is not None]
        assert len(refs) >= 8
        for ref in refs:
            assert re.fullmatch(r"\d{1,3}(\.\d{1,3}){1,4}", ref), ref

    def test_first_chunk_is_unsectioned_preamble(self):
        chunks = _final_chunks(_load("1_commercial_bank_lending_policy.md"))
        assert chunks[0].section_ref is None


# ── Doc 2: microfinance/SACCO policy — coarse header-level section_ref ─────────

class TestDoc2MicrofinancePolicy:

    def test_produces_a_handful_of_coarse_sections(self):
        """Markdown "## " headers only, no numbered subclauses — far fewer,
        broader sections than doc 1's fine-grained numbering."""
        chunks = _final_chunks(_load("2_microfinance_sacco_policy.md"))
        assert 3 <= len(chunks) <= 8

    def test_section_refs_are_header_text_not_clause_numbers(self):
        chunks = _final_chunks(_load("2_microfinance_sacco_policy.md"))
        refs = [c.section_ref for c in chunks if c.section_ref is not None]
        assert len(refs) == len(chunks)   # every chunk in this doc is sectioned
        for ref in refs:
            # Not a bare numeric clause identifier like doc 1's...
            assert not re.fullmatch(r"\d{1,3}(\.\d{1,3}){1,4}", ref)
            # ...it's a multi-word header title instead.
            assert len(ref.split()) >= 2

    def test_known_section_titles_present(self):
        chunks = _final_chunks(_load("2_microfinance_sacco_policy.md"))
        refs = {c.section_ref for c in chunks}
        assert "Loan Products and Eligibility" in refs
        assert "Governance and Board Oversight" in refs


# ── Doc 3: informal digital lender policy — fallback, no section_ref at all ────

class TestDoc3InformalDigitalLenderPolicy:
    """Zero headers, zero numbered clauses, just prose paragraphs — this is
    exactly the case where structure-aware chunking should give up and fall
    back to plain paragraph splitting. All section_ref values being None
    here is expected and correct, not a bug."""

    def test_falls_back_to_chunk_index_only(self):
        chunks = _final_chunks(_load("3_informal_digital_lender_policy.md"))
        assert len(chunks) >= 3
        assert all(c.section_ref is None for c in chunks)

    def test_chunk_index_is_sequential_after_assignment(self):
        chunks = _final_chunks(_load("3_informal_digital_lender_policy.md"))
        for i, c in enumerate(chunks):
            c.chunk_index = i
        assert [c.chunk_index for c in chunks] == list(range(len(chunks)))

    def test_would_not_have_fallen_back_if_headers_existed(self):
        """Sanity check on the fallback condition itself: the same text with
        3+ markdown headers injected should NOT fall back."""
        text = _load("3_informal_digital_lender_policy.md")
        paragraphs = text.strip().split("\n\n")
        headered = "\n\n".join(f"## Section {i}\n{p}" for i, p in enumerate(paragraphs))
        chunks = structure_aware_chunks(headered)
        assert any(c.section_ref is not None for c in chunks)


# ── Token-size enforcement (400 max / 50 min) ──────────────────────────────────

class TestTokenBounds:

    def test_all_final_fixture_chunks_are_within_band(self):
        for name in [
            "1_commercial_bank_lending_policy.md",
            "2_microfinance_sacco_policy.md",
            "3_informal_digital_lender_policy.md",
        ]:
            for c in _final_chunks(_load(name)):
                assert 50 <= _token_count(c.text) <= 400

    def test_oversized_chunk_gets_split(self):
        long_text = " ".join(
            f"This is sentence number {i} about lending policy risk controls."
            for i in range(60)
        )
        assert _token_count(long_text) > 400
        out = _enforce_token_bounds([Chunk(text=long_text, section_ref="X")])
        assert len(out) > 1
        for c in out:
            assert _token_count(c.text) <= 400
            assert c.section_ref == "X"   # split pieces keep the parent's section_ref

    def test_undersized_chunks_get_merged_into_a_neighbor(self):
        tiny_a = Chunk(text="Tiny one.", section_ref="A")
        tiny_b = Chunk(text="Tiny two.", section_ref="B")
        normal = Chunk(text=" ".join(["word"] * 80), section_ref="C")
        out = _enforce_token_bounds([tiny_a, tiny_b, normal])
        assert len(out) == 1
        assert _token_count(out[0].text) >= 50
        # Forward-merged into the chunk that ends up holding the larger
        # share of the merged text.
        assert out[0].section_ref == "C"

    def test_single_undersized_chunk_is_left_alone(self):
        """Nothing to merge into — nothing should be dropped."""
        out = _enforce_token_bounds([Chunk(text="Just one short chunk.", section_ref="A")])
        assert len(out) == 1
        assert out[0].text == "Just one short chunk."


# ── Text extraction ───────────────────────────────────────────────────────────

class TestExtractExt:

    def test_accepts_supported_extensions(self):
        for name, ext in [("a.pdf", "pdf"), ("a.docx", "docx"), ("a.md", "md"), ("a.TXT", "txt")]:
            assert _extract_ext(name) == ext

    def test_rejects_unsupported_extension(self):
        with pytest.raises(ValueError):
            _extract_ext("a.csv")


class TestExtractText:

    def test_markdown_and_text_are_passthrough(self):
        raw = "# Title\n\nSome body text.".encode("utf-8")
        text, page_count = _extract_text(raw, "md")
        assert text == "# Title\n\nSome body text."
        assert page_count is None

    def test_pdf_extraction_joins_pages_and_reports_page_count(self):
        page1, page2 = MagicMock(), MagicMock()
        page1.extract_text.return_value = "Page one text"
        page2.extract_text.return_value = "Page two text"
        fake_pdf = MagicMock()
        fake_pdf.pages = [page1, page2]
        fake_pdf.__enter__.return_value = fake_pdf
        fake_pdf.__exit__.return_value = False

        with patch("pdfplumber.open", return_value=fake_pdf):
            text, page_count = _extract_text(b"%PDF-fake%", "pdf")

        assert "Page one text" in text and "Page two text" in text
        assert page_count == 2

    def test_docx_extraction_joins_paragraphs(self):
        p1, p2 = MagicMock(text="First paragraph."), MagicMock(text="Second paragraph.")
        fake_doc = MagicMock()
        fake_doc.paragraphs = [p1, p2]

        with patch("docx.Document", return_value=fake_doc):
            text, page_count = _extract_text(b"fake-docx-bytes", "docx")

        assert text == "First paragraph.\nSecond paragraph."
        assert page_count is None


# ── Full task orchestration (MinIO / Postgres / embedding all mocked) ─────────

class FakeCursor:
    """In-memory stand-in for rag.documents / rag.doc_chunks — enough to
    exercise versioning, superseding, and the previous-chunk diff without a
    real Postgres connection."""

    def __init__(self):
        self.documents = {}
        self.chunks = []
        self._result = []

    def execute(self, query, params=None):
        q = " ".join(query.split()).upper()
        params = params or ()

        if q.startswith("SELECT CURRENT_VERSION FROM RAG.DOCUMENTS"):
            doc_id = params[0]
            self._result = [{"current_version": self.documents[doc_id]["current_version"]}] \
                if doc_id in self.documents else []
        elif q.startswith("INSERT INTO RAG.DOCUMENTS"):
            doc_id, tenant_id, filename, version = params
            self.documents[doc_id] = {"tenant_id": tenant_id, "filename": filename, "current_version": version}
        elif q.startswith("UPDATE RAG.DOCUMENTS SET CURRENT_VERSION"):
            version, doc_id = params
            self.documents[doc_id]["current_version"] = version
        elif q.startswith("SELECT CHUNK_HASH, EMBEDDING FROM RAG.DOC_CHUNKS"):
            doc_id = params[0]
            self._result = [
                {"chunk_hash": c["chunk_hash"], "embedding": c["embedding"]}
                for c in self.chunks if c["doc_id"] == doc_id and c["effective_to"] is None
            ]
        elif q.startswith("UPDATE RAG.DOC_CHUNKS SET EFFECTIVE_TO"):
            doc_id = params[0]
            for c in self.chunks:
                if c["doc_id"] == doc_id and c["effective_to"] is None:
                    c["effective_to"] = "now"
        elif q.startswith("INSERT INTO RAG.DOC_CHUNKS"):
            keys = ["tenant_id", "doc_id", "doc_version", "section_ref", "chunk_index",
                    "chunk_text", "chunk_hash", "embedding", "minio_object_path", "ingested_by"]
            row = dict(zip(keys, params))
            row["effective_to"] = None
            self.chunks.append(row)
        else:
            raise AssertionError(f"Unexpected query in FakeCursor: {query!r}")

    def fetchone(self):
        return self._result[0] if self._result else None

    def fetchall(self):
        return self._result


def _fake_get_cursor(cursor):
    @contextmanager
    def _get_cursor(commit=False):
        yield cursor
    return _get_cursor


def _b64(text: str) -> str:
    return base64.b64encode(text.encode("utf-8")).decode("ascii")


def _run_ingest(cursor, minio, filename, text, tenant_id="tenant-a", doc_id="doc-1",
                 embed_calls=None, fake_redis=None):
    def fake_embed(texts):
        if embed_calls is not None:
            embed_calls.append(list(texts))
        return [[0.1] * 1536 for _ in texts]

    with patch("rag_service.ingest_task._minio", return_value=minio), \
         patch("rag_service.ingest_task.get_cursor", _fake_get_cursor(cursor)), \
         patch("rag_service.ingest_task._embed_batch", side_effect=fake_embed), \
         patch("rag_service.ingest_task._redis", return_value=fake_redis or MagicMock()):
        from rag_service.ingest_task import ingest_document
        return ingest_document.run(tenant_id, doc_id, filename, _b64(text))


@pytest.fixture
def minio():
    m = MagicMock()
    m.bucket_exists.return_value = True
    return m


@pytest.fixture
def doc2_text():
    return _load("2_microfinance_sacco_policy.md")


class TestIngestOrchestration:

    def test_first_ingest_creates_version_v1(self, minio, doc2_text):
        cur = FakeCursor()
        result = _run_ingest(cur, minio, "sacco.md", doc2_text)
        assert result["version"] == "v1"
        assert cur.documents["doc-1"]["current_version"] == "v1"

    def test_uploads_original_to_correct_minio_path(self, minio, doc2_text):
        cur = FakeCursor()
        _run_ingest(cur, minio, "sacco.md", doc2_text, tenant_id="acme", doc_id="doc-42")
        bucket, object_path = minio.put_object.call_args[0][0], minio.put_object.call_args[0][1]
        assert bucket == "doc-chunks-raw"
        assert object_path == "acme/doc-42/v1/original.md"

    def test_creates_bucket_if_missing(self, doc2_text):
        cur = FakeCursor()
        minio = MagicMock()
        minio.bucket_exists.return_value = False
        _run_ingest(cur, minio, "sacco.md", doc2_text)
        minio.make_bucket.assert_called_once_with("doc-chunks-raw")

    def test_first_ingest_embeds_every_chunk(self, minio, doc2_text):
        cur = FakeCursor()
        calls = []
        result = _run_ingest(cur, minio, "sacco.md", doc2_text, embed_calls=calls)
        assert result["chunks_embedded"] == result["chunk_count"]
        assert result["chunks_reused"] == 0
        assert sum(len(c) for c in calls) == result["chunk_count"]

    def test_reingesting_identical_content_bumps_version_but_reuses_all_embeddings(self, minio, doc2_text):
        cur = FakeCursor()
        _run_ingest(cur, minio, "sacco.md", doc2_text)
        calls = []
        result = _run_ingest(cur, minio, "sacco.md", doc2_text, embed_calls=calls)
        assert result["version"] == "v2"
        assert result["chunks_embedded"] == 0
        assert result["chunks_reused"] == result["chunk_count"]
        assert sum(len(c) for c in calls) == 0   # _embed_batch never given any texts to embed

    def test_reingest_supersedes_previous_version_rows(self, minio, doc2_text):
        cur = FakeCursor()
        _run_ingest(cur, minio, "sacco.md", doc2_text)
        v1_rows = [c for c in cur.chunks if c["doc_version"] == "v1"]
        assert v1_rows and all(c["effective_to"] is None for c in v1_rows)

        _run_ingest(cur, minio, "sacco.md", doc2_text)
        v1_rows = [c for c in cur.chunks if c["doc_version"] == "v1"]
        v2_rows = [c for c in cur.chunks if c["doc_version"] == "v2"]
        assert all(c["effective_to"] == "now" for c in v1_rows)
        assert all(c["effective_to"] is None for c in v2_rows)

    def test_changed_content_only_reembeds_the_changed_chunk(self, minio, doc2_text):
        cur = FakeCursor()
        _run_ingest(cur, minio, "sacco.md", doc2_text)

        edited = doc2_text.replace(
            "The SACCO board reviews the aggregate loan portfolio",
            "The SACCO board reviews the entire loan portfolio",
        )
        assert edited != doc2_text

        calls = []
        result = _run_ingest(cur, minio, "sacco.md", edited, embed_calls=calls)
        assert result["chunks_embedded"] == 1
        assert result["chunks_reused"] == result["chunk_count"] - 1
        assert sum(len(c) for c in calls) == 1

    def test_different_doc_ids_version_independently(self, minio, doc2_text):
        cur = FakeCursor()
        r1 = _run_ingest(cur, minio, "sacco.md", doc2_text, doc_id="doc-a")
        r2 = _run_ingest(cur, minio, "sacco.md", doc2_text, doc_id="doc-b")
        assert r1["version"] == "v1"
        assert r2["version"] == "v1"

    def test_rejects_unsupported_file_type(self, minio, doc2_text):
        with pytest.raises(ValueError):
            _run_ingest(FakeCursor(), minio, "sacco.csv", doc2_text)


class TestIngestStatusTracking:
    """set_ingest_status/get_ingest_status back GET /v1/documents/status/{doc_id}
    — same Redis-key-per-job-id pattern as app.main's batch job status,
    not Celery's own AsyncResult (see ingest_task.py's module comment)."""

    def test_successful_ingest_ends_in_complete(self, minio, doc2_text):
        fake_redis = MagicMock()
        cur = FakeCursor()
        _run_ingest(cur, minio, "sacco.md", doc2_text, fake_redis=fake_redis)
        statuses = [c.args[1] for c in fake_redis.set.call_args_list
                    if c.args[0] == "ingest_status:doc-1"]
        assert statuses[0] == "processing"
        assert statuses[-1] == "complete"

    def test_status_keys_use_a_24h_ttl(self, minio, doc2_text):
        fake_redis = MagicMock()
        cur = FakeCursor()
        _run_ingest(cur, minio, "sacco.md", doc2_text, fake_redis=fake_redis)
        for c in fake_redis.set.call_args_list:
            assert c.kwargs.get("ex") == 60 * 60 * 24

    def test_failure_ends_in_failed_with_error_message_and_still_raises(self, minio):
        fake_redis = MagicMock()
        cur = FakeCursor()
        with pytest.raises(ValueError):
            _run_ingest(cur, minio, "bad.csv", "irrelevant", fake_redis=fake_redis)
        statuses = [c.args[1] for c in fake_redis.set.call_args_list
                    if c.args[0] == "ingest_status:doc-1"]
        assert statuses == ["processing", "failed"]
        error_calls = [c for c in fake_redis.set.call_args_list if c.args[0] == "ingest_status:doc-1:error"]
        assert len(error_calls) == 1
        assert "csv" in error_calls[0].args[1].lower() or "unsupported" in error_calls[0].args[1].lower()

    def test_get_ingest_status_reports_unknown_when_nothing_recorded(self):
        from rag_service.ingest_task import get_ingest_status
        fake_redis = MagicMock()
        fake_redis.get.return_value = None
        with patch("rag_service.ingest_task._redis", return_value=fake_redis):
            result = get_ingest_status("never-enqueued")
        assert result == {"status": "unknown", "error": None}

    def test_get_ingest_status_reports_recorded_status_and_error(self):
        from rag_service.ingest_task import get_ingest_status
        fake_redis = MagicMock()
        fake_redis.get.side_effect = lambda key: (
            "failed" if key == "ingest_status:doc-9" else
            "boom" if key == "ingest_status:doc-9:error" else None
        )
        with patch("rag_service.ingest_task._redis", return_value=fake_redis):
            result = get_ingest_status("doc-9")
        assert result == {"status": "failed", "error": "boom"}
