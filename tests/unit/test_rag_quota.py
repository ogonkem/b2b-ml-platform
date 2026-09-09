"""
tests/unit/test_rag_quota.py
Real (non-mocked) exercise of rag_service.main._check_and_increment_quota
against an in-memory fake Redis — mirrors tests/unit/test_quota.py's
approach for app.main.check_and_increment_quota, since the two functions
are deliberately built to the same pattern (check-before-increment, atomic
INCRBY, ~32-day TTL) under distinct key prefixes.
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pytest
from fastapi import HTTPException

with patch("psycopg2.connect") as _mock_connect, \
     patch("minio.Minio") as _mock_minio_cls:
    _mock_connect.return_value = MagicMock()
    _mock_minio_cls.return_value = MagicMock(bucket_exists=lambda *_: True)

    import rag_service.main as rag_main
    from rag_service.main import (
        RAG_INGESTION_QUOTA_PREFIX,
        RAG_RETRIEVAL_QUOTA_PREFIX,
        _check_and_increment_quota,
        _current_usage,
    )


class FakeRedis:
    """Just enough of the redis-py surface for _check_and_increment_quota:
    GET/INCRBY/EXPIRE backed by a plain dict, so increments are real."""

    def __init__(self):
        self.store = {}
        self.ttls = {}

    def get(self, key):
        value = self.store.get(key)
        return None if value is None else str(value)

    def incrby(self, key, amount):
        self.store[key] = self.store.get(key, 0) + amount
        return self.store[key]

    def expire(self, key, seconds):
        self.ttls[key] = seconds


@pytest.fixture
def fake_redis():
    fr = FakeRedis()
    with patch.object(rag_main, "_redis_client", fr):
        yield fr


def _month_key() -> str:
    from datetime import datetime
    return datetime.utcnow().strftime("%Y_%m")


class TestQuotaMechanics:

    def test_first_request_increments_from_zero(self, fake_redis):
        result = _check_and_increment_quota("tenant-x", RAG_INGESTION_QUOTA_PREFIX, "ingestion", 1, 100)
        assert result == 1

    def test_increments_by_given_amount(self, fake_redis):
        _check_and_increment_quota("tenant-x", RAG_INGESTION_QUOTA_PREFIX, "ingestion", 5, 100)
        key = f"{RAG_INGESTION_QUOTA_PREFIX}:tenant-x:{_month_key()}"
        assert fake_redis.store[key] == 5

    def test_ttl_set_on_first_write(self, fake_redis):
        _check_and_increment_quota("tenant-x", RAG_INGESTION_QUOTA_PREFIX, "ingestion", 1, 100)
        key = f"{RAG_INGESTION_QUOTA_PREFIX}:tenant-x:{_month_key()}"
        assert fake_redis.ttls[key] == 60 * 60 * 24 * 32

    def test_boundary_exactly_at_limit_is_allowed(self, fake_redis):
        key = f"{RAG_INGESTION_QUOTA_PREFIX}:tenant-x:{_month_key()}"
        fake_redis.store[key] = 99
        result = _check_and_increment_quota("tenant-x", RAG_INGESTION_QUOTA_PREFIX, "ingestion", 1, 100)
        assert result == 100

    def test_raises_429_when_exceeding_limit(self, fake_redis):
        key = f"{RAG_INGESTION_QUOTA_PREFIX}:tenant-x:{_month_key()}"
        fake_redis.store[key] = 99
        with pytest.raises(HTTPException) as exc_info:
            _check_and_increment_quota("tenant-x", RAG_INGESTION_QUOTA_PREFIX, "ingestion", 5, 100)
        assert exc_info.value.status_code == 429
        assert "ingestion" in exc_info.value.detail.lower()

    def test_rejected_request_does_not_consume_quota(self, fake_redis):
        key = f"{RAG_INGESTION_QUOTA_PREFIX}:tenant-x:{_month_key()}"
        fake_redis.store[key] = 99
        with pytest.raises(HTTPException):
            _check_and_increment_quota("tenant-x", RAG_INGESTION_QUOTA_PREFIX, "ingestion", 5, 100)
        assert fake_redis.store[key] == 99

    def test_different_tenants_have_independent_quotas(self, fake_redis):
        _check_and_increment_quota("tenant-a", RAG_INGESTION_QUOTA_PREFIX, "ingestion", 99, 100)
        result = _check_and_increment_quota("tenant-b", RAG_INGESTION_QUOTA_PREFIX, "ingestion", 99, 100)
        assert result == 99

    def test_ingestion_and_retrieval_counters_are_independent(self, fake_redis):
        _check_and_increment_quota("tenant-x", RAG_INGESTION_QUOTA_PREFIX, "ingestion", 5, 100)
        _check_and_increment_quota("tenant-x", RAG_RETRIEVAL_QUOTA_PREFIX, "retrieval", 3, 100)
        assert _current_usage("tenant-x", RAG_INGESTION_QUOTA_PREFIX) == 5
        assert _current_usage("tenant-x", RAG_RETRIEVAL_QUOTA_PREFIX) == 3

    def test_does_not_collide_with_selastones_own_prediction_quota_key(self, fake_redis):
        """app.main's own prediction quota lives under the "quota:" prefix
        on the same Redis instance/DB. If rag_service reused that literal
        prefix, a tenant's RAG usage would silently share (and corrupt) the
        same counter as its prediction usage. Confirm the real keys used
        here never collide with that prefix."""
        _check_and_increment_quota("tenant-x", RAG_INGESTION_QUOTA_PREFIX, "ingestion", 1, 100)
        _check_and_increment_quota("tenant-x", RAG_RETRIEVAL_QUOTA_PREFIX, "retrieval", 1, 100)
        assert f"quota:tenant-x:{_month_key()}" not in fake_redis.store
        assert all(k.startswith(("rag_ingest_quota:", "rag_retrieve_quota:")) for k in fake_redis.store)

    def test_current_usage_does_not_increment(self, fake_redis):
        _check_and_increment_quota("tenant-x", RAG_INGESTION_QUOTA_PREFIX, "ingestion", 7, 100)
        assert _current_usage("tenant-x", RAG_INGESTION_QUOTA_PREFIX) == 7
        assert _current_usage("tenant-x", RAG_INGESTION_QUOTA_PREFIX) == 7   # reading again doesn't move it

    def test_current_usage_is_zero_when_nothing_recorded(self, fake_redis):
        assert _current_usage("tenant-new", RAG_INGESTION_QUOTA_PREFIX) == 0
