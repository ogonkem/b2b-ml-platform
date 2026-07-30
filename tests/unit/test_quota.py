"""
tests/unit/test_quota.py
Real (non-mocked) exercise of check_and_increment_quota's own logic against
an in-memory fake Redis — every other test suite mocks this function out
entirely, so its actual check-before-increment behavior had zero direct
coverage.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import os
import pytest
from unittest.mock import patch, MagicMock

os.environ["API_TOKENS"]         = "dev-token,tenant-abc"
os.environ["USE_REAL_ARTEFACTS"] = "false"

with patch("redis.Redis") as mock_redis, \
     patch("clickhouse_connect.get_client") as mock_ch:
    mock_redis.return_value = MagicMock()
    mock_ch.return_value    = MagicMock()
    import app.main as app_main
    from app.main import check_and_increment_quota

from fastapi import HTTPException


class FakeRedis:
    """Just enough of the redis-py surface for check_and_increment_quota:
    GET/INCRBY/EXPIRE backed by a plain dict, so increments are real."""

    def __init__(self):
        self.store = {}
        self.ttls  = {}

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
    with patch.object(app_main, "redis_client", fr):
        yield fr


def test_first_request_increments_from_zero(fake_redis):
    result = check_and_increment_quota("tenant-x", increment=1)
    assert result == 1
    assert fake_redis.store["quota:tenant-x:" + _month_key()] == 1


def test_increments_by_given_amount(fake_redis):
    check_and_increment_quota("tenant-x", increment=50)
    assert fake_redis.store["quota:tenant-x:" + _month_key()] == 50


def test_ttl_set_on_first_write(fake_redis):
    check_and_increment_quota("tenant-x", increment=1)
    assert "quota:tenant-x:" + _month_key() in fake_redis.ttls


def test_boundary_exactly_at_limit_is_allowed(fake_redis):
    key = "quota:tenant-x:" + _month_key()
    fake_redis.store[key] = 999
    result = check_and_increment_quota("tenant-x", increment=1, monthly_limit=1000)
    assert result == 1000


def test_raises_429_when_exceeding_limit(fake_redis):
    key = "quota:tenant-x:" + _month_key()
    fake_redis.store[key] = 999
    with pytest.raises(HTTPException) as exc_info:
        check_and_increment_quota("tenant-x", increment=50, monthly_limit=1000)
    assert exc_info.value.status_code == 429


def test_rejected_request_does_not_consume_quota(fake_redis):
    """The bug: incrementing before checking permanently charged the tenant
    for a request that never went through. A single oversized request must
    leave the counter untouched when rejected."""
    key = "quota:tenant-x:" + _month_key()
    fake_redis.store[key] = 999
    with pytest.raises(HTTPException):
        check_and_increment_quota("tenant-x", increment=50, monthly_limit=1000)
    assert fake_redis.store[key] == 999


def test_repeated_rejections_never_move_the_counter(fake_redis):
    key = "quota:tenant-x:" + _month_key()
    fake_redis.store[key] = 999
    for _ in range(5):
        with pytest.raises(HTTPException):
            check_and_increment_quota("tenant-x", increment=2000, monthly_limit=1000)
    assert fake_redis.store[key] == 999


def test_accepted_request_after_prior_usage(fake_redis):
    key = "quota:tenant-x:" + _month_key()
    fake_redis.store[key] = 500
    result = check_and_increment_quota("tenant-x", increment=400, monthly_limit=1000)
    assert result == 900


def test_different_tenants_have_independent_quotas(fake_redis):
    check_and_increment_quota("tenant-a", increment=999, monthly_limit=1000)
    result = check_and_increment_quota("tenant-b", increment=999, monthly_limit=1000)
    assert result == 999


def _month_key() -> str:
    from datetime import datetime
    return datetime.utcnow().strftime("%Y_%m")
