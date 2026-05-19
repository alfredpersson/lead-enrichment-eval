"""Tests for the per-IP sliding-window rate limiter."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from services import ratelimit


def _fake_redis(count: int) -> SimpleNamespace:
    return SimpleNamespace(
        incr=AsyncMock(return_value=count),
        expire=AsyncMock(return_value=True),
    )


@pytest.mark.asyncio
async def test_check_and_consume_short_circuits_locally(monkeypatch):
    monkeypatch.setattr(ratelimit, "is_local", lambda: True)
    allowed, remaining = await ratelimit.check_and_consume("1.2.3.4", bucket="enrich")
    assert allowed is True
    assert remaining == ratelimit._DEFAULT_LIMIT


@pytest.mark.asyncio
async def test_check_and_consume_allows_under_limit(monkeypatch):
    monkeypatch.setattr(ratelimit, "is_local", lambda: False)
    fake = _fake_redis(count=3)
    monkeypatch.setattr(ratelimit, "_get_client", lambda: fake)

    allowed, remaining = await ratelimit.check_and_consume(
        "1.2.3.4", bucket="enrich", limit=10, window_secs=60
    )
    assert allowed is True
    assert remaining == 7
    # expire only fires on the first increment in a window (count==1).
    fake.expire.assert_not_called()


@pytest.mark.asyncio
async def test_check_and_consume_sets_expire_on_first_increment(monkeypatch):
    monkeypatch.setattr(ratelimit, "is_local", lambda: False)
    fake = _fake_redis(count=1)
    monkeypatch.setattr(ratelimit, "_get_client", lambda: fake)

    await ratelimit.check_and_consume(
        "1.2.3.4", bucket="enrich", limit=10, window_secs=60
    )
    fake.expire.assert_awaited_once()
    key, ttl = fake.expire.await_args.args
    assert "enrich" in key
    assert "1.2.3.4" in key
    assert ttl == 60


@pytest.mark.asyncio
async def test_check_and_consume_blocks_at_limit(monkeypatch):
    monkeypatch.setattr(ratelimit, "is_local", lambda: False)
    fake = _fake_redis(count=11)
    monkeypatch.setattr(ratelimit, "_get_client", lambda: fake)

    allowed, remaining = await ratelimit.check_and_consume(
        "1.2.3.4", bucket="enrich", limit=10, window_secs=60
    )
    assert allowed is False
    assert remaining == 0


def test_get_client_raises_without_env(monkeypatch):
    monkeypatch.setattr(ratelimit, "_client", None)
    monkeypatch.delenv("UPSTASH_REDIS_REST_URL", raising=False)
    monkeypatch.delenv("UPSTASH_REDIS_REST_TOKEN", raising=False)
    with pytest.raises(RuntimeError, match="UPSTASH_REDIS_REST_URL"):
        ratelimit._get_client()
