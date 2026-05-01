"""
Per-IP sliding-window rate limiter backed by Upstash Redis (HTTPS REST).

Demo defaults: 10 requests per IP per minute. Tighter caps live behind config.
"""

from __future__ import annotations

import os
import time

from upstash_redis.asyncio import Redis

_client: Redis | None = None
_DEFAULT_LIMIT = 10
_DEFAULT_WINDOW_SECS = 60


def _get_client() -> Redis:
    global _client
    if _client is None:
        url = os.environ.get("UPSTASH_REDIS_REST_URL")
        token = os.environ.get("UPSTASH_REDIS_REST_TOKEN")
        if not url or not token:
            raise RuntimeError("UPSTASH_REDIS_REST_URL/TOKEN are not set")
        _client = Redis(url=url, token=token)
    return _client


async def check_and_consume(
    ip: str,
    *,
    bucket: str = "demo",
    limit: int = _DEFAULT_LIMIT,
    window_secs: int = _DEFAULT_WINDOW_SECS,
) -> tuple[bool, int]:
    """
    Increment the counter for (ip, bucket) and return (allowed, remaining).
    Counter expires `window_secs` after first increment in the window.
    """
    client = _get_client()
    key = f"rl:{bucket}:{ip}:{int(time.time()) // window_secs}"
    count = await client.incr(key)
    if count == 1:
        await client.expire(key, window_secs)
    allowed = count <= limit
    remaining = max(0, limit - count)
    return allowed, remaining
