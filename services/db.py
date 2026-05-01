"""
Postgres connection pool against Neon's pooled (PgBouncer) connection string.

Modal functions are short-lived and burst-y; PgBouncer transaction mode in front
of Neon is what keeps us from exhausting the per-database connection cap. Set
DATABASE_URL to the pooled connection string (the one ending in
`-pooler.<region>.aws.neon.tech`).
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import AsyncIterator

from psycopg import AsyncConnection
from psycopg_pool import AsyncConnectionPool

_pool: AsyncConnectionPool | None = None


def _connection_string() -> str:
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError("DATABASE_URL is not set")
    return url


async def get_pool() -> AsyncConnectionPool:
    global _pool
    if _pool is None:
        _pool = AsyncConnectionPool(
            conninfo=_connection_string(),
            min_size=0,
            max_size=4,
            open=False,
        )
        await _pool.open()
    return _pool


@asynccontextmanager
async def connection() -> AsyncIterator[AsyncConnection]:
    pool = await get_pool()
    async with pool.connection() as conn:
        yield conn
