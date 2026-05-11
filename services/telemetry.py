"""
Telemetry write for both modes.

One row per call. Raw input text is never written; only structural fields and
metrics. Schema lives in migrations/0001_init.sql.
"""

from __future__ import annotations

from typing import Any

from services.config import is_local
from services.db import connection


async def write_request_row(row: dict[str, Any]) -> None:
    """
    Insert a row into `requests`. Caller passes a dict whose keys match column
    names; missing columns are NULL.
    """
    if is_local():
        return
    columns = list(row.keys())
    placeholders = ", ".join(f"%({c})s" for c in columns)
    column_list = ", ".join(columns)
    sql = f"INSERT INTO requests ({column_list}) VALUES ({placeholders})"
    async with connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(sql, row)
        await conn.commit()
