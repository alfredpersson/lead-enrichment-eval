"""Tests for the telemetry writer's local-mode short-circuit and SQL build."""

from __future__ import annotations

from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from services import telemetry


@pytest.mark.asyncio
async def test_write_request_row_noop_in_local(monkeypatch):
    monkeypatch.setattr(telemetry, "is_local", lambda: True)

    @asynccontextmanager
    async def fail_connection():
        raise AssertionError("connection() must not be called in local mode")
        yield  # pragma: no cover

    monkeypatch.setattr(telemetry, "connection", fail_connection)
    await telemetry.write_request_row({"request_id": "x", "mode": "integrated"})


@pytest.mark.asyncio
async def test_write_request_row_builds_sql_with_row_keys(monkeypatch):
    monkeypatch.setattr(telemetry, "is_local", lambda: False)

    execute = AsyncMock()
    commit = AsyncMock()
    cursor = SimpleNamespace(execute=execute)

    @asynccontextmanager
    async def cursor_cm():
        yield cursor

    @asynccontextmanager
    async def fake_connection():
        yield SimpleNamespace(cursor=lambda: cursor_cm(), commit=commit)

    monkeypatch.setattr(telemetry, "connection", fake_connection)

    row = {"request_id": "abc", "mode": "integrated", "latency_ms": 200}
    await telemetry.write_request_row(row)

    execute.assert_awaited_once()
    sql, params = execute.await_args.args
    assert sql.startswith("INSERT INTO requests")
    for key in row:
        assert key in sql
        assert f"%({key})s" in sql
    assert params == row
    commit.assert_awaited_once()
