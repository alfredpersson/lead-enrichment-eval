"""Tests for the FastAPI surface (`services/web.py`)."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest
from fastapi import Request
from fastapi.testclient import TestClient

from services import web
from services.validation import ValidationError


async def _empty_stream(*args, **kwargs):
    return
    yield  # makes this an async generator that yields nothing


@pytest.fixture
def app(monkeypatch):
    """FastAPI app with all downstream dependencies patched to harmless defaults.
    Tests override specific patches before reading the client."""
    monkeypatch.setattr(web, "check_and_consume", AsyncMock(return_value=(True, 9)))
    monkeypatch.setattr(web, "find_neighbours", AsyncMock(return_value=[]))
    monkeypatch.setattr(web, "enrich_lead_stream", _empty_stream)
    monkeypatch.setattr(web, "chat_stream", _empty_stream)
    return TestClient(web.build_app())


def _sse_events(text: str) -> list[dict]:
    return [
        json.loads(line[len("data: ") :])
        for line in text.splitlines()
        if line.startswith("data: ")
    ]


def test_healthz(app):
    r = app.get("/healthz")
    assert r.status_code == 200
    assert r.json() == {"ok": True}


def test_neighbours_endpoint_returns_payload(app, monkeypatch):
    monkeypatch.setattr(
        web,
        "find_neighbours",
        AsyncMock(return_value=[{"id": "1", "score": 0.9, "similarity": 0.95}]),
    )
    r = app.post("/neighbours", json={"text": "Maya Chen VP Product", "k": 3})
    assert r.status_code == 200
    assert r.json() == {"neighbours": [{"id": "1", "score": 0.9, "similarity": 0.95}]}


def test_enrich_endpoint_streams_events(app, monkeypatch):
    async def fake_enrich(profile, company, *, example_id=None, bypass_cache=False):
        yield {"type": "thinking", "delta": "step 1"}
        yield {"type": "result", "output": {"action": "auto_add"}}

    monkeypatch.setattr(web, "enrich_lead_stream", fake_enrich)
    r = app.post("/enrich", json={"profile": "Maya is VP Product", "company": None})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    events = _sse_events(r.text)
    assert events == [
        {"type": "thinking", "delta": "step 1"},
        {"type": "result", "output": {"action": "auto_add"}},
    ]


def test_enrich_endpoint_emits_validation_error_event(app, monkeypatch):
    async def fake_enrich(*args, **kwargs):
        raise ValidationError(code="empty_profile", message="Profile is required.")
        yield  # pragma: no cover

    monkeypatch.setattr(web, "enrich_lead_stream", fake_enrich)
    r = app.post("/enrich", json={"profile": ""})
    assert r.status_code == 200  # The error rides through the SSE stream.
    assert _sse_events(r.text) == [
        {"type": "error", "code": "empty_profile", "message": "Profile is required."}
    ]


def test_chat_endpoint_streams_text(app, monkeypatch):
    async def fake_chat(messages, *, example_id=None, context=None, bypass_cache=False):
        yield {"type": "text", "delta": "hello"}
        yield {"type": "done", "meta": {"model": "x"}}

    monkeypatch.setattr(web, "chat_stream", fake_chat)
    r = app.post("/chat", json={"messages": [{"role": "user", "content": "hi"}]})
    events = _sse_events(r.text)
    assert events == [
        {"type": "text", "delta": "hello"},
        {"type": "done", "meta": {"model": "x"}},
    ]


def test_chat_endpoint_forwards_context_and_example_id(app, monkeypatch):
    captured = {}

    async def fake_chat(messages, *, example_id=None, context=None, bypass_cache=False):
        captured["context"] = context
        captured["example_id"] = example_id
        yield {"type": "done", "meta": {}}

    monkeypatch.setattr(web, "chat_stream", fake_chat)
    r = app.post(
        "/chat",
        json={
            "messages": [{"role": "user", "content": "hi"}],
            "example_id": "1",
            "context": {"lead_name": "Maya", "profile": "VP", "company": "LF"},
        },
    )
    assert r.status_code == 200
    assert captured["example_id"] == "1"
    assert captured["context"] == {"lead_name": "Maya", "profile": "VP", "company": "LF"}


def test_rate_limit_returns_429(app, monkeypatch):
    monkeypatch.setattr(web, "check_and_consume", AsyncMock(return_value=(False, 0)))
    r = app.post("/enrich", json={"profile": "p"})
    assert r.status_code == 429
    assert "Rate limit" in r.json()["detail"]


def test_rate_limit_called_per_endpoint_bucket(app, monkeypatch):
    """Each endpoint must gate against its own bucket name."""
    buckets: list[str] = []

    async def fake_gate(ip, *, bucket):
        buckets.append(bucket)
        return True, 9

    monkeypatch.setattr(web, "check_and_consume", fake_gate)
    app.post("/enrich", json={"profile": "p"})
    app.post("/chat", json={"messages": [{"role": "user", "content": "hi"}]})
    app.post("/neighbours", json={"text": "t"})
    assert buckets == ["enrich", "chat", "neighbours"]


def _request(headers: list[tuple[bytes, bytes]], client: tuple[str, int] | None) -> Request:
    return Request({"type": "http", "headers": headers, "client": client})


def test_client_ip_prefers_x_forwarded_for():
    req = _request([(b"x-forwarded-for", b"203.0.113.5, 10.0.0.1")], ("10.0.0.99", 12345))
    assert web._client_ip(req) == "203.0.113.5"


def test_client_ip_falls_back_to_client_host():
    assert web._client_ip(_request([], ("10.0.0.99", 12345))) == "10.0.0.99"


def test_client_ip_unknown_when_no_client():
    assert web._client_ip(_request([], None)) == "unknown"
