"""Tests for the chat build (`services/chat.py`)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from services import chat
from services.validation import ValidationError


class _FakeStream:
    """Minimal async context manager that mimics anthropic.AsyncMessageStream."""

    def __init__(self, chunks: list[str], usage: SimpleNamespace):
        self._chunks = chunks
        self._usage = usage

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    @property
    def text_stream(self):
        async def _gen():
            for c in self._chunks:
                yield c
        return _gen()

    async def get_final_message(self):
        return SimpleNamespace(
            usage=self._usage,
            content=[SimpleNamespace(type="text", text="".join(self._chunks))],
        )


def _fake_client(chunks: list[str]) -> SimpleNamespace:
    usage = SimpleNamespace(input_tokens=100, output_tokens=50, cache_read_input_tokens=0)
    stream_obj = _FakeStream(chunks, usage)
    messages = MagicMock()
    messages.stream.return_value = stream_obj
    return SimpleNamespace(messages=messages)


@pytest.fixture
def patch_external(monkeypatch):
    monkeypatch.setattr(chat, "load_chat_snapshot", lambda example_id, user_message: None)
    monkeypatch.setattr(chat, "embed", AsyncMock(return_value=None))
    monkeypatch.setattr(chat, "find_neighbours", AsyncMock(return_value=[]))
    monkeypatch.setattr(chat, "write_request_row", AsyncMock(return_value=None))
    monkeypatch.setattr(chat, "check_input", lambda profile, company: "en")
    monkeypatch.setattr(chat, "is_local", lambda: True)


async def _collect(gen):
    out = []
    async for event in gen:
        out.append(event)
    return out


@pytest.mark.asyncio
async def test_chat_stream_emits_text_then_done(monkeypatch, patch_external):
    client = _fake_client(["Hello", " world"])
    monkeypatch.setattr(chat, "_get_client", lambda: client)

    events = await _collect(
        chat.chat_stream([{"role": "user", "content": "hi"}])
    )
    text_deltas = [e for e in events if e["type"] == "text"]
    done = [e for e in events if e["type"] == "done"]
    assert [e["delta"] for e in text_deltas] == ["Hello", " world"]
    assert len(done) == 1
    assert done[0]["meta"]["model"] == chat.MODEL_ID
    assert done[0]["meta"]["tokens_in"] == 100
    assert done[0]["meta"]["tokens_out"] == 50


@pytest.mark.asyncio
async def test_chat_stream_injects_context_system_block(monkeypatch, patch_external):
    client = _fake_client(["ok"])
    monkeypatch.setattr(chat, "_get_client", lambda: client)

    await _collect(
        chat.chat_stream(
            [{"role": "user", "content": "hi"}],
            context={
                "lead_name": "Maya",
                "profile": "VP Product",
                "company": "Lattice Forge",
            },
        )
    )
    kwargs = client.messages.stream.call_args.kwargs
    system_blocks = kwargs["system"]
    # Base block + the injected context block.
    assert len(system_blocks) == 2
    injected_text = system_blocks[1]["text"]
    assert "Maya" in injected_text
    assert "VP Product" in injected_text
    assert "Lattice Forge" in injected_text
    assert "Active lead" in injected_text


@pytest.mark.asyncio
async def test_chat_stream_without_context_omits_lead_block(monkeypatch, patch_external):
    client = _fake_client(["ok"])
    monkeypatch.setattr(chat, "_get_client", lambda: client)
    await _collect(chat.chat_stream([{"role": "user", "content": "hi"}]))
    kwargs = client.messages.stream.call_args.kwargs
    assert len(kwargs["system"]) == 1


@pytest.mark.asyncio
async def test_chat_stream_validates_from_context_when_present(monkeypatch, patch_external):
    """Context wins over messages for validation routing."""
    seen = {}

    def capture(profile, company):
        seen["profile"] = profile
        seen["company"] = company
        return "en"

    monkeypatch.setattr(chat, "check_input", capture)
    client = _fake_client(["ok"])
    monkeypatch.setattr(chat, "_get_client", lambda: client)

    await _collect(
        chat.chat_stream(
            [{"role": "user", "content": "totally different"}],
            context={"lead_name": "M", "profile": "P", "company": "C"},
        )
    )
    assert seen == {"profile": "P", "company": "C"}


@pytest.mark.asyncio
async def test_chat_stream_validates_from_first_user_when_no_context(monkeypatch, patch_external):
    seen = {}

    def capture(profile, company):
        seen["profile"] = profile
        seen["company"] = company
        return "en"

    monkeypatch.setattr(chat, "check_input", capture)
    client = _fake_client(["ok"])
    monkeypatch.setattr(chat, "_get_client", lambda: client)
    await _collect(
        chat.chat_stream([{"role": "user", "content": "the user message"}])
    )
    assert seen == {"profile": "the user message", "company": None}


@pytest.mark.asyncio
async def test_chat_stream_serves_snapshot_when_available(monkeypatch, patch_external):
    snap = {
        "model": "claude-sonnet-4-6",
        "user_message": "qualify against the icp",
        "assistant_text": "snapshot reply.",
        "usage": {"input_tokens": 500, "output_tokens": 200, "latency_ms": 42},
    }
    monkeypatch.setattr(chat, "load_chat_snapshot", lambda example_id, user_message: snap)
    # Anthropic client must NOT be touched.
    client = _fake_client(["should not run"])
    monkeypatch.setattr(chat, "_get_client", lambda: client)

    events = await _collect(
        chat.chat_stream(
            [{"role": "user", "content": "Qualify against the ICP"}],
            example_id="1",
        )
    )
    client.messages.stream.assert_not_called()
    text_events = [e for e in events if e["type"] == "text"]
    done = [e for e in events if e["type"] == "done"]
    assert text_events[0]["delta"] == "snapshot reply."
    assert done[0]["meta"]["snapshot_served"] is True
    assert done[0]["meta"]["latency_ms"] == 42


@pytest.mark.asyncio
async def test_chat_stream_propagates_validation_error(monkeypatch, patch_external):
    def reject(profile, company):
        raise ValidationError(code="empty_profile", message="empty")

    monkeypatch.setattr(chat, "check_input", reject)
    client = _fake_client(["ok"])
    monkeypatch.setattr(chat, "_get_client", lambda: client)
    with pytest.raises(ValidationError):
        await _collect(chat.chat_stream([{"role": "user", "content": ""}]))


@pytest.mark.asyncio
async def test_chat_stream_bypass_cache_skips_snapshot_lookup(monkeypatch, patch_external):
    def fake_load(*_args, **_kwargs):
        raise AssertionError("load_chat_snapshot must not be called when bypass_cache=True")

    monkeypatch.setattr(chat, "load_chat_snapshot", fake_load)
    client = _fake_client(["from anthropic"])
    monkeypatch.setattr(chat, "_get_client", lambda: client)

    await _collect(
        chat.chat_stream(
            [{"role": "user", "content": "Qualify against the ICP"}],
            example_id="1",
            bypass_cache=True,
        )
    )
    client.messages.stream.assert_called_once()


@pytest.mark.asyncio
async def test_chat_stream_snapshot_only_for_single_turn(monkeypatch, patch_external):
    """Multi-turn requests must skip the snapshot lookup — snapshots are
    keyed to canonical starters, not arbitrary turns."""
    def fake_load(*_args, **_kwargs):
        raise AssertionError("load_chat_snapshot must not be called on multi-turn requests")

    monkeypatch.setattr(chat, "load_chat_snapshot", fake_load)
    client = _fake_client(["ok"])
    monkeypatch.setattr(chat, "_get_client", lambda: client)

    await _collect(
        chat.chat_stream(
            [
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": "..."},
                {"role": "user", "content": "second"},
            ],
            example_id="1",
        )
    )
