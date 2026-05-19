"""Tests for the multi-turn chat loop and the per-call inference parsers."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from services.eval.dataset import EvalItem
from services.eval.inference import (
    CHAT_MODEL_ID,
    INTEGRATED_MODEL_ID,
    _call_chat,
    _call_integrated,
    _parse_chat,
    _parse_integrated,
    run_chat,
    run_chat_multiturn,
    run_integrated,
)


def _item(item_id: str = "x") -> EvalItem:
    return EvalItem(
        id=item_id,
        kind="exemplar",
        scenario="t",
        label="t",
        profile="Maya is VP Product at Lattice Forge.",
        company=None,
        gold={"input_lang": "en", "expected_action": "auto_add"},
    )


def _mock_text_response(text: str, *, in_tok: int = 100, out_tok: int = 50):
    usage = SimpleNamespace(
        input_tokens=in_tok, output_tokens=out_tok, cache_read_input_tokens=0
    )
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text)],
        usage=usage,
        stop_reason="end_turn",
    )


def _mock_tool_response(tool_input: dict):
    usage = SimpleNamespace(input_tokens=120, output_tokens=80, cache_read_input_tokens=0)
    return SimpleNamespace(
        content=[
            SimpleNamespace(type="tool_use", input=tool_input, name="enrich_lead")
        ],
        usage=usage,
        stop_reason="tool_use",
    )


# ----- _parse_* helpers ------------------------------------------------------


def test_parse_chat_extracts_text():
    resp = _mock_text_response("Hello there")
    result = _parse_chat("x", resp, latency_ms=42)
    assert result.success is True
    assert result.text == "Hello there"
    assert result.latency_ms == 42


def test_parse_chat_empty_text_marks_failure():
    resp = _mock_text_response("")
    result = _parse_chat("x", resp, latency_ms=42)
    assert result.success is False
    assert result.error == "empty response"


def test_parse_integrated_returns_tool_input():
    resp = _mock_tool_response({"action": "auto_add"})
    result = _parse_integrated("x", resp, latency_ms=100)
    assert result.success is True
    assert result.output == {"action": "auto_add"}


def test_parse_integrated_missing_tool_use_marks_failure():
    resp = _mock_text_response("just text, no tool")
    result = _parse_integrated("x", resp, latency_ms=100)
    assert result.success is False
    assert result.output is None


# ----- run_chat_multiturn end-to-end -----------------------------------------


@pytest.mark.asyncio
async def test_multiturn_stops_when_extractor_completes_on_first_turn():
    client = SimpleNamespace(
        messages=SimpleNamespace(
            create=AsyncMock(return_value=_mock_text_response("Strong fit response."))
        )
    )

    async def check(_item, _text):
        return True  # extractor reports complete immediately

    results = await run_chat_multiturn(
        [_item()], extract_and_check=check, max_turns=3, client=client
    )
    assert len(results) == 1
    r = results[0]
    assert r.success is True
    assert r.cap_hit is False
    assert r.turns_used == 1
    assert client.messages.create.await_count == 1


@pytest.mark.asyncio
async def test_multiturn_caps_at_three_turns_and_records_failure():
    client = SimpleNamespace(
        messages=SimpleNamespace(
            create=AsyncMock(
                side_effect=[
                    _mock_text_response("First reply, missing fields."),
                    _mock_text_response("Second reply, still missing."),
                    _mock_text_response("Third reply, still missing."),
                ]
            )
        )
    )

    async def check(_item, _text):
        return False  # never complete

    results = await run_chat_multiturn(
        [_item()], extract_and_check=check, max_turns=3, client=client
    )
    r = results[0]
    assert r.cap_hit is True
    assert r.success is False
    assert r.turns_used == 3
    assert client.messages.create.await_count == 3


@pytest.mark.asyncio
async def test_multiturn_stops_when_extractor_completes_mid_run():
    responses = [
        _mock_text_response("Turn 1 reply, partial."),
        _mock_text_response("Turn 2 reply, now complete."),
    ]
    client = SimpleNamespace(
        messages=SimpleNamespace(create=AsyncMock(side_effect=responses))
    )

    states = [False, True]  # first incomplete, second complete

    async def check(_item, _text):
        return states.pop(0)

    results = await run_chat_multiturn(
        [_item()], extract_and_check=check, max_turns=3, client=client
    )
    r = results[0]
    assert r.success is True
    assert r.cap_hit is False
    assert r.turns_used == 2
    assert client.messages.create.await_count == 2


@pytest.mark.asyncio
async def test_multiturn_empty_assistant_breaks_loop():
    client = SimpleNamespace(
        messages=SimpleNamespace(create=AsyncMock(return_value=_mock_text_response("")))
    )

    async def check(_item, _text):
        return False

    results = await run_chat_multiturn(
        [_item()], extract_and_check=check, max_turns=3, client=client
    )
    r = results[0]
    assert r.success is False
    assert r.cap_hit is True
    assert r.turns_used == 1
    assert "empty" in (r.error or "")


@pytest.mark.asyncio
async def test_multiturn_propagates_exception_and_marks_cap_hit():
    client = SimpleNamespace(
        messages=SimpleNamespace(create=AsyncMock(side_effect=RuntimeError("boom")))
    )

    async def check(_item, _text):
        return False

    results = await run_chat_multiturn(
        [_item()], extract_and_check=check, max_turns=3, client=client
    )
    r = results[0]
    assert r.success is False
    assert r.cap_hit is True
    assert "boom" in (r.error or "")


# ----- _call_integrated / _call_chat per-call drivers ----------------------


@pytest.mark.asyncio
async def test_call_integrated_sends_thinking_and_tool():
    """The integrated driver must request thinking + the enrich_lead tool."""
    captured = {}

    async def fake_create(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            content=[
                SimpleNamespace(
                    type="tool_use",
                    name="enrich_lead",
                    input={"action": "auto_add"},
                )
            ],
            usage=SimpleNamespace(
                input_tokens=100, output_tokens=50, cache_read_input_tokens=0
            ),
            stop_reason="tool_use",
        )

    client = SimpleNamespace(messages=SimpleNamespace(create=fake_create))
    sem = asyncio.Semaphore(1)
    result = await _call_integrated(client, _item(), sem)
    assert result.success is True
    assert result.output == {"action": "auto_add"}
    assert captured["model"] == INTEGRATED_MODEL_ID
    assert captured["thinking"]["type"] == "enabled"
    assert any(t["name"] == "enrich_lead" for t in captured["tools"])
    assert captured["tool_choice"] == {"type": "auto"}


@pytest.mark.asyncio
async def test_call_integrated_handles_exception():
    client = SimpleNamespace(
        messages=SimpleNamespace(create=AsyncMock(side_effect=RuntimeError("nope")))
    )
    sem = asyncio.Semaphore(1)
    result = await _call_integrated(client, _item(), sem)
    assert result.success is False
    assert "RuntimeError" in (result.error or "")
    assert result.output is None


@pytest.mark.asyncio
async def test_call_chat_sends_no_thinking_no_tools():
    captured = {}

    async def fake_create(**kwargs):
        captured.update(kwargs)
        return _mock_text_response("hi")

    client = SimpleNamespace(messages=SimpleNamespace(create=fake_create))
    sem = asyncio.Semaphore(1)
    result = await _call_chat(client, _item(), sem)
    assert result.success is True
    assert result.text == "hi"
    assert captured["model"] == CHAT_MODEL_ID
    assert "thinking" not in captured
    assert "tools" not in captured


@pytest.mark.asyncio
async def test_call_chat_handles_exception():
    client = SimpleNamespace(
        messages=SimpleNamespace(create=AsyncMock(side_effect=RuntimeError("net")))
    )
    sem = asyncio.Semaphore(1)
    result = await _call_chat(client, _item(), sem)
    assert result.success is False
    assert "RuntimeError" in (result.error or "")


# ----- run_integrated / run_chat fan-out + on_done -------------------------


@pytest.mark.asyncio
async def test_run_integrated_fires_on_done_per_item():
    tool_resp = SimpleNamespace(
        content=[
            SimpleNamespace(type="tool_use", name="enrich_lead", input={"action": "auto_add"})
        ],
        usage=SimpleNamespace(input_tokens=10, output_tokens=5, cache_read_input_tokens=0),
        stop_reason="tool_use",
    )
    client = SimpleNamespace(
        messages=SimpleNamespace(create=AsyncMock(return_value=tool_resp))
    )
    items = [_item(f"id-{i}") for i in range(3)]
    fired: list[str] = []

    results = await run_integrated(items, client=client, on_done=lambda r: fired.append(r.item_id))
    assert {r.item_id for r in results} == {"id-0", "id-1", "id-2"}
    assert sorted(fired) == ["id-0", "id-1", "id-2"]
    # All results carry the integrated output shape.
    for r in results:
        assert r.output == {"action": "auto_add"}


@pytest.mark.asyncio
async def test_run_integrated_empty_input():
    """Empty input must short-circuit without building a client."""
    # No client provided; would raise if construction were attempted.
    result = await run_integrated([])
    assert result == []


@pytest.mark.asyncio
async def test_run_chat_fires_on_done_per_item():
    client = SimpleNamespace(
        messages=SimpleNamespace(create=AsyncMock(return_value=_mock_text_response("reply")))
    )
    items = [_item(f"id-{i}") for i in range(2)]
    fired: list[str] = []
    results = await run_chat(items, client=client, on_done=lambda r: fired.append(r.item_id))
    assert sorted(r.item_id for r in results) == ["id-0", "id-1"]
    assert sorted(fired) == ["id-0", "id-1"]


@pytest.mark.asyncio
async def test_run_chat_empty_input():
    assert await run_chat([]) == []


@pytest.mark.asyncio
async def test_run_chat_multiturn_empty_input():
    async def check(_item, _text):
        return True

    assert await run_chat_multiturn([], extract_and_check=check) == []
