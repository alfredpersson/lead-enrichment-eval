"""Tests for the multi-turn chat loop and the per-call inference parsers."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from services.eval.dataset import EvalItem
from services.eval.inference import (
    ChatResult,
    _parse_chat,
    _parse_integrated,
    run_chat_multiturn,
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
