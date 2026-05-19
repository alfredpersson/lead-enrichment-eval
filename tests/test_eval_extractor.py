"""Tests for the chat-output extractor heuristic and refusal handling."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from services.eval.dataset import EvalItem
from services.eval.extractor import (
    REFUSAL_HINTS,
    _is_complete,
    extract_chat_outputs,
    extract_one_text,
    make_completeness_check,
)
from services.eval.inference import InferenceResult


def _item() -> EvalItem:
    return EvalItem(
        id="x",
        kind="exemplar",
        scenario="t",
        label="t",
        profile="Maya is VP Product at Lattice Forge Series B.",
        company=None,
        gold={"input_lang": "en", "expected_action": "auto_add"},
    )


# ----- _is_complete ----------------------------------------------------------


def _good_output(action: str = "auto_add") -> dict:
    return {
        "classification": {
            "industry": "B2B SaaS",
            "segment": "sales",
            "seniority": "VP",
            "company_size": "51-200",
        },
        "fit_score": {"value": 0.9, "dimensions": {}},
        "claims": [{"text": "VP", "source_quote": "VP Product", "confidence": 0.9}],
        "draft_hook": {"text": "Reach out", "claims_used": [], "confidence": 0.8},
        "action": action,
    }


def test_is_complete_happy_path():
    assert _is_complete(_good_output()) is True


def test_is_complete_missing_classification_field():
    out = _good_output()
    out["classification"]["industry"] = ""
    assert _is_complete(out) is False


def test_is_complete_missing_fit_value():
    out = _good_output()
    out["fit_score"] = {}
    assert _is_complete(out) is False


def test_is_complete_discard_does_not_require_hook():
    out = _good_output(action="discard")
    out["draft_hook"]["text"] = ""
    out["claims"] = []
    assert _is_complete(out) is True


def test_is_complete_refuse_does_not_require_claims_or_hook():
    out = _good_output(action="refuse")
    out["claims"] = []
    out["draft_hook"] = {"text": "", "claims_used": [], "confidence": 0.0}
    assert _is_complete(out) is True


def test_is_complete_unknown_action_fails():
    out = _good_output()
    out["action"] = "elsewhere"
    assert _is_complete(out) is False


def test_is_complete_none():
    assert _is_complete(None) is False


# ----- extract_one_text refusal synthesis ------------------------------------


def _mock_response_no_tool(text: str = "I can't help with that lead."):
    """Build a fake messages.create response with one text block, no tool_use."""
    text_block = SimpleNamespace(type="text", text=text)
    usage = SimpleNamespace(input_tokens=10, output_tokens=4)
    return SimpleNamespace(content=[text_block], usage=usage)


def _mock_response_tool(tool_input: dict):
    tool_block = SimpleNamespace(type="tool_use", input=tool_input, name="enrich_lead")
    usage = SimpleNamespace(input_tokens=15, output_tokens=12)
    return SimpleNamespace(content=[tool_block], usage=usage)


@pytest.mark.asyncio
async def test_extractor_returns_refuse_on_refusal_phrase():
    client = SimpleNamespace(
        messages=SimpleNamespace(create=AsyncMock(return_value=_mock_response_no_tool()))
    )
    sem = asyncio.Semaphore(1)
    result = await extract_one_text(client, _item(), "I can't help with that.", sem)
    assert result.success is True
    assert result.extractor_complete is True
    assert result.output is not None
    assert result.output["action"] == "refuse"


@pytest.mark.asyncio
async def test_extractor_records_unparsed_when_no_refusal_phrase():
    client = SimpleNamespace(
        messages=SimpleNamespace(
            create=AsyncMock(return_value=_mock_response_no_tool("I have a question for you instead."))
        )
    )
    sem = asyncio.Semaphore(1)
    result = await extract_one_text(client, _item(), "Some chitchat reply.", sem)
    assert result.success is False
    assert result.output is None
    assert result.notes_unparsed is not None
    assert "no tool_use" in (result.error or "")


@pytest.mark.asyncio
async def test_extractor_happy_path_with_tool():
    out = _good_output()
    client = SimpleNamespace(
        messages=SimpleNamespace(create=AsyncMock(return_value=_mock_response_tool(out)))
    )
    sem = asyncio.Semaphore(1)
    result = await extract_one_text(client, _item(), "Strong fit. VP at Series B.", sem)
    assert result.success is True
    assert result.extractor_complete is True
    assert result.output == out


@pytest.mark.asyncio
async def test_extractor_empty_chat_text_short_circuits():
    client = SimpleNamespace(
        messages=SimpleNamespace(create=AsyncMock())
    )
    sem = asyncio.Semaphore(1)
    result = await extract_one_text(client, _item(), None, sem)
    assert result.success is False
    assert result.output is None
    client.messages.create.assert_not_called()


def test_refusal_hints_contains_expected_phrases():
    # Lock the canonical refusal-hint list so the multi-turn loop's stop
    # behaviour doesn't drift silently.
    assert "i can't help" in REFUSAL_HINTS
    assert "i refuse" in REFUSAL_HINTS


# ----- Tool call returned but output is incomplete --------------------------


@pytest.mark.asyncio
async def test_extractor_tool_use_but_incomplete_output():
    """A tool call with missing classification fields must come back with
    extractor_complete=False so the multi-turn loop keeps going."""
    partial = _good_output()
    partial["classification"]["industry"] = ""  # missing field
    client = SimpleNamespace(
        messages=SimpleNamespace(create=AsyncMock(return_value=_mock_response_tool(partial)))
    )
    sem = asyncio.Semaphore(1)
    result = await extract_one_text(client, _item(), "some chat text", sem)
    assert result.success is True
    assert result.extractor_complete is False
    assert result.output == partial


@pytest.mark.asyncio
async def test_extractor_records_token_usage_on_tool_call():
    client = SimpleNamespace(
        messages=SimpleNamespace(create=AsyncMock(return_value=_mock_response_tool(_good_output())))
    )
    sem = asyncio.Semaphore(1)
    result = await extract_one_text(client, _item(), "chat text", sem)
    assert result.input_tokens == 15
    assert result.output_tokens == 12


@pytest.mark.asyncio
async def test_extractor_handles_anthropic_exception():
    client = SimpleNamespace(
        messages=SimpleNamespace(create=AsyncMock(side_effect=RuntimeError("boom")))
    )
    sem = asyncio.Semaphore(1)
    result = await extract_one_text(client, _item(), "chat text", sem)
    assert result.success is False
    assert "RuntimeError" in (result.error or "")


# ----- make_completeness_check -----------------------------------------------


@pytest.mark.asyncio
async def test_make_completeness_check_returns_bool_from_extractor():
    client = SimpleNamespace(
        messages=SimpleNamespace(create=AsyncMock(return_value=_mock_response_tool(_good_output())))
    )
    check = make_completeness_check(client, asyncio.Semaphore(1))
    assert await check(_item(), "complete chat text") is True


@pytest.mark.asyncio
async def test_make_completeness_check_false_on_incomplete():
    out = _good_output()
    out["classification"]["seniority"] = ""  # break completeness
    client = SimpleNamespace(
        messages=SimpleNamespace(create=AsyncMock(return_value=_mock_response_tool(out)))
    )
    check = make_completeness_check(client, asyncio.Semaphore(1))
    assert await check(_item(), "partial") is False


# ----- extract_chat_outputs preserves order and fires on_done --------------


def _chat_inference_result(item_id: str, text: str | None = None, error: str | None = None) -> InferenceResult:
    return InferenceResult(
        item_id=item_id,
        success=error is None,
        output=None,
        text=text,
        latency_ms=10,
        input_tokens=5,
        output_tokens=5,
        thinking_tokens=None,
        cache_read_tokens=0,
        error=error,
        raw_stop_reason="end_turn",
    )


def _item_with_id(item_id: str) -> EvalItem:
    return EvalItem(
        id=item_id, kind="exemplar", scenario="t", label="t",
        profile=f"profile {item_id}: VP Product", company=None,
        gold={"input_lang": "en", "expected_action": "auto_add"},
    )


@pytest.mark.asyncio
async def test_extract_chat_outputs_preserves_item_order_and_fires_on_done():
    items = [_item_with_id(str(i)) for i in range(3)]
    chat_results = [_chat_inference_result(str(i), text=f"chat {i}") for i in range(3)]
    client = SimpleNamespace(
        messages=SimpleNamespace(create=AsyncMock(return_value=_mock_response_tool(_good_output())))
    )
    fired: list[str] = []

    results = await extract_chat_outputs(
        items, chat_results, client=client, on_done=lambda r: fired.append(r.item_id)
    )
    assert [r.item_id for r in results] == ["0", "1", "2"]
    assert sorted(fired) == ["0", "1", "2"]


@pytest.mark.asyncio
async def test_extract_chat_outputs_skips_anthropic_call_when_chat_failed():
    """Failed chat results have no text — the extractor must short-circuit."""
    item = _item_with_id("1")
    failed = _chat_inference_result("1", error="chat exploded")
    client = SimpleNamespace(messages=SimpleNamespace(create=AsyncMock()))

    results = await extract_chat_outputs([item], [failed], client=client)
    assert results[0].success is False
    assert "chat exploded" in (results[0].error or "")
    client.messages.create.assert_not_called()
