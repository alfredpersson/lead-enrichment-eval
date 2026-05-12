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
    extract_one_text,
)


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
