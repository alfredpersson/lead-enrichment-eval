"""Tests for the integrated build (`services/enrich.py`)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from services import enrich
from services.enrich import _grounded_count, _user_message
from services.validation import ValidationError


def test_user_message_with_company():
    assert _user_message("profile text", "company text") == (
        "<profile>\nprofile text\n</profile>\n\n<company>\ncompany text\n</company>"
    )


def test_user_message_without_company():
    assert _user_message("profile text", None) == "<profile>\nprofile text\n</profile>"


def test_grounded_count_empty_claims():
    assert _grounded_count([], "any input") == 0


def test_grounded_count_substring_match():
    claims = [{"source_quote": "VP Product"}]
    assert _grounded_count(claims, "Maya is VP Product at Lattice") == 1


def test_grounded_count_case_insensitive():
    claims = [{"source_quote": "vp product"}]
    assert _grounded_count(claims, "Maya is VP Product at Lattice") == 1


def test_grounded_count_whitespace_normalised():
    claims = [{"source_quote": "VP   Product"}]
    assert _grounded_count(claims, "Maya is VP Product at Lattice") == 1


def test_grounded_count_skips_missing_quote():
    claims = [{"source_quote": ""}, {"source_quote": None}, {}]
    assert _grounded_count(claims, "any input") == 0


def test_grounded_count_mixed():
    claims = [
        {"source_quote": "VP Product"},  # match
        {"source_quote": "made up phrase"},  # no match
        {"source_quote": "Lattice Forge"},  # match
    ]
    assert _grounded_count(claims, "Maya is VP Product at Lattice Forge") == 2


# ----- enrich_lead end-to-end with mocked Anthropic ------------------------


def _good_tool_output() -> dict:
    return {
        "classification": {
            "industry": "B2B SaaS",
            "segment": "sales execution",
            "seniority": "VP",
            "company_size": "51-200",
        },
        "fit_score": {
            "value": 0.9,
            "dimensions": {
                "stage_match": 1.0,
                "headcount_match": 1.0,
                "arr_match": 1.0,
                "product_shape_match": 1.0,
                "role_match": 1.0,
            },
        },
        "claims": [
            {
                "text": "VP Product at Series B SaaS.",
                "source_quote": "VP Product at Lattice Forge",
                "confidence": 0.9,
            }
        ],
        "draft_hook": {"text": "Reach out", "claims_used": [], "confidence": 0.8},
        "action": "auto_add",
        "reasoning": "...",
    }


def _mock_tool_response(tool_input: dict):
    tool_block = SimpleNamespace(type="tool_use", input=tool_input, name="enrich_lead")
    usage = SimpleNamespace(
        input_tokens=120,
        output_tokens=80,
        cache_read_input_tokens=0,
        thinking_tokens=200,
    )
    return SimpleNamespace(content=[tool_block], usage=usage)


def _mock_text_only_response():
    text_block = SimpleNamespace(type="text", text="just text")
    usage = SimpleNamespace(input_tokens=10, output_tokens=4, cache_read_input_tokens=0)
    return SimpleNamespace(content=[text_block], usage=usage)


@pytest.fixture
def patch_external(monkeypatch):
    """Mock side-channels we don't want to exercise: snapshots, embeddings,
    neighbours, telemetry, validation language detector."""
    monkeypatch.setattr(enrich, "load_integrated_snapshot", lambda example_id: None)
    monkeypatch.setattr(enrich, "embed", AsyncMock(return_value=None))
    monkeypatch.setattr(enrich, "find_neighbours", AsyncMock(return_value=[]))
    monkeypatch.setattr(enrich, "write_request_row", AsyncMock(return_value=None))
    monkeypatch.setattr(enrich, "check_input", lambda profile, company: "en")
    monkeypatch.setattr(enrich, "is_local", lambda: True)


@pytest.mark.asyncio
async def test_enrich_lead_happy_path(monkeypatch, patch_external):
    tool_input = _good_tool_output()
    fake_client = SimpleNamespace(
        messages=SimpleNamespace(create=AsyncMock(return_value=_mock_tool_response(tool_input)))
    )
    monkeypatch.setattr(enrich, "_get_client", lambda: fake_client)

    output = await enrich.enrich_lead(
        "Maya is VP Product at Lattice Forge", "Lattice Forge Series B"
    )
    # Tool input is returned as the output, with meta merged in.
    for key in ("classification", "fit_score", "claims", "draft_hook", "action"):
        assert key in output
    assert output["action"] == "auto_add"
    assert "meta" in output
    assert output["meta"]["model"] == enrich.MODEL_ID
    assert output["meta"]["tokens_in"] == 120
    assert output["meta"]["tokens_out"] == 80
    assert output["meta"]["thinking_budget"] == enrich.THINKING_BUDGET_TOKENS
    assert "request_id" in output["meta"]
    assert output["meta"]["latency_ms"] >= 0


@pytest.mark.asyncio
async def test_enrich_lead_raises_when_no_tool_use(monkeypatch, patch_external):
    fake_client = SimpleNamespace(
        messages=SimpleNamespace(create=AsyncMock(return_value=_mock_text_only_response()))
    )
    monkeypatch.setattr(enrich, "_get_client", lambda: fake_client)
    with pytest.raises(RuntimeError, match="enrich_lead tool"):
        await enrich.enrich_lead("p", "c")


@pytest.mark.asyncio
async def test_enrich_lead_propagates_validation_error(monkeypatch, patch_external):
    def reject(profile, company):
        raise ValidationError(code="empty_profile", message="empty")

    monkeypatch.setattr(enrich, "check_input", reject)
    with pytest.raises(ValidationError) as exc:
        await enrich.enrich_lead("", None)
    assert exc.value.code == "empty_profile"


@pytest.mark.asyncio
async def test_enrich_lead_serves_snapshot_when_available(monkeypatch, patch_external):
    snap = {
        "model": "claude-sonnet-4-6",
        "output": _good_tool_output(),
        "thinking_trace": "thinking...",
        "usage": {
            "input_tokens": 1000,
            "output_tokens": 500,
            "thinking_tokens": 300,
            "latency_ms": 1234,
        },
    }
    monkeypatch.setattr(enrich, "load_integrated_snapshot", lambda example_id: snap)
    # Anthropic client must NOT be touched on a snapshot hit.
    fake_client = SimpleNamespace(messages=SimpleNamespace(create=AsyncMock()))
    monkeypatch.setattr(enrich, "_get_client", lambda: fake_client)

    output = await enrich.enrich_lead("Maya is VP Product", None, example_id="1")
    fake_client.messages.create.assert_not_called()
    assert output["meta"]["snapshot_served"] is True
    assert output["meta"]["latency_ms"] == 1234
    assert output["meta"]["tokens_in"] == 1000


@pytest.mark.asyncio
async def test_enrich_lead_bypass_cache_skips_snapshot_lookup(monkeypatch, patch_external):
    def fake_load(*_args, **_kwargs):
        raise AssertionError("load_integrated_snapshot must not be called when bypass_cache=True")

    monkeypatch.setattr(enrich, "load_integrated_snapshot", fake_load)
    fake_client = SimpleNamespace(
        messages=SimpleNamespace(create=AsyncMock(return_value=_mock_tool_response(_good_tool_output())))
    )
    monkeypatch.setattr(enrich, "_get_client", lambda: fake_client)

    await enrich.enrich_lead("p", None, example_id="1", bypass_cache=True)
    fake_client.messages.create.assert_awaited_once()


@pytest.mark.asyncio
async def test_enrich_lead_writes_telemetry(monkeypatch, patch_external):
    written: list[dict] = []

    async def capture(row):
        written.append(row)

    monkeypatch.setattr(enrich, "write_request_row", capture)
    fake_client = SimpleNamespace(
        messages=SimpleNamespace(create=AsyncMock(return_value=_mock_tool_response(_good_tool_output())))
    )
    monkeypatch.setattr(enrich, "_get_client", lambda: fake_client)

    await enrich.enrich_lead(
        "Maya is VP Product at Lattice Forge", "Lattice Forge Series B"
    )
    assert len(written) == 1
    row = written[0]
    assert row["mode"] == "integrated"
    assert row["thinking_enabled"] is True
    assert row["model_id"] == enrich.MODEL_ID
    assert row["action"] == "auto_add"
    assert row["claim_count"] == 1
    # Claim has source quote "VP Product at Lattice Forge" which IS in the input.
    assert row["claims_with_source_quote_count"] == 1
