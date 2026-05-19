"""Tests for the LLM-as-judge helpers and substring-aware kappa filter."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from services.eval.dataset import EvalItem
from services.eval.judges import (
    HookJudgement,
    HookResults,
    _cohen_kappa,
    _hook_judge_one,
    _openai_judge_claim,
    _opus_judge_claim,
    _parse_hook_payload,
    _parse_json_payload,
    judge_grounding,
    judge_hooks,
)


def _item() -> EvalItem:
    return EvalItem(
        id="1",
        kind="exemplar",
        scenario="t",
        label="t",
        profile="Maya Chen is VP Product at Lattice Forge.",
        company=None,
        gold={"input_lang": "en", "expected_action": "auto_add"},
    )


def _mock_text(payload_text: str) -> SimpleNamespace:
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=payload_text)],
        usage=SimpleNamespace(input_tokens=10, output_tokens=5, cache_read_input_tokens=0),
    )


def _openai_response(content: str) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )


class _StubAsyncCtx:
    """Stand-in for the AsyncAnthropic / AsyncOpenAI clients in judge_grounding —
    only its `close()` is awaited; all real calls are monkey-patched away."""

    async def close(self):
        pass


def test_cohen_kappa_perfect_agreement():
    assert _cohen_kappa([True, True, False, False], [True, True, False, False]) == 1.0


def test_cohen_kappa_chance_agreement():
    # 50/50 split, half agree by chance — kappa near 0.
    a = [True, False, True, False]
    b = [True, True, False, False]
    k = _cohen_kappa(a, b)
    assert k is not None
    assert abs(k) < 0.01


def test_cohen_kappa_returns_one_when_both_constant_and_equal():
    assert _cohen_kappa([True, True, True], [True, True, True]) == 1.0


def test_cohen_kappa_zero_when_constant_disagreement():
    assert _cohen_kappa([True, True, True], [False, False, False]) == 0.0


def test_cohen_kappa_empty():
    assert _cohen_kappa([], []) is None


def test_parse_json_payload_handles_prose_wrapper():
    text = "Here is my response: {\"grounded\": true, \"reason\": \"matches\"}"
    payload = _parse_json_payload(text)
    assert payload is not None
    assert payload["grounded"] is True


def test_parse_json_payload_returns_none_on_garbage():
    assert _parse_json_payload("nothing to see") is None
    assert _parse_json_payload("") is None
    assert _parse_json_payload("{unclosed") is None


def test_parse_hook_payload_pass_with_critique():
    j = _parse_hook_payload(
        "1",
        {"passes": True, "critique": "names the AI feature shipped"},
    )
    assert isinstance(j, HookJudgement)
    assert j.passes is True
    assert "AI feature" in (j.critique or "")


def test_parse_hook_payload_fail_with_critique():
    j = _parse_hook_payload("1", {"passes": False, "critique": "generic"})
    assert j.passes is False
    assert j.critique == "generic"


def test_parse_hook_payload_rejects_non_boolean_passes():
    j = _parse_hook_payload("1", {"passes": "yes", "critique": "..."})
    assert j.passes is None
    assert "boolean" in (j.critique or "")


def test_parse_hook_payload_rejects_missing_passes():
    j = _parse_hook_payload("1", {"critique": "..."})
    assert j.passes is None


def test_parse_hook_payload_handles_none_payload():
    j = _parse_hook_payload("1", None)
    assert j.passes is None
    assert "non-JSON" in (j.critique or "")


async def test_headline_rate_ignores_unconfigured_openai_judge(monkeypatch):
    """When OPENAI_API_KEY is unset, the OpenAI judge must not contribute to
    openai_rate or headline_rate — including on substring-failing claims.
    Regression for the bug where substring failures short-circuited to
    openai_grounded=False before the no-client check, dragging
    headline_rate to 0.0 even though the Opus judge was healthy.
    """

    async def fake_opus(client, item, claim, sem):
        return True, "ok"

    monkeypatch.setattr(
        "services.eval.judges._opus_judge_claim", fake_opus
    )

    # Override profile so the first claim's source_quote is a verbatim substring.
    item = replace(_item(), profile="Maya is VP Product.")
    outputs = {
        "1": {
            "claims": [
                {"text": "Maya is VP Product.", "source_quote": "Maya is VP Product."},
                {"text": "Maya is CTO.", "source_quote": "fabricated quote"},
            ]
        }
    }
    result = await judge_grounding(
        {"1": item},
        outputs,
        anthropic_client=_StubAsyncCtx(),
        openai_client=None,
    )
    assert result.opus_rate == pytest.approx(0.5)  # 1 grounded / 2 total
    assert result.openai_rate is None, (
        "openai_rate must be None when the OpenAI judge wasn't configured; "
        "substring-fail claims must not populate openai_grounded as False"
    )
    assert result.headline_rate == pytest.approx(result.opus_rate)
    # Both judgements must carry openai_grounded=None, not False.
    assert all(j.openai_grounded is None for j in result.judgements)


# ----- _opus_judge_claim direct ---------------------------------------------


@pytest.mark.asyncio
async def test_opus_judge_claim_happy_grounded():
    client = SimpleNamespace(
        messages=SimpleNamespace(
            create=AsyncMock(return_value=_mock_text('{"grounded": true, "reason": "matches"}'))
        )
    )
    sem = asyncio.Semaphore(1)
    grounded, reason = await _opus_judge_claim(
        client, _item(), {"text": "VP Product", "source_quote": "VP Product"}, sem
    )
    assert grounded is True
    assert "matches" in (reason or "")


@pytest.mark.asyncio
async def test_opus_judge_claim_non_json_returns_none():
    client = SimpleNamespace(
        messages=SimpleNamespace(create=AsyncMock(return_value=_mock_text("not json")))
    )
    sem = asyncio.Semaphore(1)
    grounded, reason = await _opus_judge_claim(
        client, _item(), {"text": "t", "source_quote": "q"}, sem
    )
    assert grounded is None
    assert "non-JSON" in (reason or "")


@pytest.mark.asyncio
async def test_opus_judge_claim_handles_exception():
    client = SimpleNamespace(
        messages=SimpleNamespace(create=AsyncMock(side_effect=RuntimeError("boom")))
    )
    sem = asyncio.Semaphore(1)
    grounded, reason = await _opus_judge_claim(
        client, _item(), {"text": "t", "source_quote": "q"}, sem
    )
    assert grounded is None
    assert "RuntimeError" in (reason or "")


# ----- _openai_judge_claim --------------------------------------------------


@pytest.mark.asyncio
async def test_openai_judge_claim_happy():
    completions = SimpleNamespace(
        create=AsyncMock(return_value=_openai_response('{"grounded": false, "reason": "no"}'))
    )
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    sem = asyncio.Semaphore(1)
    grounded, reason = await _openai_judge_claim(
        client, "gpt-x", _item(), {"text": "t", "source_quote": "q"}, sem
    )
    assert grounded is False
    assert reason == "no"


@pytest.mark.asyncio
async def test_openai_judge_claim_exception_returns_none():
    completions = SimpleNamespace(create=AsyncMock(side_effect=RuntimeError("503")))
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    sem = asyncio.Semaphore(1)
    grounded, reason = await _openai_judge_claim(
        client, "gpt-x", _item(), {"text": "t", "source_quote": "q"}, sem
    )
    assert grounded is None
    assert "RuntimeError" in (reason or "")


# ----- judge_grounding with both clients ------------------------------------


@pytest.mark.asyncio
async def test_judge_grounding_substring_fail_short_circuits_judges(monkeypatch):
    """Substring-failing claims must not call either judge."""
    opus_calls = 0
    openai_calls = 0

    async def fake_opus(client, item, claim, sem):
        nonlocal opus_calls
        opus_calls += 1
        return True, "ok"

    async def fake_openai(client, model_id, item, claim, sem):
        nonlocal openai_calls
        openai_calls += 1
        return True, "ok"

    monkeypatch.setattr("services.eval.judges._opus_judge_claim", fake_opus)
    monkeypatch.setattr("services.eval.judges._openai_judge_claim", fake_openai)

    item = _item()
    outputs = {
        "1": {
            "claims": [
                # Substring matches input.
                {"text": "VP", "source_quote": "VP Product"},
                # Does not match — should bypass both judges.
                {"text": "Bogus", "source_quote": "not in the input"},
            ]
        }
    }
    result = await judge_grounding(
        {"1": item},
        outputs,
        anthropic_client=_StubAsyncCtx(),
        openai_client=_StubAsyncCtx(),
    )
    # Both judges called exactly once — only for the substring-pass claim.
    assert opus_calls == 1
    assert openai_calls == 1
    # Substring-fail claim is False/False (auto-fail).
    fail_j = next(j for j in result.judgements if not j.substring_match)
    assert fail_j.opus_grounded is False
    assert fail_j.openai_grounded is False


@pytest.mark.asyncio
async def test_judge_grounding_kappa_excludes_substring_fail_claims(monkeypatch):
    """The substring-fail auto-False/auto-False would inflate kappa with a
    trivial agreement. The filter must drop it. With this disagreement
    pattern, including-vs-excluding the auto-False claim gives kappa=0.0
    vs kappa=0.4 — so asserting 0.0 proves the filter applies."""

    async def fake_opus(client, item, claim, sem):
        # Disagree with OpenAI on claim "B": opus says False, openai says True.
        return claim["text"] != "B", "ok"

    async def fake_openai(client, model_id, item, claim, sem):
        return True, "ok"

    monkeypatch.setattr("services.eval.judges._opus_judge_claim", fake_opus)
    monkeypatch.setattr("services.eval.judges._openai_judge_claim", fake_openai)

    outputs = {
        "1": {
            "claims": [
                {"text": "A", "source_quote": "Maya Chen"},  # substring pass, T/T
                {"text": "B", "source_quote": "VP Product"},  # substring pass, F/T (disagree)
                {"text": "C", "source_quote": "not in input"},  # substring fail, auto F/F
            ]
        }
    }
    result = await judge_grounding(
        {"1": _item()},
        outputs,
        anthropic_client=_StubAsyncCtx(),
        openai_client=_StubAsyncCtx(),
    )
    # Filter keeps claims A & B → opus=[T,F], openai=[T,T] → kappa = 0.0
    # Without filter on A,B,C → opus=[T,F,F], openai=[T,T,F] → kappa = 0.4
    assert result.kappa == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_judge_grounding_empty_outputs_returns_empty_result():
    result = await judge_grounding({}, {}, anthropic_client=_StubAsyncCtx(), openai_client=None)
    assert result.n_claims == 0
    assert result.judgements == []
    assert result.opus_rate is None


# ----- _hook_judge_one + judge_hooks ---------------------------------------


@pytest.mark.asyncio
async def test_hook_judge_one_returns_pass():
    completions = SimpleNamespace(
        create=AsyncMock(
            return_value=_openai_response('{"passes": true, "critique": "names AI feature"}')
        )
    )
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    sem = asyncio.Semaphore(1)
    j = await _hook_judge_one(
        client,
        "gpt-x",
        _item(),
        {"draft_hook": {"text": "Reach out about your shipped AI feature."}, "action": "auto_add"},
        sem,
    )
    assert j.passes is True
    assert "AI feature" in (j.critique or "")


@pytest.mark.asyncio
async def test_hook_judge_one_short_circuits_on_empty_hook():
    """No model call when hook text is empty."""
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=AsyncMock())))
    sem = asyncio.Semaphore(1)
    j = await _hook_judge_one(
        client, "gpt-x", _item(), {"draft_hook": {"text": ""}, "action": "discard"}, sem
    )
    assert j.passes is None
    assert "empty hook" in (j.critique or "")
    client.chat.completions.create.assert_not_called()


@pytest.mark.asyncio
async def test_hook_judge_one_handles_string_hook():
    completions = SimpleNamespace(
        create=AsyncMock(
            return_value=_openai_response('{"passes": false, "critique": "generic"}')
        )
    )
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    sem = asyncio.Semaphore(1)
    j = await _hook_judge_one(
        client, "gpt-x", _item(), {"draft_hook": "raw hook text", "action": "auto_add"}, sem
    )
    assert j.passes is False
    assert j.critique == "generic"


@pytest.mark.asyncio
async def test_judge_hooks_returns_empty_when_no_openai():
    result = await judge_hooks({}, {}, openai_client=None)
    assert isinstance(result, HookResults)
    assert result.n_scored == 0
    assert result.pass_rate is None


@pytest.mark.asyncio
async def test_judge_hooks_computes_pass_rate():
    payloads = [
        _openai_response('{"passes": true, "critique": "good"}'),
        _openai_response('{"passes": false, "critique": "bad"}'),
        _openai_response('{"passes": true, "critique": "also good"}'),
    ]
    completions = SimpleNamespace(create=AsyncMock(side_effect=payloads))
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))

    items = {
        str(i): EvalItem(
            id=str(i), kind="exemplar", scenario="t", label="t",
            profile="p", company=None, gold={"input_lang": "en", "expected_action": "auto_add"},
        )
        for i in range(3)
    }
    outputs = {
        str(i): {"draft_hook": {"text": "non-empty"}, "action": "auto_add"}
        for i in range(3)
    }
    result = await judge_hooks(items, outputs, openai_client=client)
    assert result.n_scored == 3
    assert result.pass_rate == pytest.approx(2 / 3)


@pytest.mark.asyncio
async def test_judge_hooks_skips_none_outputs():
    completions = SimpleNamespace(
        create=AsyncMock(return_value=_openai_response('{"passes": true, "critique": "ok"}'))
    )
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    items = {"1": _item()}
    result = await judge_hooks(items, {"1": None}, openai_client=client)
    assert result.n_scored == 0
    client.chat.completions.create.assert_not_called()
