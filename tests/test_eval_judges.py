"""Tests for the LLM-as-judge helpers and substring-aware kappa filter."""

from __future__ import annotations

import pytest

from services.eval.dataset import EvalItem
from services.eval.judges import (
    ClaimJudgement,
    HookJudgement,
    _cohen_kappa,
    _parse_hook_payload,
    _parse_json_payload,
)


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


def test_substring_filter_excludes_failures_from_kappa(monkeypatch):
    """Re-test the load-bearing kappa-filter fix via the data structure that
    feeds into the filter at the end of judge_grounding."""
    judgements = [
        ClaimJudgement(
            item_id="1",
            claim_index=0,
            claim_text="t",
            source_quote="q",
            substring_match=False,
            opus_grounded=False,
            openai_grounded=False,
        ),
        ClaimJudgement(
            item_id="2",
            claim_index=0,
            claim_text="t",
            source_quote="q",
            substring_match=True,
            opus_grounded=True,
            openai_grounded=False,
        ),
        ClaimJudgement(
            item_id="3",
            claim_index=0,
            claim_text="t",
            source_quote="q",
            substring_match=True,
            opus_grounded=True,
            openai_grounded=True,
        ),
    ]
    paired = [
        j for j in judgements
        if j.substring_match
        and j.opus_grounded is not None
        and j.openai_grounded is not None
    ]
    assert len(paired) == 2  # only substring-pass claims count
    # Without the filter, the substring failure would inflate kappa with a
    # trivial agreement on False/False.
    naive_paired = [
        j for j in judgements
        if j.opus_grounded is not None and j.openai_grounded is not None
    ]
    assert len(naive_paired) == 3
    assert _cohen_kappa(
        [bool(j.opus_grounded) for j in paired],
        [bool(j.openai_grounded) for j in paired],
    ) != _cohen_kappa(
        [bool(j.opus_grounded) for j in naive_paired],
        [bool(j.openai_grounded) for j in naive_paired],
    )
