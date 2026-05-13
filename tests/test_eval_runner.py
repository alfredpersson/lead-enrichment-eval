"""Tests for runner helpers: failure-mode clustering and progress counter."""

from __future__ import annotations

import pytest

from services.eval.metrics import PerItemScore
from services.eval.runner import _Progress, _failure_modes, _resolve_item_ids


def _passing_score(item_id: str = "ok") -> PerItemScore:
    return PerItemScore(
        item_id=item_id,
        kind="exemplar",
        success=True,
        classification_match={"industry": True, "segment": True, "seniority": True, "company_size": True},
        classification_overall=True,
        action_predicted="auto_add",
        action_gold="auto_add",
        action_correct=True,
        claim_count=3,
        substring_grounded_count=3,
        substring_grounded_rate=1.0,
        adversarial_pass=None,
    )


def test_failure_modes_skips_passing_items():
    scores = [_passing_score()]
    assert _failure_modes(scores) == []


def test_failure_modes_flags_action_miss():
    s = _passing_score("12")
    s.action_predicted = "propose"
    s.action_correct = False
    out = _failure_modes([s])
    assert len(out) == 1
    assert "action propose, expected auto_add" in out[0]["reasons"]


def test_failure_modes_flags_classification_misses():
    s = _passing_score("12")
    s.classification_match = {
        "industry": False,
        "segment": True,
        "seniority": True,
        "company_size": False,
    }
    s.classification_overall = False
    out = _failure_modes([s])
    reason = out[0]["reasons"][0]
    assert "classification miss" in reason
    assert "industry" in reason and "company_size" in reason


def test_failure_modes_flags_ungrounded_claims():
    s = _passing_score("12")
    s.claim_count = 4
    s.substring_grounded_count = 1
    s.substring_grounded_rate = 0.25
    out = _failure_modes([s])
    assert "3/4 claim source quote(s) not in input" in out[0]["reasons"]


def test_failure_modes_flags_adversarial_failures():
    s = _passing_score("4")
    s.adversarial_pass = False
    s.adversarial_failures = ["output echoes injection token"]
    out = _failure_modes([s])
    assert "output echoes injection token" in out[0]["reasons"]


def test_failure_modes_flags_inference_failure_first():
    s = _passing_score("12")
    s.success = False
    out = _failure_modes([s])
    assert "inference failed" in out[0]["reasons"]


def test_progress_tick_prints_at_completion(capsys):
    p = _Progress("test", total=3, interval=10.0)  # high interval so only final prints
    p.tick()
    p.tick()
    captured = capsys.readouterr().out
    assert captured == ""  # debounced
    p.tick()  # third tick = completion, must print
    captured = capsys.readouterr().out
    assert "[eval][test] 3/3" in captured


def test_resolve_item_ids_comma_list():
    assert _resolve_item_ids("1,4,13") == {"1", "4", "13"}


def test_resolve_item_ids_skips_blanks_and_whitespace():
    assert _resolve_item_ids("1, , 4, 13 ,") == {"1", "4", "13"}


def test_resolve_item_ids_from_file(tmp_path):
    p = tmp_path / "anchors.txt"
    p.write_text(
        "# header comment\n"
        "1   # first anchor\n"
        "\n"
        "4\n"
        "   13   # trailing comment\n"
    )
    assert _resolve_item_ids(f"@{p}") == {"1", "4", "13"}


def test_resolve_item_ids_empty_raises():
    with pytest.raises(SystemExit):
        _resolve_item_ids("")
    with pytest.raises(SystemExit):
        _resolve_item_ids(",,,")
