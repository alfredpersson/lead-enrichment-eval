"""Tests for runner helpers: failure-mode clustering and progress counter."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from services.eval import runner
from services.eval.dataset import EvalItem
from services.eval.inference import ChatResult, InferenceResult
from services.eval.judges import (
    ClaimJudgement,
    GroundingResults,
    HookJudgement,
    HookResults,
)
from services.eval.metrics import PerItemScore
from services.eval.perturb import perturb_all
from services.eval.runner import (
    _Progress,
    _chat_meta,
    _failure_modes,
    _grounding_to_dict,
    _hooks_to_dict,
    _inference_meta,
    _resolve_item_ids,
    _robustness_block,
    _score_mode,
    _write_snapshot,
)


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


# ----- Snapshot meta dict shapes (frontend contract) -----------------------


def _inference_result(item_id: str = "x") -> InferenceResult:
    return InferenceResult(
        item_id=item_id,
        success=True,
        output={"action": "auto_add"},
        text=None,
        latency_ms=200,
        input_tokens=100,
        output_tokens=50,
        thinking_tokens=300,
        cache_read_tokens=10,
        error=None,
        raw_stop_reason="tool_use",
    )


def _chat_result(item_id: str = "x") -> ChatResult:
    return ChatResult(
        item_id=item_id,
        success=True,
        text="ok",
        transcript=[],
        latency_ms=300,
        input_tokens=120,
        output_tokens=80,
        cache_read_tokens=20,
        turns_used=2,
        cap_hit=False,
        error=None,
    )


def test_inference_meta_keys():
    meta = _inference_meta(_inference_result())
    assert set(meta.keys()) == {
        "item_id",
        "success",
        "error",
        "latency_ms",
        "input_tokens",
        "output_tokens",
        "thinking_tokens",
        "cache_read_tokens",
        "stop_reason",
    }
    assert meta["stop_reason"] == "tool_use"
    assert meta["thinking_tokens"] == 300


def test_chat_meta_keys():
    meta = _chat_meta(_chat_result())
    assert set(meta.keys()) == {
        "item_id",
        "success",
        "error",
        "latency_ms",
        "input_tokens",
        "output_tokens",
        "cache_read_tokens",
        "turns_used",
        "cap_hit",
    }
    assert meta["turns_used"] == 2
    assert meta["cap_hit"] is False


def test_grounding_to_dict_keys():
    g = GroundingResults(
        judgements=[
            ClaimJudgement(
                item_id="1",
                claim_index=0,
                claim_text="t",
                source_quote="q",
                substring_match=True,
                opus_grounded=True,
                openai_grounded=True,
            )
        ],
        opus_rate=1.0,
        openai_rate=1.0,
        headline_rate=1.0,
        kappa=1.0,
        n_claims=1,
        n_judged=1,
    )
    d = _grounding_to_dict(g)
    assert set(d.keys()) == {
        "n_claims",
        "n_judged_by_opus",
        "opus_grounding_rate",
        "openai_grounding_rate",
        "headline_grounding_rate",
        "cohen_kappa",
        "judgements",
    }
    assert d["headline_grounding_rate"] == 1.0
    assert isinstance(d["judgements"], list) and len(d["judgements"]) == 1


def test_hooks_to_dict_keys():
    h = HookResults(
        judgements=[HookJudgement(item_id="1", passes=True, critique="good")],
        pass_rate=1.0,
        n_scored=1,
    )
    d = _hooks_to_dict(h)
    assert set(d.keys()) == {"n_scored", "pass_rate", "judgements"}
    assert d["pass_rate"] == 1.0


# ----- _score_mode + _robustness_block --------------------------------------


def _item(item_id: str) -> EvalItem:
    return EvalItem(
        id=item_id,
        kind="exemplar",
        scenario="t",
        label="t",
        profile="Maya Chen is VP Product at Lattice Forge.",
        company=None,
        gold={
            "input_lang": "en",
            "expected_action": "auto_add",
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
        },
    )


def _output() -> dict:
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
                "text": "VP Product",
                "source_quote": "VP Product at Lattice Forge",
                "confidence": 0.9,
            }
        ],
        "draft_hook": {"text": "h", "claims_used": [], "confidence": 0.8},
        "action": "auto_add",
        "reasoning": "...",
    }


def test_score_mode_happy_path():
    items = [_item("1"), _item("2")]
    latencies = {"1": 100, "2": 200}
    tokens_in = {"1": 50, "2": 60}
    tokens_out = {"1": 20, "2": 30}
    outputs = {"1": _output(), "2": _output()}
    scores, agg = _score_mode(items, latencies, tokens_in, tokens_out, outputs)
    assert len(scores) == 2
    assert agg.n == 2
    assert agg.action_accuracy == 1.0


def test_score_mode_handles_missing_output():
    items = [_item("1"), _item("2")]
    outputs = {"1": _output()}  # Item 2 has no output (inference failed).
    scores, agg = _score_mode(
        items, {"1": 100}, {"1": 50}, {"1": 20}, outputs
    )
    assert scores[0].success is True
    assert scores[1].success is False
    # Aggregate only counts successful inferences.
    assert agg.n == 2


def _pass_results_for(perturbed):
    integrated = [
        InferenceResult(
            item_id=p.item.id, success=True, output=_output(), text=None,
            latency_ms=100, input_tokens=10, output_tokens=5,
            thinking_tokens=None, cache_read_tokens=0, error=None, raw_stop_reason=None,
        )
        for p in perturbed
    ]
    chat = [
        ChatResult(
            item_id=p.item.id, success=True, text="t", transcript=[],
            latency_ms=120, input_tokens=10, output_tokens=5, cache_read_tokens=0,
            turns_used=1, cap_hit=False, error=None,
        )
        for p in perturbed
    ]
    return {
        "integrated": integrated,
        "chat": chat,
        "extractor": [],
        "integrated_outputs": {r.item_id: r.output for r in integrated},
        "chat_outputs": {r.item_id: _output() for r in chat},
    }


def test_robustness_block_aggregates_three_variants():
    items = [_item("1"), _item("2")]
    perturbed = perturb_all(items)
    block = _robustness_block(perturbed, _pass_results_for(perturbed), include_per_item=False)
    assert set(block["by_variant"]) == {"typos", "sentence_reorder", "injection"}
    assert block["n_base_items"] == 2
    for v in block["by_variant"].values():
        assert v["n"] == len(items)
        assert "per_item_integrated" not in v


def test_robustness_block_with_per_item_includes_per_item_arrays():
    items = [_item("1")]
    perturbed = perturb_all(items)
    block = _robustness_block(perturbed, _pass_results_for(perturbed), include_per_item=True)
    for v in block["by_variant"].values():
        assert len(v["per_item_integrated"]) == 1
        assert len(v["per_item_chat"]) == 1


# ----- _write_snapshot round-trip ------------------------------------------


def test_write_snapshot_writes_dated_file_and_latest(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "SNAPSHOTS_DIR", tmp_path)
    completed = datetime(2026, 5, 19, 8, 30, 0, tzinfo=timezone.utc).isoformat()
    snap = {"schema_version": 3, "completed_at": completed}

    path = _write_snapshot(snap, tag="test")
    assert path.name == "20260519T083000Z-test.json"
    assert json.loads(path.read_text()) == snap
    assert json.loads((tmp_path / "latest.json").read_text()) == snap


def test_write_snapshot_no_tag_omits_suffix(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "SNAPSHOTS_DIR", tmp_path)
    completed = datetime(2026, 5, 19, 8, 30, 0, tzinfo=timezone.utc).isoformat()
    path = _write_snapshot({"completed_at": completed}, tag=None)
    assert path.name == "20260519T083000Z.json"
