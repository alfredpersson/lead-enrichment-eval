"""End-to-end integrity checks on `data/exemplars.json`.

These walk the real dataset and verify that the invariants the rest of the
suite assumes hold up: every gold source quote is a substring of its input,
every action is in the canonical enum, every classification value is in the
schema enum, fit scores are in range, and adversarial check declarations
are well-formed.
"""

from __future__ import annotations

import pytest

from services.eval.dataset import load_dataset
from services.eval.metrics import ACTION_VALUES, _normalise
from services.prompts import ENRICH_LEAD_TOOL

SENIORITY_ENUM = set(
    ENRICH_LEAD_TOOL["input_schema"]["properties"]["classification"]["properties"]["seniority"]["enum"]
)
COMPANY_SIZE_ENUM = set(
    ENRICH_LEAD_TOOL["input_schema"]["properties"]["classification"]["properties"]["company_size"]["enum"]
)

SUPPORTED_ADVERSARIAL_KINDS = {
    "no_token_in_output",
    "no_quote_contains",
    "max_fit_score",
    "action_in",
    "action_is",
    "all_claims_substring_grounded",
}

_, ITEMS = load_dataset()
ITEM_IDS = [it.id for it in ITEMS]


@pytest.mark.parametrize("item", ITEMS, ids=ITEM_IDS)
def test_every_gold_source_quote_is_a_substring_of_input(item):
    """The substring-grounding metric depends on this — if gold quotes
    don't actually appear in the input, score_item flags every claim as
    ungrounded and the metric is useless."""
    norm_input = _normalise(item.input_text)
    for claim in item.gold.get("claims_allowed") or []:
        quote = claim.get("source_quote") or ""
        if not quote:
            continue
        assert _normalise(quote) in norm_input, (
            f"gold source_quote not found in input for item {item.id}: {quote!r}"
        )


@pytest.mark.parametrize("item", ITEMS, ids=ITEM_IDS)
def test_expected_action_is_in_canonical_enum(item):
    action = item.gold.get("expected_action")
    assert action in ACTION_VALUES, (
        f"item {item.id} expected_action {action!r} not in {ACTION_VALUES}"
    )


@pytest.mark.parametrize("item", ITEMS, ids=ITEM_IDS)
def test_classification_enum_bound_fields_in_schema(item):
    """Only seniority and company_size are enum-bound for scoring (industry
    and segment are scored via token-overlap in metrics.score_item). Gold may
    use any industry phrasing — the model is constrained to the enum on
    output, not the gold author."""
    cls = item.gold.get("classification") or {}
    if not cls:
        return
    seniority = cls.get("seniority")
    if seniority:
        assert seniority in SENIORITY_ENUM, (
            f"item {item.id} seniority {seniority!r} not in {SENIORITY_ENUM}"
        )
    company_size = cls.get("company_size")
    if company_size:
        assert company_size in COMPANY_SIZE_ENUM, (
            f"item {item.id} company_size {company_size!r} not in {COMPANY_SIZE_ENUM}"
        )


@pytest.mark.parametrize("item", ITEMS, ids=ITEM_IDS)
def test_fit_score_in_range(item):
    fs = item.gold.get("fit_score") or {}
    value = fs.get("value")
    if value is None:
        return
    assert 0.0 <= float(value) <= 1.0, f"item {item.id} fit_score.value {value} out of range"
    for dim, v in (fs.get("dimensions") or {}).items():
        assert 0.0 <= float(v) <= 1.0, (
            f"item {item.id} dim {dim} value {v} out of range"
        )


@pytest.mark.parametrize("item", ITEMS, ids=ITEM_IDS)
def test_adversarial_checks_well_formed(item):
    checks = item.gold.get("adversarial_pass_checks") or []
    for i, check in enumerate(checks):
        assert "kind" in check, f"item {item.id} adversarial check #{i} missing 'kind'"
        # Unknown kinds are forward-compatible per metrics._adversarial_checks,
        # so we only warn here by allowing them — but we lock the current set.
        # If a new kind is added intentionally, extend SUPPORTED_ADVERSARIAL_KINDS.
        assert check["kind"] in SUPPORTED_ADVERSARIAL_KINDS, (
            f"item {item.id} declares unknown adversarial kind {check['kind']!r}; "
            f"either add it to metrics._adversarial_checks or remove it"
        )


def test_dataset_non_empty():
    assert len(ITEMS) > 0


def test_item_ids_are_unique():
    ids = [it.id for it in ITEMS]
    assert len(ids) == len(set(ids)), "duplicate item ids in exemplars.json"
