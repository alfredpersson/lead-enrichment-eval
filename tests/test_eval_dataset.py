"""Tests for the eval dataset loader and EvalItem accessor properties."""

from __future__ import annotations

import json

from services.eval.dataset import EvalItem, iter_by_kind, load_dataset


def test_load_dataset_returns_version_and_items(tmp_path):
    payload = {
        "version": "test-1.0",
        "items": [
            {
                "id": "1",
                "kind": "exemplar",
                "scenario": "strong_fit",
                "label": "Strong fit",
                "profile": "Maya is VP Product.",
                "company": "Lattice Forge.",
                "gold": {"expected_action": "auto_add"},
            },
            {
                "id": "2",
                "kind": "adversarial",
                "scenario": "injection",
                "label": "Injection probe",
                "profile": "Adversarial profile.",
                "company": None,
                "gold": {"expected_action": "refuse"},
            },
        ],
    }
    path = tmp_path / "ds.json"
    path.write_text(json.dumps(payload))

    version, items = load_dataset(path)
    assert version == "test-1.0"
    assert len(items) == 2
    assert items[0].id == "1"
    assert items[0].kind == "exemplar"
    assert items[1].kind == "adversarial"


def test_load_dataset_defaults_version_to_0a(tmp_path):
    payload = {"items": [{"id": "1", "profile": "p", "gold": {}}]}
    path = tmp_path / "ds.json"
    path.write_text(json.dumps(payload))
    version, items = load_dataset(path)
    assert version == "0a"
    assert items[0].kind == "exemplar"


def test_eval_item_input_text_with_company():
    item = EvalItem(
        id="1",
        kind="exemplar",
        scenario="t",
        label="t",
        profile="Profile text.",
        company="Company text.",
        gold={},
    )
    assert item.input_text == "Profile text.\n\nCompany text."


def test_eval_item_input_text_without_company():
    item = EvalItem(
        id="1",
        kind="exemplar",
        scenario="t",
        label="t",
        profile="Profile text.",
        company=None,
        gold={},
    )
    assert item.input_text == "Profile text."


def test_eval_item_expected_action_defaults_to_empty_string():
    item = EvalItem(
        id="1", kind="exemplar", scenario="t", label="t",
        profile="p", company=None, gold={},
    )
    assert item.expected_action == ""


def test_eval_item_expected_lang_defaults_to_en():
    item = EvalItem(
        id="1", kind="exemplar", scenario="t", label="t",
        profile="p", company=None, gold={},
    )
    assert item.expected_lang == "en"


def test_eval_item_adversarial_pass_checks_defaults_to_empty_list():
    item = EvalItem(
        id="1", kind="exemplar", scenario="t", label="t",
        profile="p", company=None, gold={},
    )
    assert item.adversarial_pass_checks == []
    assert item.adversarial_pass_criteria == []


def test_eval_item_adversarial_pass_checks_returns_payload():
    checks = [{"kind": "no_token_in_output", "token": "ACCEPTED"}]
    item = EvalItem(
        id="1", kind="adversarial", scenario="t", label="t",
        profile="p", company=None,
        gold={"adversarial_pass_checks": checks, "adversarial_pass_criteria": ["does not echo"]},
    )
    assert item.adversarial_pass_checks == checks
    assert item.adversarial_pass_criteria == ["does not echo"]


def test_iter_by_kind_filters():
    items = [
        EvalItem(id=str(i), kind=k, scenario="t", label="t", profile="p", company=None, gold={})
        for i, k in enumerate(["exemplar", "adversarial", "exemplar", "edge"])
    ]
    exemplars = list(iter_by_kind(items, "exemplar"))
    assert [it.id for it in exemplars] == ["0", "2"]
    adv = list(iter_by_kind(items, "adversarial"))
    assert [it.id for it in adv] == ["1"]
    none = list(iter_by_kind(items, "missing"))
    assert none == []
