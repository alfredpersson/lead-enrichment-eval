"""Tests for the exemplar-snapshot loader and fingerprint helpers."""

from __future__ import annotations

import json

from services import snapshots
from services.snapshots import (
    chat_fingerprint,
    integrated_fingerprint,
    load_chat_snapshot,
    load_integrated_snapshot,
    normalize_message,
)


def test_normalize_message_collapses_whitespace_and_lowercases():
    assert normalize_message("  Qualify   against\nthe ICP  ") == "qualify against the icp"


def test_normalize_message_empty():
    assert normalize_message("") == ""


def test_integrated_fingerprint_shape():
    fp = integrated_fingerprint()
    assert set(fp.keys()) == {"system_prompt_sha256", "tool_schema_sha256"}


def test_chat_fingerprint_shape():
    fp = chat_fingerprint()
    assert set(fp.keys()) == {"system_prompt_sha256"}


def test_load_integrated_snapshot_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(snapshots, "INTEGRATED_DIR", tmp_path)
    assert load_integrated_snapshot("does-not-exist") is None


def test_load_integrated_snapshot_hit(monkeypatch, tmp_path):
    monkeypatch.setattr(snapshots, "INTEGRATED_DIR", tmp_path)
    payload = {"output": {"action": "auto_add"}, "model": "x"}
    (tmp_path / "42.json").write_text(json.dumps(payload))
    loaded = load_integrated_snapshot("42")
    assert loaded == payload


def test_load_chat_snapshot_matches_normalized_starter(monkeypatch, tmp_path):
    monkeypatch.setattr(snapshots, "CHAT_DIR", tmp_path)
    payload = {
        "user_message": "Qualify against the ICP",
        "assistant_text": "Strong fit.",
        "model": "x",
    }
    (tmp_path / "1-0.json").write_text(json.dumps(payload))
    # Whitespace + case variants should match.
    assert load_chat_snapshot("1", "qualify   against the ICP") == payload
    assert load_chat_snapshot("1", "QUALIFY AGAINST THE ICP") == payload


def test_load_chat_snapshot_no_match(monkeypatch, tmp_path):
    monkeypatch.setattr(snapshots, "CHAT_DIR", tmp_path)
    (tmp_path / "1-0.json").write_text(
        json.dumps({"user_message": "Qualify", "assistant_text": "x", "model": "y"})
    )
    assert load_chat_snapshot("1", "totally different starter") is None


def test_load_chat_snapshot_no_files(monkeypatch, tmp_path):
    monkeypatch.setattr(snapshots, "CHAT_DIR", tmp_path)
    assert load_chat_snapshot("1", "anything") is None


def test_load_chat_snapshot_filters_by_example_id(monkeypatch, tmp_path):
    monkeypatch.setattr(snapshots, "CHAT_DIR", tmp_path)
    # Two snapshots for different exemplars with the same starter text.
    (tmp_path / "1-0.json").write_text(
        json.dumps({"user_message": "Same", "assistant_text": "from one", "model": "x"})
    )
    (tmp_path / "2-0.json").write_text(
        json.dumps({"user_message": "Same", "assistant_text": "from two", "model": "x"})
    )
    one = load_chat_snapshot("1", "Same")
    assert one is not None
    assert one["assistant_text"] == "from one"
    two = load_chat_snapshot("2", "Same")
    assert two is not None
    assert two["assistant_text"] == "from two"
