"""
Snapshot loader for exemplar replays.

The integrated and chat endpoints can serve precomputed JSON for the five
exemplars instead of calling Anthropic on every prospect click. Layout:

    data/exemplar_snapshots/integrated/{exemplar_id}.json
    data/exemplar_snapshots/chat/{exemplar_id}-{starter_index}.json

Each snapshot stores a fingerprint of (model, system_prompt, tool_schema). On
import this module logs a warning for any snapshot whose fingerprint no longer
matches current code, so a stale snapshot is visible in Modal logs but still
served (regeneration is manual via scripts/regenerate_exemplar_snapshots.py).
"""

from __future__ import annotations

import hashlib
import json
import logging
import pathlib
from typing import Any

from services.prompts import (
    CHAT_SYSTEM_PROMPT,
    ENRICH_LEAD_TOOL,
    INTEGRATED_SYSTEM_PROMPT,
)

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
SNAPSHOTS_DIR = REPO_ROOT / "data" / "exemplar_snapshots"
INTEGRATED_DIR = SNAPSHOTS_DIR / "integrated"
CHAT_DIR = SNAPSHOTS_DIR / "chat"

logger = logging.getLogger(__name__)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def integrated_fingerprint() -> dict[str, str]:
    return {
        "system_prompt_sha256": _sha(INTEGRATED_SYSTEM_PROMPT),
        "tool_schema_sha256": _sha(json.dumps(ENRICH_LEAD_TOOL, sort_keys=True)),
    }


def chat_fingerprint() -> dict[str, str]:
    return {"system_prompt_sha256": _sha(CHAT_SYSTEM_PROMPT)}


def normalize_message(text: str) -> str:
    return " ".join(text.split()).lower()


def load_integrated_snapshot(example_id: str) -> dict[str, Any] | None:
    path = INTEGRATED_DIR / f"{example_id}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def load_chat_snapshot(example_id: str, user_message: str) -> dict[str, Any] | None:
    target = normalize_message(user_message)
    for path in sorted(CHAT_DIR.glob(f"{example_id}-*.json")):
        data = json.loads(path.read_text())
        if normalize_message(data["user_message"]) == target:
            return data
    return None


def _warn_on_drift() -> None:
    current = integrated_fingerprint()
    for path in sorted(INTEGRATED_DIR.glob("*.json")):
        data = json.loads(path.read_text())
        if (
            data["system_prompt_sha256"] != current["system_prompt_sha256"]
            or data["tool_schema_sha256"] != current["tool_schema_sha256"]
        ):
            logger.warning("Stale integrated snapshot %s; re-run regen script.", path.name)

    current = chat_fingerprint()
    for path in sorted(CHAT_DIR.glob("*.json")):
        data = json.loads(path.read_text())
        if data["system_prompt_sha256"] != current["system_prompt_sha256"]:
            logger.warning("Stale chat snapshot %s; re-run regen script.", path.name)


_warn_on_drift()
