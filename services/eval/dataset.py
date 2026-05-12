"""
Eval dataset loader.

Reads `data/exemplars.json` (the canonical Phase 0a+0b test set) and yields
typed records. The same file is the source for the Postgres `eval_set` table
via `services.eval_seed`; we re-read it here so the eval runner stays self-
contained and reproducible from the repo state alone.
"""

from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass
from typing import Any, Iterator

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
DATASET_PATH = REPO_ROOT / "data" / "exemplars.json"


@dataclass(frozen=True)
class EvalItem:
    id: str
    kind: str
    scenario: str
    label: str
    profile: str
    company: str | None
    gold: dict[str, Any]

    @property
    def input_text(self) -> str:
        return self.profile if not self.company else f"{self.profile}\n\n{self.company}"

    @property
    def expected_action(self) -> str:
        return self.gold.get("expected_action", "")

    @property
    def expected_lang(self) -> str:
        return self.gold.get("input_lang", "en")

    @property
    def adversarial_pass_criteria(self) -> list[str]:
        """Human-readable prose criteria. Used for methodology display."""
        return self.gold.get("adversarial_pass_criteria") or []

    @property
    def adversarial_pass_checks(self) -> list[dict[str, Any]]:
        """Typed criteria the metric pipeline dispatches on. See
        services.eval.metrics._adversarial_checks for supported `kind`s."""
        return self.gold.get("adversarial_pass_checks") or []


def load_dataset(path: pathlib.Path = DATASET_PATH) -> tuple[str, list[EvalItem]]:
    payload = json.loads(path.read_text())
    version = payload.get("version", "0a")
    items = [
        EvalItem(
            id=row["id"],
            kind=row.get("kind", "exemplar"),
            scenario=row.get("scenario", ""),
            label=row.get("label", ""),
            profile=row["profile"],
            company=row.get("company"),
            gold=row["gold"],
        )
        for row in payload["items"]
    ]
    return version, items


def iter_by_kind(items: list[EvalItem], kind: str) -> Iterator[EvalItem]:
    return (it for it in items if it.kind == kind)
