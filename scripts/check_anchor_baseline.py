"""PR-gate anchor comparator.

Reads a fresh eval snapshot and compares each anchor's per-mode outcome
against `anchor_baseline.json`.

Two regression severities, by mode:
- `integrated` is the production build — any baseline=PASS → new=fail is a
  blocking regression and the gate exits 1.
- `chat` is the comparison surface — its free-form multi-turn output is
  inherently stochastic between runs, so baseline=PASS → new=fail is
  reported as `drift` (informational) and does NOT fail the gate.
  This protects the gate against false alarms from chat's run-to-run
  variance, which empirically affects 30-40% of items on a 10-item set.
Improvements (baseline=fail → new=PASS) are reported in both modes but
do not fail the gate.

Usage:
  python -m scripts.check_anchor_baseline \\
    [--snapshot PATH] [--baseline PATH]

Defaults to data/eval_runs/latest.json and data/eval_runs/anchor_baseline.json.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any

from services.eval.metrics import score_passes_all_checks

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_SNAPSHOT = REPO_ROOT / "data" / "eval_runs" / "latest.json"
DEFAULT_BASELINE = REPO_ROOT / "data" / "eval_runs" / "anchor_baseline.json"

# Modes whose baseline=PASS → new=fail flips block the gate. Others get
# reported as informational drift. See module docstring.
GATING_MODES: frozenset[str] = frozenset({"integrated"})


def passes(p: dict[str, Any] | None) -> bool:
    if not p:
        return False
    return score_passes_all_checks(
        success=p["success"],
        action_correct=p["action_correct"],
        classification_overall=p["classification_overall"],
        adversarial_pass=p.get("adversarial_pass"),
        substring_grounded_rate=p["substring_grounded_rate"],
        claim_count=p.get("claim_count") or 0,
    )


def regression_reasons(p: dict[str, Any] | None) -> str:
    if not p:
        return "no per-item entry in snapshot"
    out: list[str] = []
    if not p["success"]:
        out.append("inference failed")
    if not p["action_correct"]:
        out.append(
            f"action={p.get('action_predicted')!r} expected={p.get('action_gold')!r}"
        )
    if not p["classification_overall"]:
        missed = [k for k, v in (p.get("classification_match") or {}).items() if not v]
        if missed:
            out.append("classification miss: " + ", ".join(missed))
    if p.get("claim_count") and p["substring_grounded_rate"] < 1.0:
        n_miss = p["claim_count"] - p["substring_grounded_count"]
        out.append(f"{n_miss}/{p['claim_count']} claim quote(s) not in input")
    if p.get("adversarial_pass") is False:
        out.append("adversarial check failed")
    return "; ".join(out) or "unknown"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare eval snapshot against anchor baseline."
    )
    parser.add_argument("--snapshot", type=pathlib.Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--baseline", type=pathlib.Path, default=DEFAULT_BASELINE)
    args = parser.parse_args()

    snapshot = json.loads(args.snapshot.read_text())
    baseline = json.loads(args.baseline.read_text())

    int_per = {p["item_id"]: p for p in snapshot["modes"]["integrated"]["per_item"]}
    chat_per = {p["item_id"]: p for p in snapshot["modes"]["chat"]["per_item"]}

    regressions = 0  # gating-mode flips — block the gate
    drifts = 0  # non-gating-mode flips — informational only
    improvements = 0
    n_assertions = 0

    print(f"Anchor gate: {args.snapshot.name} vs {args.baseline.name}")
    if snapshot.get("subset_filter"):
        print(f"  snapshot ran subset: {snapshot['subset_filter']}")
    if snapshot.get("judges_skipped"):
        print("  judges skipped (deterministic metrics only)")
    print(f"  gating modes: {sorted(GATING_MODES)}")
    print()

    for anchor in baseline["anchors"]:
        iid = anchor["id"]
        cells: list[str] = []
        for mode_key, mode_label, lookup in (
            ("integrated", "int", int_per),
            ("chat", "chat", chat_per),
        ):
            n_assertions += 1
            new = lookup.get(iid)
            new_pass = passes(new)
            base_pass = bool(anchor[mode_key]["passes"])
            if new_pass == base_pass:
                cells.append(f"{mode_label}: OK")
            elif new_pass and not base_pass:
                improvements += 1
                cells.append(f"{mode_label}: ++IMPROVEMENT (was fail, now PASS)")
            else:
                is_gating = mode_key in GATING_MODES
                if is_gating:
                    regressions += 1
                    severity = "!!REGRESSION"
                else:
                    drifts += 1
                    severity = "~~drift"
                cells.append(
                    f"{mode_label}: {severity} (was PASS, now fail: "
                    f"{regression_reasons(new)})"
                )
        note = anchor.get("note") or ""
        print(f"  [id={iid:>3}] {' | '.join(cells)}")
        if note:
            print(f"          {note}")

    ok = n_assertions - regressions - drifts - improvements
    print()
    print(
        f"Summary: {ok}/{n_assertions} hold baseline | "
        f"{improvements} improvement(s) | "
        f"{regressions} blocking regression(s) | "
        f"{drifts} drift(s) [informational]"
    )
    return 1 if regressions else 0


if __name__ == "__main__":
    sys.exit(main())
