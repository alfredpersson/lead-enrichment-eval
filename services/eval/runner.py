"""
Top-level eval orchestrator.

Reads the test set, runs both modes (integrated + chat→extractor) over the
full set and a robustness pass over perturbed variants, runs the LLM-judge
passes, then writes a JSON snapshot under `data/eval_runs/<timestamp>.json`
plus a `latest.json` pointer the scorecard reads.

Run locally:
    APP_ENV=local ANTHROPIC_API_KEY=... \
        python -m services.eval.runner --tag manual

CI runs this nightly via .github/workflows/eval.yml. `--robustness quick`
samples 10 items for smoke runs; `--robustness none` skips the perturbation
pass entirely.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import pathlib
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any, Callable

from anthropic import AsyncAnthropic

from services.eval.dataset import EvalItem, load_dataset
from services.eval.extractor import (
    EXTRACTOR_CONCURRENCY,
    ExtractorResult,
    extract_one_text,
    make_completeness_check,
)
from services.eval.inference import (
    CHAT_MODEL_ID,
    INTEGRATED_MODEL_ID,
    LIVE_CONCURRENCY,
    ChatResult,
    InferenceResult,
    run_chat_multiturn,
    run_integrated,
)
from services.eval.judges import (
    GroundingResults,
    HookResults,
    _anthropic as _anthropic_client_factory,
    _openai_client as _openai_client_factory,
    judge_grounding,
    judge_hooks,
)
from services.eval.metrics import (
    AggregateMetrics,
    PerItemScore,
    aggregate,
    score_item,
)
from services.eval.perturb import PerturbedItem, perturb_all

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
SNAPSHOTS_DIR = REPO_ROOT / "data" / "eval_runs"

DEFAULT_MAX_CHAT_TURNS = 3


def _git_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, stderr=subprocess.DEVNULL
        )
        return out.decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


# ----- Progress logging -------------------------------------------------------


class _Progress:
    """Per-phase counter with 1-second debounced printing.

    Designed for the long-running eval: the GitHub Actions log stays
    readable, and operators don't have to guess whether a 600-call run
    is making progress."""

    def __init__(self, label: str, total: int, *, interval: float = 1.0) -> None:
        self.label = label
        self.total = total
        self.done = 0
        self.interval = interval
        # Anchored at construction so the first tick respects the interval —
        # the print at completion is enough for short fast phases.
        self._last_print = time.monotonic()
        self._lock = asyncio.Lock()

    def tick(self, _result: Any = None) -> None:
        self.done += 1
        now = time.monotonic()
        if self.done == self.total or now - self._last_print >= self.interval:
            self._last_print = now
            print(
                f"[eval][{self.label}] {self.done}/{self.total}",
                flush=True,
            )


# ----- Result-to-dict helpers ------------------------------------------------


def _inference_meta(res: InferenceResult) -> dict[str, Any]:
    return {
        "item_id": res.item_id,
        "success": res.success,
        "error": res.error,
        "latency_ms": res.latency_ms,
        "input_tokens": res.input_tokens,
        "output_tokens": res.output_tokens,
        "thinking_tokens": res.thinking_tokens,
        "cache_read_tokens": res.cache_read_tokens,
        "stop_reason": res.raw_stop_reason,
    }


def _chat_meta(res: ChatResult) -> dict[str, Any]:
    return {
        "item_id": res.item_id,
        "success": res.success,
        "error": res.error,
        "latency_ms": res.latency_ms,
        "input_tokens": res.input_tokens,
        "output_tokens": res.output_tokens,
        "cache_read_tokens": res.cache_read_tokens,
        "turns_used": res.turns_used,
        "cap_hit": res.cap_hit,
    }


def _per_item_to_dict(s: PerItemScore) -> dict[str, Any]:
    return asdict(s)


def _aggregate_to_dict(a: AggregateMetrics) -> dict[str, Any]:
    return asdict(a)


def _grounding_to_dict(g: GroundingResults) -> dict[str, Any]:
    return {
        "n_claims": g.n_claims,
        "n_judged_by_opus": g.n_judged,
        "opus_grounding_rate": g.opus_rate,
        "openai_grounding_rate": g.openai_rate,
        "headline_grounding_rate": g.headline_rate,
        "cohen_kappa": g.kappa,
        "judgements": [asdict(j) for j in g.judgements],
    }


def _hooks_to_dict(h: HookResults) -> dict[str, Any]:
    return {
        "n_scored": h.n_scored,
        "pass_rate": h.pass_rate,
        "judgements": [asdict(j) for j in h.judgements],
    }


# ----- Scoring helpers --------------------------------------------------------


def _score_mode(
    items: list[EvalItem],
    latencies_by_id: dict[str, int],
    tokens_in_by_id: dict[str, int],
    tokens_out_by_id: dict[str, int],
    outputs: dict[str, dict[str, Any] | None],
) -> tuple[list[PerItemScore], AggregateMetrics]:
    scores = [score_item(it, outputs.get(it.id)) for it in items]
    latencies = [latencies_by_id[it.id] for it in items if it.id in latencies_by_id]
    tokens_in = [tokens_in_by_id[it.id] for it in items if it.id in tokens_in_by_id]
    tokens_out = [tokens_out_by_id[it.id] for it in items if it.id in tokens_out_by_id]
    agg = aggregate(scores, latencies, tokens_in, tokens_out)
    return scores, agg


def _failure_modes(scores: list[PerItemScore]) -> list[dict[str, Any]]:
    """Cluster failed items by reason so the scorecard shows concrete misses,
    not just aggregate numbers."""
    out = []
    for s in scores:
        grounded_ok = not s.claim_count or s.substring_grounded_rate >= 1.0
        if (
            s.success
            and s.action_correct
            and s.classification_overall
            and s.adversarial_pass is not False
            and grounded_ok
        ):
            continue
        reasons: list[str] = []
        if not s.success:
            reasons.append("inference failed")
        if s.action_predicted and s.action_gold and not s.action_correct:
            reasons.append(f"action {s.action_predicted}, expected {s.action_gold}")
        if not s.classification_overall:
            missed = [k for k, v in s.classification_match.items() if not v]
            if missed:
                reasons.append("classification miss: " + ", ".join(missed))
        if s.substring_grounded_rate < 1.0 and s.claim_count:
            reasons.append(
                f"{s.claim_count - s.substring_grounded_count}/{s.claim_count}"
                " claim source quote(s) not in input"
            )
        if s.adversarial_pass is False:
            reasons.extend(s.adversarial_failures)
        if reasons:
            out.append({"item_id": s.item_id, "kind": s.kind, "reasons": reasons})
    return out


# ----- Combined pass: integrated + chat-multiturn + extractor ----------------


async def _run_combined_pass(
    items: list[EvalItem],
    *,
    label: str,
    anthropic_client: AsyncAnthropic,
    max_chat_turns: int,
) -> dict[str, Any]:
    """Run one labelled pass of integrated + chat-multiturn + extractor.

    Returns a dict of intermediate results so a caller can run multiple
    passes (main, robustness variants) concurrently."""
    if not items:
        return {
            "integrated": [],
            "chat": [],
            "extractor": [],
            "integrated_outputs": {},
            "chat_outputs": {},
        }
    progress_integrated = _Progress(f"{label}:integrated", len(items))
    progress_chat = _Progress(f"{label}:chat", len(items))

    extractor_sem = asyncio.Semaphore(EXTRACTOR_CONCURRENCY)
    completeness = make_completeness_check(anthropic_client, extractor_sem)

    integrated_task = asyncio.create_task(
        run_integrated(
            items,
            client=anthropic_client,
            on_done=progress_integrated.tick,
        )
    )
    chat_task = asyncio.create_task(
        run_chat_multiturn(
            items,
            extract_and_check=completeness,
            max_turns=max_chat_turns,
            client=anthropic_client,
            on_done=progress_chat.tick,
        )
    )
    integrated_results, chat_results = await asyncio.gather(
        integrated_task, chat_task
    )

    # Final extractor pass against accumulated chat text. This is the
    # canonical structured-output set for scoring; the per-turn checks above
    # only drove the multi-turn stop condition.
    print(f"[eval][{label}] final extractor pass...", flush=True)
    extractor_results: list[ExtractorResult] = []
    for chat in chat_results:
        item = next(it for it in items if it.id == chat.item_id)
        r = await extract_one_text(
            anthropic_client,
            item,
            chat.text if chat.success else None,
            extractor_sem,
            fallback_error=chat.error if not chat.success else None,
        )
        extractor_results.append(r)

    integrated_outputs = {r.item_id: r.output for r in integrated_results}
    chat_outputs = {r.item_id: r.output for r in extractor_results}
    return {
        "integrated": integrated_results,
        "chat": chat_results,
        "extractor": extractor_results,
        "integrated_outputs": integrated_outputs,
        "chat_outputs": chat_outputs,
    }


def _robustness_block(
    perturbed: list[PerturbedItem],
    pass_results: dict[str, Any],
    *,
    include_per_item: bool,
) -> dict[str, Any]:
    integrated_results = pass_results["integrated"]
    chat_results = pass_results["chat"]
    integrated_outputs = pass_results["integrated_outputs"]
    chat_outputs = pass_results["chat_outputs"]

    int_lat = {r.item_id: r.latency_ms for r in integrated_results}
    int_in = {r.item_id: r.input_tokens for r in integrated_results}
    int_out = {r.item_id: r.output_tokens for r in integrated_results}
    chat_lat = {r.item_id: r.latency_ms for r in chat_results}
    chat_in = {r.item_id: r.input_tokens for r in chat_results}
    chat_out = {r.item_id: r.output_tokens for r in chat_results}

    by_variant: dict[str, dict[str, Any]] = {}
    for variant in ("typos", "sentence_reorder", "injection"):
        variant_items = [p.item for p in perturbed if p.variant == variant]
        i_scores, i_agg = _score_mode(
            variant_items, int_lat, int_in, int_out, integrated_outputs
        )
        c_scores, c_agg = _score_mode(
            variant_items, chat_lat, chat_in, chat_out, chat_outputs
        )
        block: dict[str, Any] = {
            "n": len(variant_items),
            "integrated": _aggregate_to_dict(i_agg),
            "chat": _aggregate_to_dict(c_agg),
        }
        if include_per_item:
            block["per_item_integrated"] = [_per_item_to_dict(s) for s in i_scores]
            block["per_item_chat"] = [_per_item_to_dict(s) for s in c_scores]
        by_variant[variant] = block
    return {"by_variant": by_variant, "n_base_items": len({p.base.id for p in perturbed})}


# ----- Top-level run ----------------------------------------------------------


async def _run_full(
    tag: str | None,
    robustness: str,
    *,
    max_chat_turns: int,
    include_per_item: bool,
) -> dict[str, Any]:
    version, items = load_dataset()
    started = time.time()
    started_iso = datetime.now(timezone.utc).isoformat()

    anthropic_client = _anthropic_client_factory()
    openai_client = _openai_client_factory()

    try:
        # Main pass and robustness pass run concurrently — they share the
        # one Anthropic client and are bounded by its concurrency setting.
        main_task = asyncio.create_task(
            _run_combined_pass(
                items,
                label="main",
                anthropic_client=anthropic_client,
                max_chat_turns=max_chat_turns,
            )
        )

        if robustness == "full":
            perturbed = perturb_all(items)
            robustness_task = asyncio.create_task(
                _run_combined_pass(
                    [p.item for p in perturbed],
                    label="robust",
                    anthropic_client=anthropic_client,
                    max_chat_turns=max_chat_turns,
                )
            )
        elif robustness == "quick":
            sample = items[:10]
            perturbed = perturb_all(sample)
            robustness_task = asyncio.create_task(
                _run_combined_pass(
                    [p.item for p in perturbed],
                    label="robust",
                    anthropic_client=anthropic_client,
                    max_chat_turns=max_chat_turns,
                )
            )
        else:
            perturbed = []
            robustness_task = None

        if robustness_task is not None:
            main_results, robust_results = await asyncio.gather(
                main_task, robustness_task
            )
        else:
            main_results = await main_task
            robust_results = None

        integrated_inference = main_results["integrated"]
        chat_results = main_results["chat"]
        extractor_results = main_results["extractor"]
        integrated_outputs = main_results["integrated_outputs"]
        chat_outputs = main_results["chat_outputs"]

        # Score main pass.
        int_lat = {r.item_id: r.latency_ms for r in integrated_inference}
        int_in = {r.item_id: r.input_tokens for r in integrated_inference}
        int_out = {r.item_id: r.output_tokens for r in integrated_inference}
        chat_lat = {r.item_id: r.latency_ms for r in chat_results}
        chat_in = {r.item_id: r.input_tokens for r in chat_results}
        chat_out = {r.item_id: r.output_tokens for r in chat_results}

        integrated_scores, integrated_agg = _score_mode(
            items, int_lat, int_in, int_out, integrated_outputs
        )
        chat_scores, chat_agg = _score_mode(
            items, chat_lat, chat_in, chat_out, chat_outputs
        )

        print("[eval] grounding judges...", flush=True)
        items_by_id = {it.id: it for it in items}
        integrated_grounding = await judge_grounding(
            items_by_id,
            integrated_outputs,
            anthropic_client=anthropic_client,
            openai_client=openai_client,
        )
        chat_grounding = await judge_grounding(
            items_by_id,
            chat_outputs,
            anthropic_client=anthropic_client,
            openai_client=openai_client,
        )

        print("[eval] hook coherence judge...", flush=True)
        integrated_hooks = await judge_hooks(
            items_by_id, integrated_outputs, openai_client=openai_client
        )
        chat_hooks = await judge_hooks(
            items_by_id, chat_outputs, openai_client=openai_client
        )

        robustness_block: dict[str, Any] | None = (
            _robustness_block(perturbed, robust_results, include_per_item=include_per_item)
            if robust_results is not None
            else None
        )

        # Chat steps-to-completion summary.
        chat_turns_dist = {1: 0, 2: 0, 3: 0}
        chat_cap_hits = 0
        for r in chat_results:
            chat_turns_dist[min(r.turns_used, 3)] = (
                chat_turns_dist.get(min(r.turns_used, 3), 0) + 1
            )
            if r.cap_hit:
                chat_cap_hits += 1
        avg_turns = (
            sum(r.turns_used for r in chat_results) / len(chat_results)
            if chat_results
            else None
        )

        completed_iso = datetime.now(timezone.utc).isoformat()
        elapsed_s = round(time.time() - started, 1)

        by_kind: dict[str, dict[str, Any]] = {}
        for kind in {it.kind for it in items}:
            kind_items = [it for it in items if it.kind == kind]
            _, kind_int_agg = _score_mode(
                kind_items, int_lat, int_in, int_out, integrated_outputs
            )
            _, kind_chat_agg = _score_mode(
                kind_items, chat_lat, chat_in, chat_out, chat_outputs
            )
            by_kind[kind] = {
                "n": len(kind_items),
                "integrated": _aggregate_to_dict(kind_int_agg),
                "chat": _aggregate_to_dict(kind_chat_agg),
            }

        snapshot = {
            "schema_version": 3,
            "run_tag": tag,
            "git_sha": _git_sha(),
            "test_set_version": version,
            "n_items": len(items),
            "started_at": started_iso,
            "completed_at": completed_iso,
            "elapsed_seconds": elapsed_s,
            "models": {
                "integrated": INTEGRATED_MODEL_ID,
                "chat": CHAT_MODEL_ID,
                "extractor": "claude-haiku-4-5-20251001",
                "grounding_judges": {
                    "anthropic": "claude-opus-4-7",
                    "openai": os.environ.get("OPENAI_GROUNDING_MODEL", "gpt-5"),
                },
                "hook_judge": os.environ.get("OPENAI_HOOK_MODEL", "gpt-5-mini"),
            },
            "headline": {
                "integrated": {
                    "classification_accuracy": integrated_agg.classification_accuracy,
                    "action_accuracy": integrated_agg.action_accuracy,
                    "fit_spearman": integrated_agg.fit_spearman,
                    "substring_grounding_rate": integrated_agg.substring_grounded_rate,
                    "judge_grounding_rate": integrated_grounding.headline_rate,
                    "hook_pass_rate": integrated_hooks.pass_rate,
                    "latency_p50_ms": integrated_agg.latency_p50_ms,
                    "latency_p95_ms": integrated_agg.latency_p95_ms,
                },
                "chat": {
                    "classification_accuracy": chat_agg.classification_accuracy,
                    "action_accuracy": chat_agg.action_accuracy,
                    "fit_spearman": chat_agg.fit_spearman,
                    "substring_grounding_rate": chat_agg.substring_grounded_rate,
                    "judge_grounding_rate": chat_grounding.headline_rate,
                    "hook_pass_rate": chat_hooks.pass_rate,
                    "latency_p50_ms": chat_agg.latency_p50_ms,
                    "latency_p95_ms": chat_agg.latency_p95_ms,
                    "extractor_complete_rate": (
                        sum(1 for r in extractor_results if r.extractor_complete)
                        / len(extractor_results)
                        if extractor_results
                        else 0.0
                    ),
                    "avg_turns_used": avg_turns,
                    "cap_hit_rate": (
                        chat_cap_hits / len(chat_results) if chat_results else 0.0
                    ),
                },
            },
            "modes": {
                "integrated": {
                    "aggregate": _aggregate_to_dict(integrated_agg),
                    "grounding": _grounding_to_dict(integrated_grounding),
                    "hooks": _hooks_to_dict(integrated_hooks),
                    "per_item": [_per_item_to_dict(s) for s in integrated_scores],
                    "inference_meta": [
                        _inference_meta(r) for r in integrated_inference
                    ],
                    "failure_modes": _failure_modes(integrated_scores),
                },
                "chat": {
                    "aggregate": _aggregate_to_dict(chat_agg),
                    "grounding": _grounding_to_dict(chat_grounding),
                    "hooks": _hooks_to_dict(chat_hooks),
                    "per_item": [_per_item_to_dict(s) for s in chat_scores],
                    "inference_meta": [_chat_meta(r) for r in chat_results],
                    "turns_distribution": chat_turns_dist,
                    "cap_hits": chat_cap_hits,
                    "extractor": [
                        {
                            "item_id": r.item_id,
                            "success": r.success,
                            "extractor_complete": r.extractor_complete,
                            "error": r.error,
                            "input_tokens": r.input_tokens,
                            "output_tokens": r.output_tokens,
                            "notes_unparsed": r.notes_unparsed,
                        }
                        for r in extractor_results
                    ],
                    "failure_modes": _failure_modes(chat_scores),
                },
            },
            "by_kind": by_kind,
            "robustness": robustness_block,
        }
        return snapshot
    finally:
        await anthropic_client.close()
        if openai_client is not None and hasattr(openai_client, "close"):
            close = getattr(openai_client, "close")
            if asyncio.iscoroutinefunction(close):
                await close()


def _write_snapshot(snapshot: dict[str, Any], tag: str | None) -> pathlib.Path:
    SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.fromisoformat(snapshot["completed_at"]).strftime("%Y%m%dT%H%M%SZ")
    name = f"{stamp}{('-' + tag) if tag else ''}.json"
    path = SNAPSHOTS_DIR / name
    path.write_text(json.dumps(snapshot, indent=2, default=str))
    (SNAPSHOTS_DIR / "latest.json").write_text(
        json.dumps(snapshot, indent=2, default=str)
    )
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the lead-enrichment eval.")
    parser.add_argument("--tag", help="Snapshot filename suffix (e.g. 'prefix').")
    parser.add_argument(
        "--robustness",
        choices=("full", "quick", "none"),
        default="full",
        help="Robustness pass scope (default: full).",
    )
    parser.add_argument(
        "--max-chat-turns",
        type=int,
        default=DEFAULT_MAX_CHAT_TURNS,
        help=(
            "Cap on chat-mode user turns (default: 3). Cap-hit is scored "
            "as failure per the plan's steps-to-completion metric."
        ),
    )
    parser.add_argument(
        "--include-per-item",
        action="store_true",
        help=(
            "Include per-item arrays in the robustness block. Off by "
            "default — the scorecard never renders them and they roughly "
            "triple snapshot size."
        ),
    )
    args = parser.parse_args()

    try:
        snapshot = asyncio.run(
            _run_full(
                args.tag,
                args.robustness,
                max_chat_turns=args.max_chat_turns,
                include_per_item=args.include_per_item,
            )
        )
    except KeyboardInterrupt:
        print("[eval] interrupted", file=sys.stderr)
        return 130
    path = _write_snapshot(snapshot, args.tag)
    print(f"[eval] wrote {path}")
    print(
        f"[eval] integrated.classification = "
        f"{snapshot['headline']['integrated']['classification_accuracy']:.2%}"
    )
    print(
        f"[eval] chat.classification       = "
        f"{snapshot['headline']['chat']['classification_accuracy']:.2%}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
