"""
Latency-only re-measure for chat-multiturn.

Re-runs run_chat_multiturn against the same exemplars set the live snapshot
was scored on, using the corrected timer (per-call wall time inside the
semaphore). Writes a small JSON of item_id -> latency_ms to the path given
as --out. The companion patcher (`patch_chat_latency.py`) merges those
values into the snapshot in place.

This is intentionally lean: no scoring, no extractor pass for grounding,
no judge calls. The extractor is still invoked between turns because it
drives the multi-turn stop condition, which affects how many turns each
item runs.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import pathlib
import time

from anthropic import AsyncAnthropic

from services.eval.dataset import load_dataset
from services.eval.extractor import EXTRACTOR_CONCURRENCY, make_completeness_check
from services.eval.inference import REQUEST_TIMEOUT_S, MAX_RETRIES, run_chat_multiturn


async def _amain(out_path: pathlib.Path, max_turns: int) -> None:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise SystemExit("ANTHROPIC_API_KEY is not set")

    _, items = load_dataset()
    print(f"loaded {len(items)} items")

    client = AsyncAnthropic(
        api_key=api_key,
        timeout=REQUEST_TIMEOUT_S,
        max_retries=MAX_RETRIES,
    )
    extractor_sem = asyncio.Semaphore(EXTRACTOR_CONCURRENCY)
    completeness = make_completeness_check(client, extractor_sem)

    started = time.perf_counter()
    done = 0

    def _tick(_r) -> None:  # noqa: ANN001
        nonlocal done
        done += 1
        print(f"[chat] {done}/{len(items)}", flush=True)

    try:
        results = await run_chat_multiturn(
            items,
            extract_and_check=completeness,
            max_turns=max_turns,
            client=client,
            on_done=_tick,
        )
    finally:
        await client.close()

    elapsed = time.perf_counter() - started
    print(f"wall-clock pass took {elapsed:.1f}s")

    payload = {
        "schema_version": 1,
        "max_turns": max_turns,
        "n_items": len(results),
        "latencies": [
            {
                "item_id": r.item_id,
                "latency_ms": r.latency_ms,
                "turns_used": r.turns_used,
                "success": r.success,
                "cap_hit": r.cap_hit,
                "error": r.error,
            }
            for r in results
        ],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"wrote {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out",
        type=pathlib.Path,
        default=pathlib.Path("data/eval_runs/chat_latency_remeasure.json"),
    )
    ap.add_argument("--max-turns", type=int, default=3)
    args = ap.parse_args()
    asyncio.run(_amain(args.out, args.max_turns))


if __name__ == "__main__":
    main()
