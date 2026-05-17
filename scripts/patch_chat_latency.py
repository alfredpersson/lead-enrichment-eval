"""
Patch chat latency fields in an eval snapshot from a re-measure run.

Reads the new per-item latencies produced by `remeasure_chat_latency.py` and
writes them into:
  - modes.chat.inference_meta[i].latency_ms (by item_id)
  - modes.chat.aggregate.latency_p50_ms / latency_p95_ms (over patched list)
  - headline.chat.latency_p50_ms / latency_p95_ms
  - by_kind.<kind>.chat.latency_p50_ms / latency_p95_ms (sliced by kind)

robustness.* chat latencies are left untouched on purpose — they're not
displayed by the scorecard UI, so re-running the perturbation pass for
latency alone isn't worth the budget. The methodology page can call this
out.
"""

from __future__ import annotations

import argparse
import json
import pathlib

from services.eval.dataset import load_dataset


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    s = sorted(values)
    if len(s) == 1:
        return float(s[0])
    pos = q * (len(s) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    frac = pos - lo
    return float(s[lo] + (s[hi] - s[lo]) * frac)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--snapshot",
        type=pathlib.Path,
        default=pathlib.Path("data/eval_runs/latest.json"),
    )
    ap.add_argument(
        "--remeasure",
        type=pathlib.Path,
        default=pathlib.Path("data/eval_runs/chat_latency_remeasure.json"),
    )
    ap.add_argument(
        "--mirror",
        type=pathlib.Path,
        default=pathlib.Path("web/public/eval/latest.json"),
        help="Second path to write the patched snapshot to (web bundle).",
    )
    args = ap.parse_args()

    snapshot = json.loads(args.snapshot.read_text())
    remeasure = json.loads(args.remeasure.read_text())

    new_latency = {row["item_id"]: row["latency_ms"] for row in remeasure["latencies"]}
    _, items = load_dataset()
    item_kind = {it.id: it.kind for it in items}

    chat_meta = snapshot["modes"]["chat"]["inference_meta"]
    patched = 0
    missing = []
    for row in chat_meta:
        nl = new_latency.get(row["item_id"])
        if nl is None:
            missing.append(row["item_id"])
            continue
        row["latency_ms"] = nl
        patched += 1
    print(f"patched {patched} of {len(chat_meta)} chat inference_meta entries")
    if missing:
        print(f"WARNING: no re-measure for {len(missing)} items: {missing[:10]}")

    all_lat = [row["latency_ms"] for row in chat_meta]
    chat_agg = snapshot["modes"]["chat"]["aggregate"]
    chat_agg["latency_p50_ms"] = _percentile(all_lat, 0.5)
    chat_agg["latency_p95_ms"] = _percentile(all_lat, 0.95)

    head = snapshot["headline"]["chat"]
    head["latency_p50_ms"] = chat_agg["latency_p50_ms"]
    head["latency_p95_ms"] = chat_agg["latency_p95_ms"]

    for kind, block in snapshot["by_kind"].items():
        ids = {iid for iid, k in item_kind.items() if k == kind}
        lat_by_kind = [
            row["latency_ms"] for row in chat_meta if row["item_id"] in ids
        ]
        if not lat_by_kind:
            continue
        block["chat"]["latency_p50_ms"] = _percentile(lat_by_kind, 0.5)
        block["chat"]["latency_p95_ms"] = _percentile(lat_by_kind, 0.95)

    args.snapshot.write_text(json.dumps(snapshot, indent=2))
    print(f"wrote {args.snapshot}")
    if args.mirror:
        args.mirror.parent.mkdir(parents=True, exist_ok=True)
        args.mirror.write_text(json.dumps(snapshot, indent=2))
        print(f"mirrored to {args.mirror}")

    print(
        f"new chat headline latency: "
        f"p50={head['latency_p50_ms']:.0f}ms  p95={head['latency_p95_ms']:.0f}ms"
    )


if __name__ == "__main__":
    main()
