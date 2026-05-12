# Eval run snapshots

Each file in this directory is one JSON snapshot produced by
`services.eval.runner`. The scorecard at `/scorecard` reads `latest.json` (a
copy of the most recent snapshot) plus `annotations.json` (curator-edited
incidents) and renders both. Historical dated snapshots (`YYYYMMDDTHHMMSSZ.json`)
stay in git so prospects can browse the timeline.

## File layout

```
data/eval_runs/
  latest.json                       # always the most recent snapshot
  annotations.json                  # dated incidents (eval-and-fix loop)
  20260512T020000Z.json             # historical (nightly cron, 02:00 UTC)
  20260512T020000Z-prefix.json      # tagged manual run (pre-fix)
  20260513T020000Z-postfix.json     # tagged manual run (post-fix)
  README.md                         # this file
```

## Running locally

```
APP_ENV=local ANTHROPIC_API_KEY=... OPENAI_API_KEY=... \
  uv run python -m services.eval.runner
```

Flags:

- `--tag <name>`  Append `-<name>` to the snapshot filename. Used to mark
  pre-fix / post-fix runs in the eval-and-fix loop.
- `--robustness {full,quick,none}`  `full` runs all three variants for every
  item (≈ 4 × API cost). `quick` samples 10 base items. `none` skips
  perturbations entirely; the scorecard's robustness section degrades gracefully.

Snapshots are written to this directory and also commit-mirrored to
`latest.json`.

## The eval-and-fix loop

The differentiating artifact is showing a real shipped fix that
addresses an observed eval failure, not just a static scorecard. The loop:

1. Run the eval at the current `HEAD`:
   `uv run python -m services.eval.runner --tag prefix`
2. Inspect the failure-modes block in the snapshot. Pick one perturbation
   class where the integrated build degrades (the injection variant is the
   most likely target — a model citing injection text inside a claim quote
   passes the substring check but fails the semantic grounding judge).
3. Ship the targeted fix as a prompt or tool-schema change. Keep the diff
   small: one rule added, one edge case named.
4. Re-run the eval:
   `uv run python -m services.eval.runner --tag postfix`
5. Append a new entry to `annotations.json` with the metric name, failure
   summary, fix summary, both snapshot paths, and the headline numbers
   before/after.
6. Commit both snapshots, the prompt diff, and the annotation in one PR.

The scorecard renders the annotation as a dated callout on the relevant
metric, with links to both snapshots. The point is the *production loop*,
visible alongside the static numbers — ship, measure, diagnose, fix.

## Cost notes

Rough order-of-magnitude estimates pending first real run. Update with
measured values from the first committed snapshot.

- Main pass: 73 items × 2 modes + Haiku extractor (single-turn equivalent
  multiplied by chat's average `turns_used`).
- Robustness pass: ≈ 3× main pass when `--robustness full`.
- Grounding judges: per claim across both modes (Opus + the configured
  OpenAI flagship).
- Hook coherence judge: per item across both modes (GPT-5-mini class).

The CI cron schedules at 02:00 UTC. No automatic budget cap is wired today;
the daily-limit tripwire on the plan is enforced live (the demo site
disables Run/Send buttons), not on the eval harness.
