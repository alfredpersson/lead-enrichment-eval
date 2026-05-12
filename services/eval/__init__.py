"""Eval harness for the integrated vs chat comparison.

Public entry point is `services.eval.runner.run_eval`, which loads the test set,
runs both modes against Anthropic's batch API, scores deterministic and
judge-based metrics, and writes a JSON snapshot under `data/eval_runs/`.
"""
