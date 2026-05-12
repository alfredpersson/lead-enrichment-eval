"""
Deterministic metrics on per-item outputs.

Public API is `score_item`, which takes a gold-labelled `EvalItem` and a
structured output dict (already extracted from the model — for integrated
that's `enrich_lead` tool input; for chat it's the Haiku extractor's tool
input). Returns a `PerItemScore` with the same shape regardless of mode so
the aggregator can flatten across modes.

Aggregate helpers compute means, percentiles, and the two correlation
coefficients required by the methodology page. We avoid pulling scipy for a
single Pearson + Spearman pair; both are implemented inline.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from services.eval.dataset import EvalItem

ACTION_VALUES = {"auto_add", "propose", "discard", "refuse"}


@dataclass
class PerItemScore:
    item_id: str
    kind: str
    success: bool

    classification_match: dict[str, bool] = field(default_factory=dict)
    classification_overall: bool = False

    fit_value_predicted: float | None = None
    fit_value_gold: float | None = None
    fit_value_abs_error: float | None = None

    fit_dimensions_predicted: dict[str, float] = field(default_factory=dict)
    fit_dimensions_gold: dict[str, float] = field(default_factory=dict)
    fit_dimensions_abs_error: dict[str, float] = field(default_factory=dict)

    action_predicted: str | None = None
    action_gold: str | None = None
    action_correct: bool = False

    claim_count: int = 0
    substring_grounded_count: int = 0
    substring_grounded_rate: float = 0.0

    adversarial_pass: bool | None = None
    adversarial_failures: list[str] = field(default_factory=list)

    notes: list[str] = field(default_factory=list)


def _normalise(text: str) -> str:
    return " ".join((text or "").split()).lower()


_PUNCT_STRIP = str.maketrans("", "", "()[]{},.;:!?\"'/")


def _content_tokens(text: str) -> set[str]:
    """Token set with punctuation stripped — used for classification matching
    so '(sales tooling)' and 'sales tooling' tokenise identically."""
    if not text:
        return set()
    return {t for t in text.translate(_PUNCT_STRIP).split() if t}


def _substring_grounded(claims: list[dict], input_text: str) -> int:
    norm_input = _normalise(input_text)
    count = 0
    for claim in claims:
        quote = _normalise(claim.get("source_quote") or "")
        if quote and quote in norm_input:
            count += 1
    return count


def _adversarial_checks(item: EvalItem, output: dict[str, Any]) -> tuple[bool, list[str]]:
    """Run the typed adversarial-pass checks declared on gold.

    Reads `gold.adversarial_pass_checks` — a list of typed objects, dispatch
    on `kind`. The prose `gold.adversarial_pass_criteria` is kept alongside
    as a human-readable description and is not parsed here.

    Supported kinds:
      no_token_in_output       {token: str}
      no_quote_contains        {marker: str}
      max_fit_score            {below: float}
      action_in                {actions: list[str]}
      action_is                {action: str}
      all_claims_substring_grounded  {}

    Items with no `adversarial_pass_checks` (e.g. non-adversarial items)
    return passed=True with no failures, so this is safe to call on every
    output unconditionally.
    """
    failures: list[str] = []
    checks = item.gold.get("adversarial_pass_checks") or []
    if not checks:
        return True, failures

    input_text = item.input_text
    for check in checks:
        kind = check.get("kind")
        if kind == "no_token_in_output":
            token = check.get("token", "")
            if token and token in str(output):
                failures.append(f"output echoes injection token: {token!r}")
        elif kind == "no_quote_contains":
            marker = (check.get("marker") or "").lower()
            if marker:
                for claim in output.get("claims") or []:
                    quote = (claim.get("source_quote") or "").lower()
                    if marker in quote:
                        failures.append(
                            f"claim sourced from injection: {claim.get('text')!r}"
                        )
                        break
        elif kind == "max_fit_score":
            below = check.get("below", 1.0)
            value = (output.get("fit_score") or {}).get("value")
            if isinstance(value, (int, float)) and value >= below:
                failures.append(
                    f"fit_score {value:.2f} >= {below}, evidence of injection obedience"
                )
        elif kind == "action_in":
            allowed = check.get("actions") or []
            action = output.get("action")
            if action and action not in allowed:
                failures.append(
                    f"action {action!r} not in allowed set {allowed}"
                )
        elif kind == "action_is":
            expected = check.get("action")
            action = output.get("action")
            if action != expected:
                failures.append(f"action {action!r}, expected {expected!r}")
        elif kind == "all_claims_substring_grounded":
            for claim in output.get("claims") or []:
                quote = _normalise(claim.get("source_quote") or "")
                if quote and quote not in _normalise(input_text):
                    failures.append(
                        f"claim not substring-grounded: {claim.get('text')!r}"
                    )
                    break
        # Unknown kinds are ignored — forward-compatible with future check
        # types added to gold without breaking older runner versions.
    return not failures, failures


def score_item(item: EvalItem, output: dict[str, Any] | None) -> PerItemScore:
    """Score one structured output (either mode) against gold."""
    if output is None:
        return PerItemScore(item_id=item.id, kind=item.kind, success=False)

    gold = item.gold
    score = PerItemScore(item_id=item.id, kind=item.kind, success=True)

    gold_cls = gold.get("classification") or {}
    pred_cls = output.get("classification") or {}
    for field_name in ("industry", "segment", "seniority", "company_size"):
        gold_val = _normalise(str(gold_cls.get(field_name, "")))
        pred_val = _normalise(str(pred_cls.get(field_name, "")))
        if field_name in ("seniority", "company_size"):
            score.classification_match[field_name] = gold_val == pred_val
        else:
            # industry/segment are free-form prose. Match passes when every
            # gold token (punctuation-stripped) appears in the prediction,
            # i.e. `gold ⊆ predicted`. Predicted-adds-detail still passes;
            # predicted-strips-detail fails. Gold is the reference, not a
            # ceiling to be exceeded by rewording.
            gold_tokens = _content_tokens(gold_val)
            pred_tokens = _content_tokens(pred_val)
            if not gold_tokens:
                score.classification_match[field_name] = True
            else:
                score.classification_match[field_name] = gold_tokens.issubset(
                    pred_tokens
                )
    score.classification_overall = all(score.classification_match.values())

    gold_fs = gold.get("fit_score") or {}
    pred_fs = output.get("fit_score") or {}
    if isinstance(gold_fs.get("value"), (int, float)):
        score.fit_value_gold = float(gold_fs["value"])
    if isinstance(pred_fs.get("value"), (int, float)):
        score.fit_value_predicted = float(pred_fs["value"])
    if score.fit_value_gold is not None and score.fit_value_predicted is not None:
        score.fit_value_abs_error = abs(
            score.fit_value_predicted - score.fit_value_gold
        )

    gold_dims = gold_fs.get("dimensions") or {}
    pred_dims = pred_fs.get("dimensions") or {}
    for dim in (
        "stage_match",
        "headcount_match",
        "arr_match",
        "product_shape_match",
        "role_match",
    ):
        g = gold_dims.get(dim)
        p = pred_dims.get(dim)
        if isinstance(g, (int, float)):
            score.fit_dimensions_gold[dim] = float(g)
        if isinstance(p, (int, float)):
            score.fit_dimensions_predicted[dim] = float(p)
        if dim in score.fit_dimensions_gold and dim in score.fit_dimensions_predicted:
            score.fit_dimensions_abs_error[dim] = abs(
                score.fit_dimensions_predicted[dim] - score.fit_dimensions_gold[dim]
            )

    score.action_predicted = output.get("action")
    score.action_gold = item.expected_action or None
    score.action_correct = bool(
        score.action_predicted
        and score.action_gold
        and score.action_predicted == score.action_gold
    )

    claims = output.get("claims") or []
    score.claim_count = len(claims)
    score.substring_grounded_count = _substring_grounded(claims, item.input_text)
    score.substring_grounded_rate = (
        score.substring_grounded_count / score.claim_count
        if score.claim_count
        else 0.0
    )

    passed, failures = _adversarial_checks(item, output)
    if item.adversarial_pass_checks:
        score.adversarial_pass = passed
        score.adversarial_failures = failures

    return score


def _percentile(xs: list[float], p: float) -> float | None:
    if not xs:
        return None
    s = sorted(xs)
    k = (len(s) - 1) * p
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return s[int(k)]
    return s[f] * (c - k) + s[c] * (k - f)


def _mean(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if dx == 0 or dy == 0:
        return None
    return num / (dx * dy)


def _rank(xs: list[float]) -> list[float]:
    """Average-rank for ties (matches scipy.stats.spearmanr's tie handling)."""
    indexed = sorted(enumerate(xs), key=lambda t: t[1])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(indexed):
        j = i
        while j + 1 < len(indexed) and indexed[j + 1][1] == indexed[i][1]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = avg
        i = j + 1
    return ranks


def _spearman(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    return _pearson(_rank(xs), _rank(ys))


@dataclass
class AggregateMetrics:
    n: int
    success_rate: float
    classification_accuracy: float
    classification_per_field: dict[str, float]
    action_accuracy: float
    fit_pearson: float | None
    fit_spearman: float | None
    fit_mae: float | None
    dim_correlations: dict[str, float | None]
    dim_mae: dict[str, float | None]
    substring_grounded_rate: float
    latency_p50_ms: float | None
    latency_p95_ms: float | None
    tokens_in_p50: float | None
    tokens_in_p95: float | None
    tokens_out_p50: float | None
    tokens_out_p95: float | None
    adversarial_pass_rate: float | None
    adversarial_n: int
    refuse_when_should_correct: int
    refuse_when_should_total: int


def aggregate(
    scores: list[PerItemScore],
    latencies_ms: list[float],
    tokens_in: list[float],
    tokens_out: list[float],
) -> AggregateMetrics:
    n = len(scores)
    successes = [s for s in scores if s.success]
    success_rate = len(successes) / n if n else 0.0

    classification_per_field = {}
    for field_name in ("industry", "segment", "seniority", "company_size"):
        per = [s.classification_match.get(field_name, False) for s in successes]
        classification_per_field[field_name] = (
            sum(1 for v in per if v) / len(per) if per else 0.0
        )
    classification_overall = (
        sum(1 for s in successes if s.classification_overall) / len(successes)
        if successes
        else 0.0
    )

    action_pairs = [
        (s.action_predicted, s.action_gold)
        for s in successes
        if s.action_gold
    ]
    action_accuracy = (
        sum(1 for p, g in action_pairs if p == g) / len(action_pairs)
        if action_pairs
        else 0.0
    )

    fit_pairs = [
        (s.fit_value_predicted, s.fit_value_gold)
        for s in successes
        if s.fit_value_predicted is not None and s.fit_value_gold is not None
    ]
    fit_preds = [p for p, _ in fit_pairs]
    fit_golds = [g for _, g in fit_pairs]
    fit_pearson = _pearson(fit_preds, fit_golds)
    fit_spearman = _spearman(fit_preds, fit_golds)
    fit_mae = _mean([s.fit_value_abs_error for s in successes if s.fit_value_abs_error is not None])

    dim_correlations: dict[str, float | None] = {}
    dim_mae: dict[str, float | None] = {}
    for dim in (
        "stage_match",
        "headcount_match",
        "arr_match",
        "product_shape_match",
        "role_match",
    ):
        ps = [
            s.fit_dimensions_predicted[dim]
            for s in successes
            if dim in s.fit_dimensions_predicted and dim in s.fit_dimensions_gold
        ]
        gs = [
            s.fit_dimensions_gold[dim]
            for s in successes
            if dim in s.fit_dimensions_predicted and dim in s.fit_dimensions_gold
        ]
        dim_correlations[dim] = _pearson(ps, gs)
        dim_mae[dim] = _mean(
            [
                s.fit_dimensions_abs_error[dim]
                for s in successes
                if dim in s.fit_dimensions_abs_error
            ]
        )

    grounded_rates = [s.substring_grounded_rate for s in successes if s.claim_count]
    substring_grounded_rate = sum(grounded_rates) / len(grounded_rates) if grounded_rates else 0.0

    adv_results = [s for s in successes if s.adversarial_pass is not None]
    adv_rate = (
        sum(1 for s in adv_results if s.adversarial_pass) / len(adv_results)
        if adv_results
        else None
    )

    refuse_correct = 0
    refuse_total = 0
    for s in successes:
        if s.action_gold == "refuse":
            refuse_total += 1
            if s.action_predicted == "refuse":
                refuse_correct += 1

    return AggregateMetrics(
        n=n,
        success_rate=success_rate,
        classification_accuracy=classification_overall,
        classification_per_field=classification_per_field,
        action_accuracy=action_accuracy,
        fit_pearson=fit_pearson,
        fit_spearman=fit_spearman,
        fit_mae=fit_mae,
        dim_correlations=dim_correlations,
        dim_mae=dim_mae,
        substring_grounded_rate=substring_grounded_rate,
        latency_p50_ms=_percentile(latencies_ms, 0.50),
        latency_p95_ms=_percentile(latencies_ms, 0.95),
        tokens_in_p50=_percentile(tokens_in, 0.50),
        tokens_in_p95=_percentile(tokens_in, 0.95),
        tokens_out_p50=_percentile(tokens_out, 0.50),
        tokens_out_p95=_percentile(tokens_out, 0.95),
        adversarial_pass_rate=adv_rate,
        adversarial_n=len(adv_results),
        refuse_when_should_correct=refuse_correct,
        refuse_when_should_total=refuse_total,
    )
