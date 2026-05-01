"""
Deterministic scoring for the integrated feature demo.

Covers:
- stage_match, headcount_match, arr_match: numeric range checks against the ICP.
- compute_action: action selection from holistic fit score and grounding state.

product_shape_match and role_match are anchored 0 / 0.25 / 0.5 / 0.75 / 1.0
rubrics evaluated by the model at runtime and by the labeller for gold; their
implementation is not in this module.

The holistic fit_score.value comes from the model, not a weighted sum of the
five dimensions. Dimensions exist for transparency and per-dimension correlation
in eval, not as inputs to the holistic score.
"""

from typing import Literal

from pydantic import BaseModel


Stage = Literal[
    'Pre-seed', 'Seed', 'Series A', 'Series B', 'Series C', 'Series D+', 'Public'
]
Action = Literal['auto_add', 'propose', 'discard', 'refuse']

STAGE_ORDER: list[Stage] = [
    'Pre-seed', 'Seed', 'Series A', 'Series B', 'Series C', 'Series D+', 'Public',
]


class ICPDefinition(BaseModel):
    stages: list[Stage]
    headcount_min: int
    headcount_max: int
    arr_min_usd: int
    arr_max_usd: int
    product_shape: str
    target_roles: list[str]


def stage_match(stage: Stage | str, icp: ICPDefinition) -> float:
    """1.0 in target. 0.5 one stage off. 0.25 two stages off. 0.0 otherwise."""
    if stage in icp.stages:
        return 1.0
    if stage not in STAGE_ORDER:
        return 0.0
    target_indices = [STAGE_ORDER.index(s) for s in icp.stages]
    if not target_indices:
        return 0.0
    distance = min(abs(STAGE_ORDER.index(stage) - i) for i in target_indices)
    if distance == 1:
        return 0.5
    if distance == 2:
        return 0.25
    return 0.0


def _range_score(value: float, low: float, high: float) -> float:
    """
    Anchored taper using a multiplicative ratio so low and high sides are
    symmetric. ratio = (larger / smaller) - 1, computed against whichever
    boundary the value crosses.

    Tiers: in-range 1.0, ratio <= 0.25 -> 0.75, <= 0.5 -> 0.5, <= 1.0 -> 0.25,
    else 0.0.
    """
    if value <= 0 or low <= 0 or high <= 0:
        return 0.0
    if low <= value <= high:
        return 1.0
    if value < low:
        ratio = low / value - 1.0
    else:
        ratio = value / high - 1.0
    if ratio <= 0.25:
        return 0.75
    if ratio <= 0.5:
        return 0.5
    if ratio <= 1.0:
        return 0.25
    return 0.0


def headcount_match(headcount: int, icp: ICPDefinition) -> float:
    return _range_score(headcount, icp.headcount_min, icp.headcount_max)


def arr_match(arr_usd: int | None, icp: ICPDefinition) -> float:
    if arr_usd is None:
        return 0.0
    return _range_score(arr_usd, icp.arr_min_usd, icp.arr_max_usd)


def compute_action(
    fit_score: float,
    all_claims_grounded: bool,
    sufficient_data: bool,
) -> Action:
    """
    Action thresholds (see methodology page):
      refuse   when input lacks data to judge
      auto_add when fit > 0.80 and every claim grounded
      propose  when fit in [0.50, 0.80] OR any claim ungrounded
      discard  when fit < 0.50 with all claims grounded
    """
    if not sufficient_data:
        return 'refuse'
    if not all_claims_grounded:
        return 'propose'
    if fit_score > 0.80:
        return 'auto_add'
    if fit_score >= 0.50:
        return 'propose'
    return 'discard'


DEMO_ICP = ICPDefinition(
    stages=['Series A', 'Series B', 'Series C'],
    headcount_min=20,
    headcount_max=250,
    arr_min_usd=2_000_000,
    arr_max_usd=50_000_000,
    product_shape=(
        'B2B SaaS company shipping at least one user-facing AI feature, '
        'or with one in active development.'
    ),
    target_roles=[
        'VP Product',
        'Head of AI / Head of ML',
        'Director of Engineering',
        'Founder or CTO with a technical background',
    ],
)
