import pytest

from services.scoring import (
    DEMO_ICP, arr_match, compute_action, headcount_match, stage_match,
)


def test_stage_match_in_target():
    assert stage_match('Series B', DEMO_ICP) == 1.0


def test_stage_match_one_step_off():
    assert stage_match('Seed', DEMO_ICP) == 0.5
    assert stage_match('Series D+', DEMO_ICP) == 0.5


def test_stage_match_two_steps_off():
    assert stage_match('Pre-seed', DEMO_ICP) == 0.25
    assert stage_match('Public', DEMO_ICP) == 0.25


def test_stage_match_unknown_string():
    assert stage_match('Bootstrapped', DEMO_ICP) == 0.0


@pytest.mark.parametrize('headcount,expected', [
    (20, 1.0), (140, 1.0), (250, 1.0),
    (17, 0.75), (310, 0.75),
    (14, 0.5), (370, 0.5),
    (11, 0.25), (490, 0.25),
    (5, 0.0), (1, 0.0), (600, 0.0),
])
def test_headcount_match_taper(headcount, expected):
    assert headcount_match(headcount, DEMO_ICP) == expected


def test_arr_match_missing_returns_zero():
    assert arr_match(None, DEMO_ICP) == 0.0


def test_compute_action_auto_add():
    assert compute_action(0.92, True, True) == 'auto_add'


def test_compute_action_propose_on_ungrounded():
    assert compute_action(0.95, False, True) == 'propose'


def test_compute_action_propose_in_band():
    assert compute_action(0.55, True, True) == 'propose'


def test_compute_action_discard_low_fit():
    assert compute_action(0.05, True, True) == 'discard'


def test_compute_action_refuse_insufficient():
    assert compute_action(0.92, True, False) == 'refuse'
