"""Unit tests for the scorer. Tests the formula directly with known
inputs — no env, no agent."""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.scorer import (  # noqa: E402
    CRASH_PENALTY,
    compute_estimation_bonus,
    compute_estimation_rmse,
    compute_score,
)


def test_crash_returns_penalty():
    score, _ = compute_score(
        outcome="CRASHED",
        landing_position_error=None,
        time_to_land=10.0,
        duration_max=60.0,
        battery_remaining=0.5,
        max_descent_velocity=4.0,
    )
    assert score == CRASH_PENALTY


@pytest.mark.parametrize("outcome", ["TIMEOUT", "OUT_OF_BATTERY", "ABORTED"])
def test_zero_outcomes(outcome):
    score, _ = compute_score(
        outcome=outcome,
        landing_position_error=None,
        time_to_land=60.0,
        duration_max=60.0,
        battery_remaining=0.0,
        max_descent_velocity=0.5,
    )
    assert score == 0.0


def test_perfect_landing_max_score():
    """err=0, instant landing, full battery, soft descent, perfect estimate
    -> 50*1*1*1 + 5 + 15 = 70."""
    score, breakdown = compute_score(
        outcome="LANDED",
        landing_position_error=0.0,
        time_to_land=0.0,
        duration_max=60.0,
        battery_remaining=1.0,
        max_descent_velocity=0.2,
        estimation_rmse=0.0,
    )
    assert math.isclose(score, 70.0, rel_tol=1e-9)
    comps = breakdown["components"]
    assert math.isclose(comps["precision_factor"], 1.0)
    assert math.isclose(comps["time_factor"], 1.0)
    assert math.isclose(comps["battery_factor"], 1.0)
    assert comps["soft_landing_bonus"] == 5.0
    assert math.isclose(comps["estimation_bonus"], 15.0)


def test_landing_with_err_caps_precision():
    """err = 1.5 m -> precision_factor = 0."""
    score, _ = compute_score(
        outcome="LANDED",
        landing_position_error=1.5,
        time_to_land=10.0,
        duration_max=60.0,
        battery_remaining=1.0,
        max_descent_velocity=0.5,
    )
    # base * 0 * ... = 0 plus soft bonus 5
    assert math.isclose(score, 5.0, rel_tol=1e-9)


def test_time_factor_floor_at_half():
    """Even running the full duration -> time_factor floored at 0.5."""
    score, breakdown = compute_score(
        outcome="LANDED",
        landing_position_error=0.0,
        time_to_land=60.0,
        duration_max=60.0,
        battery_remaining=1.0,
        max_descent_velocity=0.2,
        estimation_rmse=None,
    )
    # time_factor floored at 0.5; battery_factor=1; precision=1; soft=5
    expected = 50.0 * 1.0 * 0.5 * 1.0 + 5.0
    assert math.isclose(score, expected)
    assert math.isclose(breakdown["components"]["time_factor"], 0.5)


def test_battery_factor_floor():
    score, breakdown = compute_score(
        outcome="LANDED",
        landing_position_error=0.0,
        time_to_land=0.0,
        duration_max=60.0,
        battery_remaining=0.05,
        max_descent_velocity=2.0,  # not soft; no soft bonus
    )
    # base=50, precision=1, time=1, battery floored at 0.3
    expected = 50.0 * 1.0 * 1.0 * 0.3
    assert math.isclose(score, expected)
    assert breakdown["components"]["soft_landing_bonus"] == 0.0


def test_estimation_bonus_endpoints():
    assert compute_estimation_bonus(None) == 0.0
    assert compute_estimation_bonus(0.0) == 15.0
    assert compute_estimation_bonus(2.0) == 0.0
    assert compute_estimation_bonus(5.0) == 0.0
    # Linear in between
    assert math.isclose(compute_estimation_bonus(1.0), 7.5)


def test_estimation_rmse_basic():
    log = [
        (np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])),
        (np.array([0.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])),
    ]
    rmse = compute_estimation_rmse(log)
    assert math.isclose(rmse, 1.0)


def test_estimation_rmse_empty_returns_none():
    assert compute_estimation_rmse([]) is None
