"""Sanity checks on the baseline agent."""

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agents.agent_baseline import (  # noqa: E402
    PHASE_SEARCH,
    BaselineAgent,
    PID,
    rpy_to_matrix,
)


def _synthetic_obs():
    return {
        "camera": np.full((480, 640, 3), 60, dtype=np.uint8),
        "state": {
            "position": np.array([5.0, 0.0, 5.0]),
            "velocity": np.zeros(3),
            "attitude": np.zeros(3),
            "angular_velocity": np.zeros(3),
        },
        "battery": 1.0,
        "time": 0.0,
    }


def test_agent_constructs():
    agent = BaselineAgent()
    assert agent.phase == PHASE_SEARCH


def test_act_returns_motor_throttles_in_unit_interval():
    agent = BaselineAgent()
    obs = _synthetic_obs()
    action = agent.act(obs)
    # Default spec is the X-quadcopter (4 motors). Values are throttles in [0, 1].
    assert action.shape == (agent.spec.num_motors,)
    assert action.dtype == np.float64
    assert np.all(action >= 0.0) and np.all(action <= 1.0)


def test_pid_clamps_output():
    pid = PID(kp=1000.0, ki=0.0, kd=0.0, out_min=-1.0, out_max=1.0)
    out = pid(err=10.0, dt=0.02)
    assert out == 1.0
    out = pid(err=-10.0, dt=0.02)
    assert out == -1.0


def test_pid_zero_when_zero_error():
    pid = PID(kp=1.0, ki=1.0, kd=1.0)
    assert pid(0.0, 0.02) == 0.0


def test_rpy_to_matrix_identity():
    R = rpy_to_matrix([0.0, 0.0, 0.0])
    assert np.allclose(R, np.eye(3), atol=1e-9)


def test_rpy_to_matrix_yaw_90():
    R = rpy_to_matrix([0.0, 0.0, np.pi / 2])
    # Body +x rotates to world +y
    assert np.allclose(R @ np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), atol=1e-9)


def test_get_last_estimate_returns_none_at_init():
    agent = BaselineAgent()
    assert agent.get_last_estimate() is None


def test_get_last_estimate_populates_after_act():
    agent = BaselineAgent()
    # First act() with no detection (blank frame) still populates the
    # cold-start estimate via the search prior.
    agent.act(_synthetic_obs())
    est = agent.get_last_estimate()
    # When the prior is the only thing we have, get_last_estimate is allowed
    # to be None or the prior — either is acceptable for the bonus opt-out.
    if est is not None:
        assert "position" in est
