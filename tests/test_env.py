"""Smoke tests for BoatLandingEnv. These are not full simulator tests —
they only verify the env constructs, resets, and steps without crashing,
and that the contract on obs/info is upheld."""

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from boat_landing.camera import CAMERA_HEIGHT, CAMERA_WIDTH  # noqa: E402
from boat_landing.env import BoatLandingEnv, load_scenario  # noqa: E402


SCENARIO_DIR = REPO_ROOT / "scenarios"


@pytest.fixture
def easy_env():
    env = BoatLandingEnv(str(SCENARIO_DIR / "easy.yaml"), gui=False)
    yield env
    env.close()


def _hover_action(env: BoatLandingEnv) -> np.ndarray:
    """Per-motor throttle that holds altitude under no perturbations.

    Uses the default attitude controller fed neutral setpoints — works
    for any drone spec without hardcoding the motor count or hover RPM.
    """
    from boat_landing.controllers import DefaultAttitudeController
    ctrl = DefaultAttitudeController(env.drone_spec)
    return ctrl(
        {"attitude": np.zeros(3), "angular_velocity": np.zeros(3)},
        roll_des=0.0, pitch_des=0.0, yaw_rate_des=0.0, thrust_norm=0.0,
    )


def test_load_scenario_returns_dict():
    cfg = load_scenario(SCENARIO_DIR / "easy.yaml")
    assert isinstance(cfg, dict)
    assert cfg["scenario_id"] == "easy"
    assert "drone_start" in cfg
    assert "boat" in cfg


def test_reset_returns_valid_obs(easy_env):
    obs, info = easy_env.reset(seed=0)
    assert "camera" in obs
    assert obs["camera"].shape == (CAMERA_HEIGHT, CAMERA_WIDTH, 3)
    assert obs["camera"].dtype == np.uint8
    assert set(obs["state"].keys()) == {
        "position",
        "velocity",
        "attitude",
        "angular_velocity",
    }
    assert obs["state"]["position"].shape == (3,)
    assert 0.0 <= obs["battery"] <= 1.0
    assert obs["time"] == 0.0
    assert "boat_position" in info


def test_obs_does_not_leak_boat_ground_truth(easy_env):
    """Critical contract: ground-truth boat pose stays in `info`, never `obs`."""
    obs, _ = easy_env.reset(seed=0)
    assert "boat_position" not in obs
    assert "boat_velocity" not in obs
    assert "boat_pose" not in obs


def test_step_returns_5_tuple(easy_env):
    easy_env.reset(seed=0)
    action = _hover_action(easy_env)
    result = easy_env.step(action)
    assert len(result) == 5
    obs2, reward, terminated, truncated, info = result
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "outcome" in info


def test_action_shape_is_validated(easy_env):
    easy_env.reset(seed=0)
    n = easy_env.drone_spec.num_motors
    # Wrong shape (one short) must raise.
    with pytest.raises(ValueError):
        easy_env.step(np.zeros(n - 1))


def test_action_values_are_motor_throttles_in_unit_interval(easy_env):
    """Action contract: shape (num_motors,), values in [0, 1]. Values
    outside the range are clipped silently (no exception)."""
    easy_env.reset(seed=0)
    n = easy_env.drone_spec.num_motors
    # Out-of-range action gets clipped, doesn't raise.
    out_of_range = np.full(n, 5.0)
    easy_env.step(out_of_range)


def test_step_runs_for_50_steps(easy_env):
    """50 steps == 1 simulated second of stable hover on EASY."""
    easy_env.reset(seed=0)
    action = _hover_action(easy_env)
    for _ in range(50):
        obs, _, terminated, truncated, _ = easy_env.step(action)
        if terminated or truncated:
            break
    assert easy_env.t > 0.9  # got essentially the full second


def test_termination_after_termination_raises(easy_env):
    easy_env.reset(seed=0)
    # Force a termination by setting the internal flag.
    easy_env._terminated_flag = True
    with pytest.raises(RuntimeError):
        easy_env.step(_hover_action(easy_env))
