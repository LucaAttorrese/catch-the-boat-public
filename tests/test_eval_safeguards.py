"""Safeguards: wall-clock cap + exception catcher.

These tests verify that the eval pipeline never returns garbage data
even when a team's agent misbehaves. The two failure modes covered:

    1. Agent's act() raises an unhandled exception
       → outcome = "ERROR", score = 0
    2. Agent's act() takes too long (or env loops forever)
       → outcome = "WALL_TIMEOUT", score = 0

If either of these tests fails, the eval pipeline is no longer
crash-safe and one buggy submission could halt the whole scoring run.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agents.drone_sim_baseline import BaselineDroneSimulator  # noqa: E402
from boat_landing.env import BoatLandingEnv  # noqa: E402
from evaluation.evaluate import run_episode_safe  # noqa: E402


SCENARIO = REPO_ROOT / "scenarios" / "easy.yaml"
SPEC = REPO_ROOT / "drones" / "vtol.yaml"


def _make_env():
    sim = BaselineDroneSimulator(str(SPEC))
    return BoatLandingEnv(str(SCENARIO), drone_sim=sim, gui=False)


class _CrashAgent:
    """Agent whose act() raises immediately. Mirrors a team submission
    with a buggy perception pipeline (IndexError, LinAlgError, etc.)."""

    def act(self, obs):  # noqa: D401 — short
        raise IndexError("simulated bug: index 3 is out of bounds for axis 0 with size 3")


class _SlowAgent:
    """Agent whose act() sleeps long enough to trip the wall cap."""

    def __init__(self, sleep_s: float = 0.5):
        self.sleep_s = float(sleep_s)
        self.spec = None  # not used here

    def act(self, obs):
        time.sleep(self.sleep_s)
        # Return a hover throttle so the env doesn't reject the action.
        return np.full(4, 0.5, dtype=np.float64)


def test_crash_agent_produces_error_result():
    """An agent that raises must yield outcome=ERROR, score=0, and a
    structured error_type field — never a Python traceback to stdout."""
    env = _make_env()
    try:
        result = run_episode_safe(env, _CrashAgent(), seed=42, progress=False)
    finally:
        env.close()
    assert result["outcome"] == "ERROR", result
    assert result["score"] == 0.0, result
    assert result["error_type"] == "IndexError", result
    assert "index 3 is out of bounds" in result["error_message"], result


def test_slow_agent_hits_wall_cap():
    """An agent slow enough to exceed wall_cap_s should yield
    outcome=WALL_TIMEOUT, score=0, and the run must terminate within
    a small margin of the cap (no runaway loop)."""
    env = _make_env()
    cap = 1.0
    t0 = time.perf_counter()
    try:
        # 0.5 s per step × 50 Hz × 60 s sim = ~25 minutes wall without cap.
        # With cap = 1 s we expect to abort after ~2-3 act() calls.
        result = run_episode_safe(
            env, _SlowAgent(sleep_s=0.5), seed=42, progress=False, wall_cap_s=cap,
        )
    finally:
        env.close()
    elapsed = time.perf_counter() - t0
    assert result["outcome"] == "WALL_TIMEOUT", result
    assert result["score"] == 0.0, result
    # Allow some slack: cap=1s + a few hundred ms for the in-flight act().
    assert elapsed < cap + 1.5, f"Cap not enforced; elapsed={elapsed:.2f}s"
