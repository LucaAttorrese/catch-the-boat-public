"""Smoke tests for the simulator-validation suite.

These are *meta*-tests: they run each validation module against the
reference baseline simulator and assert that the documented outcomes
(pass / fail) actually match. If you change a test's pass criterion,
update the corresponding assertion here.

The baseline sim implements:
    * First-order motor lag           → expected PASS on T1.A
    * Newton-Euler ω×(I·ω) coupling   → expected PASS on T1.D
    * Light internal sub-stepping     → expected PASS on T0.5 + T1.E
And does NOT implement:
    * Battery sag                     → expected FAIL on T1.B
    * Aero drag                       → expected FAIL on T1.C
    * Ground effect                   → expected FAIL on T1.F
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.sim_validation import _runner  # noqa: E402
from evaluation.sim_validation import (  # noqa: E402
    t0_protocol,
    t0_hover_steady,
    t0_determinism,
    t0_spec_driven,
    t0_variable_dt,
    t1_motor_lag,
    t1_battery_sag,
    t1_aero_drag,
    t1_cross_coupling,
    t1_substepping,
    t1_ground_effect,
)


BASELINE_SIM_PATH = REPO_ROOT / "agents" / "drone_sim_baseline.py"
QUAD_SPEC_PATH = REPO_ROOT / "drones" / "quadcopter.yaml"


def _factory():
    return _runner.load_sim_factory(str(BASELINE_SIM_PATH))


# --- Tier 0 (gate) — baseline must pass all four ---------------------------

def test_t0_protocol_passes_for_baseline():
    r = t0_protocol.run(_factory(), str(QUAD_SPEC_PATH))
    assert r.passed, f"{r.name} failed: {r.message}"


def test_t0_hover_steady_passes_for_baseline():
    r = t0_hover_steady.run(_factory(), str(QUAD_SPEC_PATH))
    assert r.passed, f"{r.name} failed: {r.message}"


def test_t0_determinism_passes_for_baseline():
    r = t0_determinism.run(_factory(), str(QUAD_SPEC_PATH))
    assert r.passed, f"{r.name} failed: {r.message}"


def test_t0_spec_driven_passes_for_baseline():
    r = t0_spec_driven.run(_factory(), str(QUAD_SPEC_PATH))
    assert r.passed, f"{r.name} failed: {r.message}"


def test_t0_variable_dt_passes_for_baseline():
    """T0.5 with public defaults. Baseline's 5 ms cap on internal
    integration step should keep it within the 5 cm threshold.
    Eval-time constants (set via env vars) may be stricter."""
    r = t0_variable_dt.run(_factory(), str(QUAD_SPEC_PATH))
    assert r.passed, f"{r.name} failed: {r.message}"


# --- Tier 1 — features the baseline implements should PASS ----------------

def test_t1_motor_lag_passes_for_baseline():
    r = t1_motor_lag.run(_factory(), str(QUAD_SPEC_PATH))
    assert r.passed, f"{r.name} failed: {r.message}"


def test_t1_cross_coupling_passes_for_baseline():
    r = t1_cross_coupling.run(_factory(), str(QUAD_SPEC_PATH))
    assert r.passed, f"{r.name} failed: {r.message}"


def test_t1_substepping_passes_for_baseline():
    """Baseline does light internal sub-stepping (5 ms cap) so it
    passes T1.E for free. This is what claims `substepping=true` in
    submission_baseline.yaml."""
    r = t1_substepping.run(_factory(), str(QUAD_SPEC_PATH))
    assert r.passed, f"{r.name} failed: {r.message}"


# --- Tier 1 — features the baseline does NOT implement should FAIL --------

def test_t1_battery_sag_fails_for_baseline():
    r = t1_battery_sag.run(_factory(), str(QUAD_SPEC_PATH))
    assert not r.passed, (
        f"{r.name} unexpectedly passed for baseline (which doesn't model sag): {r.message}"
    )


def test_t1_aero_drag_fails_for_baseline():
    r = t1_aero_drag.run(_factory(), str(QUAD_SPEC_PATH))
    assert not r.passed, (
        f"{r.name} unexpectedly passed for baseline (vacuum sim): {r.message}"
    )


def test_t1_ground_effect_fails_for_baseline():
    r = t1_ground_effect.run(_factory(), str(QUAD_SPEC_PATH))
    assert not r.passed, (
        f"{r.name} unexpectedly passed for baseline (which ignores ext_ground_z): {r.message}"
    )
