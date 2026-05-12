"""T0.3 — Determinism.

Construct two independent simulator instances from the same spec, feed
each the same sequence of motor commands, and verify their final states
match to within numerical noise (~ 1e-10).

A non-deterministic sim cannot be fairly evaluated: scenarios run with a
fixed seed must produce the same trajectory every time. The most common
sources of non-determinism are: unseeded RNGs inside the sim (e.g.
"random vibration"), floating-point ordering changes from parallelism,
and reliance on wall-clock time. None of these are appropriate for a
physics simulator at this stage.
"""

from __future__ import annotations

import numpy as np

from evaluation.sim_validation._runner import TestResult, safe_load_spec


NAME = "T0.3 Determinism"

N_STEPS = 200
DT = 0.004
ACTION_SEED = 42
POSITION_TOL = 1.0e-10
QUAT_TOL = 1.0e-10


def run(sim_factory, drone_spec_path: str, **_) -> TestResult:
    spec = safe_load_spec(drone_spec_path)

    rng = np.random.default_rng(ACTION_SEED)
    actions = rng.uniform(0.3, 0.9, size=(N_STEPS, spec.num_motors))

    sim_a = sim_factory(drone_spec_path)
    sim_b = sim_factory(drone_spec_path)
    sim_a.reset(np.zeros(3), np.zeros(3))
    sim_b.reset(np.zeros(3), np.zeros(3))

    for k in range(N_STEPS):
        sim_a.step(actions[k], np.zeros(3), DT)
        sim_b.step(actions[k], np.zeros(3), DT)

    sa = sim_a.get_state()
    sb = sim_b.get_state()
    dpos = float(np.linalg.norm(sa.position - sb.position))
    dvel = float(np.linalg.norm(sa.velocity - sb.velocity))
    dquat = float(np.linalg.norm(sa.quaternion - sb.quaternion))
    metrics = {
        "pos_diff_m": dpos,
        "vel_diff_m_per_s": dvel,
        "quat_diff": dquat,
        "tol_pos_m": POSITION_TOL,
        "tol_quat": QUAT_TOL,
    }

    if dpos > POSITION_TOL or dquat > QUAT_TOL:
        return TestResult(
            NAME,
            False,
            f"two instances diverged: |Δpos|={dpos:.3e} m, |Δquat|={dquat:.3e}; "
            f"check for unseeded randomness or wall-clock dependencies",
            metrics,
        )
    return TestResult(
        NAME, True, f"two instances byte-equal (|Δpos|={dpos:.1e} m)", metrics
    )
