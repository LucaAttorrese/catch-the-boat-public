"""T1.E — Internal sub-stepping (numerical-integration robustness).

Verify the simulator integrates with sufficient internal resolution to
handle a coarse outer dt without significant error growth.

Test:
    1. Run sim_a for one outer step of dt = 0.02 s with a stiff
       command profile (full throttle ramp).
    2. Run sim_b for 5 outer steps of dt = 0.004 s with the same
       commands (so 5× higher cadence on the integrator).
    3. Compare final positions / velocities.

If the sim sub-steps internally (e.g. breaks dt = 0.02 into 5 or more
sub-steps of 0.004 s or finer), both runs should converge to nearly
the same result. If the sim does ONE Euler step regardless of dt size,
the dt=0.02 run will accumulate noticeably more integration error.

Pass criterion: the position difference between the two runs is below
2 cm. The threshold is loose so that single-Euler-step sims fail
clearly while well-integrated sims pass with margin.

This test exists to reward sims that handle the full env physics rate
(250 Hz) AND additionally sub-step the stiff motor dynamics internally,
without forcing a specific implementation strategy.
"""

from __future__ import annotations

import numpy as np

from evaluation.sim_validation._runner import TestResult, safe_load_spec


NAME = "T1.E Internal sub-stepping"

POSITION_DIFF_TOLERANCE_M = 0.02
DT_COARSE = 0.02
DT_FINE = 0.004
N_OUTER_COARSE = 25      # 0.5 s of coarse stepping
SUBSTEPS_PER_OUTER = 5   # so DT_FINE * SUBSTEPS_PER_OUTER == DT_COARSE


def _run_step_input(sim_factory, drone_spec_path, dt, n_outer, substeps_per_outer):
    sim = sim_factory(drone_spec_path)
    sim.reset(np.array([0.0, 0.0, 100.0]), np.zeros(3))
    n_motors = sim.spec.num_motors
    full = np.ones(n_motors)
    # Each "outer" step is split into `substeps_per_outer` sim.step calls
    # of duration `dt`. Total wall time integrated = n_outer * substeps_per_outer * dt.
    for _ in range(n_outer):
        for _ in range(substeps_per_outer):
            sim.step(full, np.zeros(3), dt)
    return sim.get_state()


def run(sim_factory, drone_spec_path: str, **_) -> TestResult:
    spec = safe_load_spec(drone_spec_path)
    # Run A: coarse — N_OUTER_COARSE outer steps of dt=0.02, no internal substeps from us.
    state_coarse = _run_step_input(sim_factory, drone_spec_path, DT_COARSE, N_OUTER_COARSE, 1)
    # Run B: fine — same total time, but called at dt=0.004 (5× more outer calls).
    state_fine = _run_step_input(
        sim_factory, drone_spec_path, DT_FINE, N_OUTER_COARSE, SUBSTEPS_PER_OUTER
    )

    pos_diff = float(np.linalg.norm(state_coarse.position - state_fine.position))
    vel_diff = float(np.linalg.norm(state_coarse.velocity - state_fine.velocity))
    metrics = {
        "pos_diff_m": pos_diff,
        "vel_diff_m_per_s": vel_diff,
        "tolerance_m": POSITION_DIFF_TOLERANCE_M,
        "dt_coarse_s": DT_COARSE,
        "dt_fine_s": DT_FINE,
        "total_simulated_time_s": N_OUTER_COARSE * DT_COARSE,
    }

    if pos_diff > POSITION_DIFF_TOLERANCE_M:
        return TestResult(
            NAME,
            False,
            f"coarse vs fine integration diverged by {pos_diff*100:.2f} cm "
            f"(tolerance {POSITION_DIFF_TOLERANCE_M*100:.0f} cm); sim does not "
            f"sub-step internally — large dt accumulates Euler error",
            metrics,
        )
    return TestResult(
        NAME,
        True,
        f"coarse vs fine integration agree within {pos_diff*1000:.2f} mm "
        f"(tolerance {POSITION_DIFF_TOLERANCE_M*100:.0f} cm)",
        metrics,
    )
