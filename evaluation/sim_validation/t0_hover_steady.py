"""T0.2 — Hover steady-state.

With the drone level, motors at the analytic hover throttle, no wind,
no perturbation: integrate for 5 simulated seconds at 250 Hz and
verify the drone has not drifted by more than 1 cm.

This is a pure static-balance test. It does NOT require an attitude
controller — the inputs are constant per-motor throttles. A correctly
implemented sim with symmetric motor placement will see zero net torque
and zero net horizontal force, and gravity will be balanced exactly by
the hover thrust. Any drift larger than 1 cm in 5 s indicates either
the static thrust calculation is wrong (T = k_T * omega^2 mismatched)
or motor RPMs were not initialised to hover at reset() time.
"""

from __future__ import annotations

import numpy as np

from evaluation.sim_validation._runner import TestResult, make_hover_action, safe_load_spec


NAME = "T0.2 Hover steady-state"

DURATION_S = 5.0
DT = 1.0 / 250.0
DRIFT_TOLERANCE_M = 0.01
INITIAL_POS = np.array([0.0, 0.0, 5.0], dtype=np.float64)


def run(sim_factory, drone_spec_path: str, **_) -> TestResult:
    spec = safe_load_spec(drone_spec_path)
    sim = sim_factory(drone_spec_path)
    sim.reset(INITIAL_POS, np.zeros(3))

    throttles = make_hover_action(spec)
    n_steps = int(DURATION_S / DT)
    for _ in range(n_steps):
        sim.step(throttles, np.zeros(3), DT)

    state = sim.get_state()
    drift = float(np.linalg.norm(state.position - INITIAL_POS))
    speed = float(np.linalg.norm(state.velocity))
    tilt = float(np.linalg.norm(state.angular_velocity_body))
    metrics = {
        "drift_m": drift,
        "speed_m_per_s": speed,
        "ang_speed_rad_per_s": tilt,
        "tolerance_m": DRIFT_TOLERANCE_M,
    }

    if drift > DRIFT_TOLERANCE_M:
        return TestResult(
            NAME,
            False,
            f"drift={drift*100:.2f} cm in {DURATION_S} s exceeds tolerance "
            f"{DRIFT_TOLERANCE_M*100:.0f} cm; check hover thrust calc and "
            f"motor RPM initialization in reset()",
            metrics,
        )
    return TestResult(
        NAME, True, f"drift={drift*1000:.2f} mm in {DURATION_S} s", metrics
    )
