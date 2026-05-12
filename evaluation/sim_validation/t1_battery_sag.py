"""T1.B — Battery voltage sag under sustained load.

Verify the simulator models the battery's internal resistance: under
sustained high-power demand (full throttle), the achievable max RPM
should drop measurably as battery voltage sags below nominal.

Test:
    1. Reset and let the sim settle to hover briefly.
    2. Command full throttle for 20 s straight (continuous load).
    3. Measure the steady-state motor omega at the end.
    4. Compare to omega_max from the spec.

Pass criterion: omega_after_sustained_load / omega_max < 0.95. That is,
sustained max throttle reduces the achievable RPM by more than 5%.

This test is loose because individual sims will model sag with very
different aggressiveness depending on `battery.internal_resistance`,
how they relate voltage to omega_max (V/Kv), and whether they model
discharge curves. Anything noticeable beyond instantaneous full RPM is
accepted.

Requires `state.motor_omegas` to be exposed.
"""

from __future__ import annotations

import numpy as np

from evaluation.sim_validation._runner import TestResult, safe_load_spec


NAME = "T1.B Battery voltage sag"

DT = 0.004
LOAD_DURATION_S = 20.0
SAG_THRESHOLD = 0.95  # achievable / max must be below this


def run(sim_factory, drone_spec_path: str, **_) -> TestResult:
    spec = safe_load_spec(drone_spec_path)
    sim = sim_factory(drone_spec_path)
    sim.reset(np.array([0.0, 0.0, 100.0]), np.zeros(3))

    n_motors = spec.num_motors
    omega_max = float(spec.motor.omega_max)

    # 0.5 s warm-up at full throttle to settle motor RPM.
    full = np.ones(n_motors)
    for _ in range(int(0.5 / DT)):
        sim.step(full, np.zeros(3), DT)

    state0 = sim.get_state()
    if state0.motor_omegas is None:
        return TestResult(
            NAME,
            False,
            "sim.get_state().motor_omegas is None — sim must track per-motor RPM "
            "to claim T1.B",
        )
    omega_initial = float(state0.motor_omegas[0])

    # Sustained full throttle for LOAD_DURATION_S.
    for _ in range(int(LOAD_DURATION_S / DT)):
        sim.step(full, np.zeros(3), DT)

    state1 = sim.get_state()
    omega_after = float(state1.motor_omegas[0])

    ratio_to_max = omega_after / omega_max
    ratio_to_initial = omega_after / max(omega_initial, 1e-9)
    metrics = {
        "omega_max_spec_rad_per_s": omega_max,
        "omega_initial_rad_per_s": omega_initial,
        "omega_after_load_rad_per_s": omega_after,
        "ratio_after_to_max": ratio_to_max,
        "ratio_after_to_initial": ratio_to_initial,
        "load_duration_s": LOAD_DURATION_S,
        "threshold": SAG_THRESHOLD,
    }

    if ratio_to_max >= SAG_THRESHOLD:
        return TestResult(
            NAME,
            False,
            f"after {LOAD_DURATION_S:.0f} s at full throttle, ω={omega_after:.0f} rad/s "
            f"({ratio_to_max*100:.1f}% of ω_max); no measurable sag — "
            f"sim does not model battery internal resistance / discharge",
            metrics,
        )
    return TestResult(
        NAME,
        True,
        f"battery sag detected: ω dropped to {ratio_to_max*100:.1f}% of ω_max "
        f"after {LOAD_DURATION_S:.0f} s sustained full throttle",
        metrics,
    )
