"""T1.A — Motor RPM lag (first-order response).

Verify the simulator models the motor as having finite spool-up time
rather than instantaneous RPM response. With the drone clamped (we just
read `motor_omegas` without integrating attitude), command a step from
0 throttle to full throttle and measure the time for omega to reach
~63% of its steady-state value (the time constant for a first-order
response).

The measured tau must lie within [0.5, 2.0] times the spec's nominal
`motor.time_constant`. This wide band is intentional: a sim that
implements a richer BLDC model (V→I→torque) won't exactly match a
first-order response but should still respect the time scale.

Requires the simulator to expose `state.motor_omegas`. A sim that does
not track per-motor RPMs (e.g. instant-response model) will return
None for that field and fail this test.
"""

from __future__ import annotations

import numpy as np

from evaluation.sim_validation._runner import TestResult, safe_load_spec


NAME = "T1.A Motor RPM lag (first-order)"

DT = 0.001  # 1 kHz to resolve fast motor dynamics
TAU_TOLERANCE_LOW = 0.5
TAU_TOLERANCE_HIGH = 2.0
TARGET_FRACTION = 1.0 - 1.0 / np.e  # ≈ 0.632


def run(sim_factory, drone_spec_path: str, **_) -> TestResult:
    spec = safe_load_spec(drone_spec_path)
    expected_tau = float(spec.motor.time_constant)
    if expected_tau <= 0:
        return TestResult(
            NAME, False, "spec.motor.time_constant <= 0; sim has no defined motor lag"
        )

    sim = sim_factory(drone_spec_path)
    sim.reset(np.array([0.0, 0.0, 100.0]), np.zeros(3))  # high altitude so we don't worry about ground

    # Force RPMs to ~0 with a few zero-throttle steps before the step input.
    n_motors = spec.num_motors
    for _ in range(int(0.5 / DT)):
        sim.step(np.zeros(n_motors), np.zeros(3), DT)

    omega_max = float(spec.motor.omega_max)
    target_omega = TARGET_FRACTION * omega_max

    # Step input: full throttle, all motors.
    full = np.ones(n_motors)
    n_window = int(5.0 * expected_tau / DT) + 100  # several time constants
    times = []
    omegas = []
    for k in range(n_window):
        sim.step(full, np.zeros(3), DT)
        state = sim.get_state()
        if state.motor_omegas is None:
            return TestResult(
                NAME,
                False,
                "sim.get_state().motor_omegas is None — sim must track per-motor RPM "
                "to claim T1.A",
            )
        times.append((k + 1) * DT)
        omegas.append(float(state.motor_omegas[0]))

    omegas_arr = np.asarray(omegas)
    times_arr = np.asarray(times)

    # Steady-state value reached at the end of the window.
    omega_steady = float(np.mean(omegas_arr[-20:]))
    if omega_steady < 0.5 * omega_max:
        return TestResult(
            NAME,
            False,
            f"motor never reached half of omega_max: steady={omega_steady:.0f} rad/s, "
            f"omega_max={omega_max:.0f} rad/s",
        )

    # Find first time omega crosses TARGET_FRACTION * steady-state.
    target = TARGET_FRACTION * omega_steady
    idx = int(np.searchsorted(omegas_arr, target))
    if idx <= 0 or idx >= len(times_arr):
        return TestResult(
            NAME, False, f"could not bracket τ from rise; omega_steady={omega_steady:.0f}"
        )
    # Linear interpolation between idx-1 and idx
    o0, o1 = omegas_arr[idx - 1], omegas_arr[idx]
    t0, t1 = times_arr[idx - 1], times_arr[idx]
    measured_tau = float(t0 + (target - o0) * (t1 - t0) / max(o1 - o0, 1e-12))

    metrics = {
        "expected_tau_s": expected_tau,
        "measured_tau_s": measured_tau,
        "ratio": measured_tau / expected_tau,
        "omega_steady_rad_per_s": omega_steady,
        "tol_low": TAU_TOLERANCE_LOW,
        "tol_high": TAU_TOLERANCE_HIGH,
    }
    ratio = measured_tau / expected_tau
    if not (TAU_TOLERANCE_LOW <= ratio <= TAU_TOLERANCE_HIGH):
        return TestResult(
            NAME,
            False,
            f"measured τ={measured_tau*1000:.1f} ms vs expected {expected_tau*1000:.1f} ms "
            f"(ratio {ratio:.2f} outside [{TAU_TOLERANCE_LOW:.1f}, {TAU_TOLERANCE_HIGH:.1f}])",
            metrics,
        )
    return TestResult(
        NAME,
        True,
        f"measured τ={measured_tau*1000:.1f} ms (ratio {ratio:.2f} of spec)",
        metrics,
    )
