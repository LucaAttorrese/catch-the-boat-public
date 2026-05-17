"""T1.C — Aerodynamic drag (linear and/or quadratic).

Verify the simulator applies a body-velocity-dependent drag force
distinguishable from the no-drag (vacuum) reference.

Test:
    1. Reset at altitude with zero velocity, level attitude.
    2. Command zero throttle (free fall).
    3. Integrate for 4 s.
    4. Compare measured downward velocity to the analytic no-drag
       value v_no_drag = -g * t = -39.24 m/s.

Pass criterion:
    |measured - v_no_drag| / |v_no_drag|  >  0.02
That is, drag must slow the fall by more than 2% relative to vacuum.

Why this works: with zero throttle, no thrust at steady-state, no wind,
the only forces are gravity (always present) and drag (the feature under
test). If `aero.drag_*` are zero or unused, the sim matches free-fall
within the small residual motor-spin-down impulse. If drag is active,
the measured velocity at 4 s is strictly less in magnitude.

The 4 s duration is chosen so the (mostly-fixed) residual impulse from
the first-order motor decay during the first ~0.5 s amortises below the
2 % threshold, while a real drag force keeps growing with |v|. A sim
that resets motors to hover RPM (the baseline does, to pass T0.2) will
not falsely pass this test thanks to the long enough window.

NB: this test does NOT verify the drag formula is correct, only that
SOME velocity-dependent damping is applied.
"""

from __future__ import annotations

import numpy as np

from evaluation.sim_validation._runner import TestResult, safe_load_spec


NAME = "T1.C Aerodynamic drag present"

DT = 0.004
DURATION_S = 4.0
GRAVITY = 9.81
RELATIVE_DEVIATION_THRESHOLD = 0.02  # 2% slower than no-drag


def run(sim_factory, drone_spec_path: str, **_) -> TestResult:
    spec = safe_load_spec(drone_spec_path)
    sim = sim_factory(drone_spec_path)
    sim.reset(np.array([0.0, 0.0, 100.0]), np.zeros(3))

    n_motors = spec.num_motors
    zero = np.zeros(n_motors)
    n_steps = int(DURATION_S / DT)
    for _ in range(n_steps):
        sim.step(zero, np.zeros(3), DT)

    v_z = float(sim.get_state().velocity[2])
    v_no_drag = -GRAVITY * (n_steps * DT)
    deviation = abs(v_z - v_no_drag) / abs(v_no_drag)
    metrics = {
        "v_z_measured_m_per_s": v_z,
        "v_z_no_drag_m_per_s": v_no_drag,
        "relative_deviation": deviation,
        "threshold": RELATIVE_DEVIATION_THRESHOLD,
        "duration_s": DURATION_S,
    }

    if v_z < v_no_drag - 1.0:
        # Sim went faster than vacuum gravity — physics broken (or extra force pulling down).
        return TestResult(
            NAME,
            False,
            f"sim fall ({v_z:.2f} m/s) exceeds no-drag prediction ({v_no_drag:.2f} m/s); "
            f"check gravity application",
            metrics,
        )
    if deviation < RELATIVE_DEVIATION_THRESHOLD:
        return TestResult(
            NAME,
            False,
            f"free-fall matched vacuum to within {deviation*100:.2f}% "
            f"({v_z:.3f} vs {v_no_drag:.3f} m/s); aero.drag_* not applied",
            metrics,
        )
    return TestResult(
        NAME,
        True,
        f"drag detected: free-fall {v_z:.2f} m/s vs vacuum {v_no_drag:.2f} m/s "
        f"({deviation*100:.1f}% deviation)",
        metrics,
    )
