"""T0.4 — Spec-driven physics.

Verify the simulator's behaviour actually depends on the YAML spec
contents — i.e. it isn't hardcoding mass, inertia, or thrust constants.

Test: load the simulator twice, once with `drones/quadcopter.yaml`
(1.5 kg) and once with `drones/vtol.yaml` (10 kg). The analytic hover
throttle for these specs is very different (~63% vs ~71%) AND the
velocity reached after 0.5 s of free fall must reflect the different
masses (gravity is constant, so masses can't differ here — this test
specifically uses the per-motor THRUST output as the discriminator).

Specifically: with all motors commanded at full throttle (1.0) for
0.5 s starting from rest, the upward acceleration is
  a = (sum_T_max - mg) / m
For the quad (T_max = 4 * k_T * omega_max^2 = 4 * 7.6e-6 * 1100^2 ≈
36.8 N, m = 1.5 kg, T/W ≈ 2.5):  a ≈ 14.7 m/s²,  v(0.5s) ≈ 7.4 m/s.
For the VTOL (T_max ≈ 196 N, m = 10 kg, T/W ≈ 2.0):  a ≈ 9.8 m/s²,
  v(0.5s) ≈ 4.9 m/s.

We require the two velocities to differ by at least 30%. If they don't,
the sim is ignoring at least one of (mass, k_T, omega_max).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from evaluation.sim_validation._runner import TestResult, REPO_ROOT


NAME = "T0.4 Spec-driven physics"

DURATION_S = 0.5
DT = 0.004
RELATIVE_DIFFERENCE_THRESHOLD = 0.30  # 30% — well above noise floor


def _max_throttle_velocity(sim_factory, spec_path: Path, dt: float, n_steps: int) -> float:
    sim = sim_factory(str(spec_path))
    sim.reset(np.array([0.0, 0.0, 5.0]), np.zeros(3))
    n_motors = sim.spec.num_motors
    full = np.ones(n_motors)
    for _ in range(n_steps):
        sim.step(full, np.zeros(3), dt)
    return float(sim.get_state().velocity[2])  # vertical (up = positive)


def run(sim_factory, drone_spec_path: str, **_) -> TestResult:
    quad_spec = REPO_ROOT / "drones" / "quadcopter.yaml"
    vtol_spec = REPO_ROOT / "drones" / "vtol.yaml"
    if not quad_spec.is_file() or not vtol_spec.is_file():
        return TestResult(
            NAME,
            False,
            f"reference specs missing under {REPO_ROOT/'drones'}",
        )

    n_steps = int(DURATION_S / DT)
    try:
        v_quad = _max_throttle_velocity(sim_factory, quad_spec, DT, n_steps)
        v_vtol = _max_throttle_velocity(sim_factory, vtol_spec, DT, n_steps)
    except Exception as exc:
        return TestResult(NAME, False, f"sim raised on spec swap: {exc!r}")

    abs_diff = abs(v_quad - v_vtol)
    rel_diff = abs_diff / max(abs(v_quad), abs(v_vtol), 1e-9)
    metrics = {
        "v_quad_after_full_throttle_m_per_s": v_quad,
        "v_vtol_after_full_throttle_m_per_s": v_vtol,
        "relative_difference": rel_diff,
        "threshold": RELATIVE_DIFFERENCE_THRESHOLD,
    }

    if rel_diff < RELATIVE_DIFFERENCE_THRESHOLD:
        return TestResult(
            NAME,
            False,
            f"quad and VTOL produced near-identical climb velocities "
            f"({v_quad:.2f} vs {v_vtol:.2f} m/s, rel diff {rel_diff*100:.1f}%); "
            f"sim appears to be ignoring spec parameters",
            metrics,
        )
    return TestResult(
        NAME,
        True,
        f"quad climb {v_quad:.2f} m/s vs VTOL {v_vtol:.2f} m/s "
        f"(rel diff {rel_diff*100:.0f}%, threshold {RELATIVE_DIFFERENCE_THRESHOLD*100:.0f}%)",
        metrics,
    )
