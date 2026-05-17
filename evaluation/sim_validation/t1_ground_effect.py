"""T1.F — Ground effect (thrust amplification near a surface).

Verify the simulator amplifies rotor thrust when the drone is close to
a ground surface, the way real rotorcraft do due to air recirculation.

Physical model (Cheng-Frantz approximation, the textbook starting point):
    T_eff = T * (1 + a * (R / h)^2),      with R = rotor radius,
                                          h = altitude above surface,
                                          a ≈ 0.10-0.50 (literature range)

Test:
    1. Reset the drone exactly one rotor diameter (h = R) above a
       virtual ground surface at z = `GROUND_Z` (chosen to match the
       env's platform top, ~0.5 m).
    2. Apply the analytic hover throttle — i.e. the throttle that
       balances mg for the FREE-FLIGHT thrust calculation.
    3. Pass `ext_ground_z = GROUND_Z` to every `step()` call so the sim
       knows where the surface is.
    4. Integrate for 0.5 s.

Without ground effect: T = mg exactly, so the drone hovers (Δz ≈ 0).
With ground effect: T_eff > mg at h = R, so the drone climbs measurably.

Pass criterion: vertical drift > 5 cm in 0.5 s. With a = 0.05 and h = R
the multiplier is 1.05 → excess upward acceleration ≈ 0.05g ≈ 0.49 m/s²
→ Δz ≈ 0.5 · 0.49 · 0.5² ≈ 6 cm, comfortably above the threshold. A
sim that ignores `ext_ground_z` produces Δz ≈ 0 and fails the test.

This test is intentionally indifferent to the specific formula used —
it only checks that some altitude-dependent thrust amplification is
applied when the simulator receives `ext_ground_z`.
"""

from __future__ import annotations

import numpy as np

from evaluation.sim_validation._runner import (
    TestResult,
    make_hover_action,
    safe_load_spec,
)


NAME = "T1.F Ground effect"

GROUND_Z = 0.5         # matches env's platform top (BOAT_HEIGHT + PLATFORM_HEIGHT)
DT = 0.004
DURATION_S = 0.5
DRIFT_THRESHOLD_M = 0.05


def run(sim_factory, drone_spec_path: str, **_) -> TestResult:
    spec = safe_load_spec(drone_spec_path)
    R = float(spec.propeller.diameter) / 2.0
    if R <= 0:
        return TestResult(
            NAME,
            False,
            f"propeller.diameter <= 0 in spec; cannot place drone at h=R",
        )
    initial_z = GROUND_Z + R
    initial_pos = np.array([0.0, 0.0, initial_z], dtype=np.float64)

    sim = sim_factory(drone_spec_path)
    sim.reset(initial_pos, np.zeros(3))

    throttles = make_hover_action(spec)
    n_steps = int(DURATION_S / DT)
    for _ in range(n_steps):
        sim.step(throttles, np.zeros(3), DT, ext_ground_z=GROUND_Z)

    final_z = float(sim.get_state().position[2])
    dz = final_z - initial_z
    metrics = {
        "rotor_radius_m": R,
        "ground_z_m": GROUND_Z,
        "initial_z_m": initial_z,
        "final_z_m": final_z,
        "dz_m": dz,
        "threshold_m": DRIFT_THRESHOLD_M,
        "duration_s": DURATION_S,
    }

    if dz < DRIFT_THRESHOLD_M:
        return TestResult(
            NAME,
            False,
            f"hover throttle at h=R produced Δz={dz*100:.2f} cm in {DURATION_S} s "
            f"(threshold {DRIFT_THRESHOLD_M*100:.0f} cm); sim ignores ext_ground_z "
            f"and does not amplify thrust near surface",
            metrics,
        )
    return TestResult(
        NAME,
        True,
        f"ground effect detected: drone rose {dz*100:.1f} cm in {DURATION_S} s "
        f"with hover throttle at h=R",
        metrics,
    )
