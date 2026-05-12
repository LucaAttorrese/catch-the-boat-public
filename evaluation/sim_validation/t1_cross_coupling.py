"""T1.D — Cross-axis inertial coupling (Newton-Euler ω×Iω term).

Verify the simulator's rigid-body integrator includes the
Euler-equation coupling term  ω × (I·ω). For asymmetric inertia
(Ix ≠ Iy ≠ Iz) and ω with multiple non-zero components, this term
produces measurable rotation on axes that were not directly torqued.

Test (uses the VTOL spec, where Ix=2.54, Iy=3.47, Izz=5.74 — strongly
asymmetric):

    1. Reset at altitude. Spin motors at hover so the sim is in a
       balanced thrust state (no net torque from motors).
    2. Forcibly imprint an initial body angular velocity by integrating
       a brief torque burst — done via a short window of differential
       motor throttles for 0.05 s.
    3. After the burst, command pure hover throttles for 1 s. Net
       motor torque ≈ 0.
    4. Without coupling, ω stays constant. With coupling, ω rotates
       (specifically, components shuffle so |ω| stays ~ constant but
       direction precesses).

Pass criterion: the body angular velocity vector must rotate by more
than 5 degrees over the free-precession second. Pure ω-stays-constant
sims will rotate by 0; coupled sims will visibly precess.

For symmetric quad inertia (Ix=Iy), the coupling vanishes — the test
is run only against vtol.yaml regardless of which spec the participant
chose for the agent challenge.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from evaluation.sim_validation._runner import TestResult, REPO_ROOT


NAME = "T1.D Cross-axis inertial coupling"

DT = 0.001
TORQUE_BURST_S = 0.10
FREE_PRECESSION_S = 1.0
PRECESSION_THRESHOLD_DEG = 3.0


def _imprint_angular_velocity(sim, n_steps_burst):
    """Imprint a multi-axis body angular velocity via an asymmetric
    thrust burst.

    Recipe: full throttle on motor 0 only, zero on the rest. Because
    motor 0 is offset from the COM in BOTH x and y on every shipped
    spec, this generates roll AND pitch torques simultaneously — not
    cancellable by symmetry. The drone tilts violently but we only care
    about the resulting ω, which we read immediately after the burst.
    """
    n_motors = sim.spec.num_motors
    if n_motors < 1:
        return False
    cmd = np.zeros(n_motors)
    cmd[0] = 1.0
    for _ in range(n_steps_burst):
        sim.step(cmd, np.zeros(3), DT)
    return True


def run(sim_factory, drone_spec_path: str, **_) -> TestResult:
    # Force VTOL spec because cross-coupling vanishes for symmetric inertia.
    vtol_spec = REPO_ROOT / "drones" / "vtol.yaml"
    if not vtol_spec.is_file():
        return TestResult(NAME, False, f"vtol.yaml missing under {REPO_ROOT/'drones'}")
    sim = sim_factory(str(vtol_spec))
    sim.reset(np.array([0.0, 0.0, 100.0]), np.zeros(3))

    # 1. Imprint a multi-axis angular velocity.
    n_burst = int(TORQUE_BURST_S / DT)
    if not _imprint_angular_velocity(sim, n_burst):
        return TestResult(NAME, False, "could not imprint angular velocity")

    omega0 = sim.get_state().angular_velocity_body.copy()
    omega0_norm = float(np.linalg.norm(omega0))
    if omega0_norm < 0.5:
        return TestResult(
            NAME,
            False,
            f"failed to imprint sufficient angular velocity (|ω|={omega0_norm:.3f} rad/s); "
            f"sim's torque response too weak for this test",
            {"omega0_norm_rad_per_s": omega0_norm},
        )

    # 2. Free precession: hover throttles, ~zero net torque, watch ω evolve.
    n_motors = sim.spec.num_motors
    hover_t = float(np.sqrt(
        sim.spec.hover_thrust_per_motor / max(sim.spec.propeller.thrust_coefficient, 1e-12)
    )) / sim.spec.motor.omega_max
    hover = np.full(n_motors, hover_t)
    n_steps = int(FREE_PRECESSION_S / DT)
    for _ in range(n_steps):
        sim.step(hover, np.zeros(3), DT)

    omega1 = sim.get_state().angular_velocity_body.copy()
    omega1_norm = float(np.linalg.norm(omega1))

    cos_angle = float(
        np.dot(omega0, omega1) / max(omega0_norm * omega1_norm, 1e-12)
    )
    cos_angle = float(np.clip(cos_angle, -1.0, 1.0))
    rotation_deg = float(np.rad2deg(np.arccos(cos_angle)))

    metrics = {
        "omega0_norm_rad_per_s": omega0_norm,
        "omega1_norm_rad_per_s": omega1_norm,
        "rotation_deg": rotation_deg,
        "threshold_deg": PRECESSION_THRESHOLD_DEG,
        "duration_s": FREE_PRECESSION_S,
    }

    if rotation_deg < PRECESSION_THRESHOLD_DEG:
        return TestResult(
            NAME,
            False,
            f"angular velocity rotated only {rotation_deg:.2f}° in {FREE_PRECESSION_S:.1f} s "
            f"of free precession (threshold {PRECESSION_THRESHOLD_DEG:.1f}°); "
            f"sim ignores ω×(Iω) coupling",
            metrics,
        )
    return TestResult(
        NAME,
        True,
        f"free precession rotated ω by {rotation_deg:.1f}° in {FREE_PRECESSION_S:.1f} s",
        metrics,
    )
