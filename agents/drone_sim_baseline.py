"""Reference drone simulator — tier 0 + light sub-stepping.

This is the minimum-viable implementation of the DroneSimulator protocol.
It covers what the challenge classifies as "tier 0" (gate tests, incl.
T0.5 variable-dt robustness, which forces it to also do *some* internal
sub-stepping) plus the two Tier-1 features that come for free with an
honest motor + Newton-Euler model. Forking and extending this file is
the intended starting point if you're focused on perception/control.

What's implemented (tier 0 + light extras)
------------------------------------------
* First-order motor RPM dynamics (single time constant per motor):
      omega_dot = (omega_cmd - omega) / tau
  Claims T1.A (motor_lag).
* Static propeller model:
      T = k_T * omega^2     (per motor, along motor.thrust_axis)
      Q = k_Q * omega^2     (per motor, drag torque, sign from motor.spin)
* Body force/torque assembly (mixer):
      F_body  = sum(T_i * axis_i)
      tau_body = sum(r_i x F_i) + sum(-spin_i * Q_i * axis_i)
* Rigid body 6DoF Newton-Euler:
      m * a_world = R(q) * F_body + F_gravity + F_ext
      I * omega_dot_body = tau_body - omega_body x (I * omega_body)
  Claims T1.D (cross_coupling, since the omega × Iω term is kept).
* Quaternion attitude integration with renormalization.
* Gravity applied internally (mass is known).
* Internal sub-stepping capped at MAX_INTERNAL_DT_S = 5 ms. Required
  to pass T0.5 (variable-dt robustness gate) and to score T1.E
  (substepping) on the Tier-1 suite.

Baseline scores 15 / 30: motor_lag (T1.A) + cross_coupling (T1.D) +
substepping (T1.E). Beat it by adding aero_drag, battery_sag, or
ground_effect to your fork.

What's NOT implemented (tier 1+ — adding any of these earns sim-quality
bonus, provided the corresponding validation test passes):

* Full BLDC electromechanical motor model (V -> I -> torque -> omega)
  using motor.rotor_inertia and battery.internal_resistance.
* Battery voltage sag under load (omega_max drops with V_bat).
* Linear + quadratic aerodynamic drag (uses spec.aero.drag_*).
* Rotor gyroscopic torque on the body (visible during fast yaw).
* Ground effect on thrust near the platform.
* VTOL-specific: tilt-servo dynamics, hover<->cruise transition logic.

Structure
---------
The class is intentionally split into small methods so a fork can replace
just the part it cares about (e.g. swap `_motor_dynamics_step` for a
BLDC model without touching the rigid-body integrator).
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from boat_landing.drone_interface import DroneSimulator, DroneSpec, DroneState, load_drone_spec


GRAVITY = 9.81

# Largest integration step the baseline takes internally. Any incoming
# `dt` above this is split into chunks of `MAX_INTERNAL_DT_S` or less
# before integrating. With τ_motor ≈ 50 ms and the env's 250 Hz cadence
# the baseline integrates at dt = 4 ms by default, so this only kicks
# in when callers pass larger dt — e.g. the T0.5 gate test does.
MAX_INTERNAL_DT_S = 0.005


class BaselineDroneSimulator:
    """Reference DroneSimulator implementation. Tier 0 only."""

    def __init__(self, spec_path: str):
        self.spec: DroneSpec = load_drone_spec(spec_path)

        # Pre-extract arrays for speed in the inner loop.
        self._motor_positions = np.stack([m.position for m in self.spec.motors])     # (N, 3)
        self._motor_axes = np.stack([m.thrust_axis for m in self.spec.motors])       # (N, 3) unit
        self._motor_spins = np.array([m.spin for m in self.spec.motors], dtype=np.float64)
        self._inv_inertia = np.linalg.inv(self.spec.inertia)
        self._mass = float(self.spec.mass)
        self._k_T = float(self.spec.propeller.thrust_coefficient)
        self._k_Q = float(self.spec.propeller.drag_coefficient)
        self._omega_max = float(self.spec.motor.omega_max)
        self._tau_motor = float(self.spec.motor.time_constant)

        # Mutable state — initialized in reset().
        self._position = np.zeros(3, dtype=np.float64)
        self._velocity = np.zeros(3, dtype=np.float64)
        self._quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)  # (x, y, z, w)
        self._omega_body = np.zeros(3, dtype=np.float64)
        self._motor_omegas = np.zeros(self.spec.num_motors, dtype=np.float64)

    # ------------------------------------------------------------------ public
    def reset(self, position: np.ndarray, attitude: np.ndarray) -> None:
        self._position = np.asarray(position, dtype=np.float64).copy()
        self._velocity = np.zeros(3, dtype=np.float64)
        self._quat = self._coerce_attitude(attitude)
        self._omega_body = np.zeros(3, dtype=np.float64)
        # Spin motors up to hover so we don't drop on the first step.
        self._motor_omegas = np.full(
            self.spec.num_motors, self._hover_omega(), dtype=np.float64
        )

    def step(
        self,
        motor_cmds: np.ndarray,
        ext_force_world: np.ndarray,
        dt: float,
        ext_torque_world: Optional[np.ndarray] = None,
        ext_ground_z: Optional[float] = None,
    ) -> None:
        # ext_ground_z is the world-frame z of the surface below the drone,
        # used by tier-1 ground-effect models. The baseline does NOT model
        # ground effect, so this argument is accepted for protocol
        # compatibility and ignored. To pick up the +5 T1.F bonus, fork
        # this method and amplify per-motor thrust by
        #   1 + a * (R / max(z - ext_ground_z, eps))^2
        # before summing into F_body.
        del ext_ground_z

        motor_cmds = np.clip(np.asarray(motor_cmds, dtype=np.float64), 0.0, 1.0)
        if motor_cmds.shape != (self.spec.num_motors,):
            raise ValueError(
                f"motor_cmds shape {motor_cmds.shape} != ({self.spec.num_motors},)"
            )
        ext_force_world = np.asarray(ext_force_world, dtype=np.float64).reshape(3)
        if ext_torque_world is None:
            ext_torque_world = np.zeros(3, dtype=np.float64)
        else:
            ext_torque_world = np.asarray(ext_torque_world, dtype=np.float64).reshape(3)

        # Cap the integration step at MAX_INTERNAL_DT_S. For the env's
        # default 250 Hz cadence (dt = 4 ms) this is a single iteration;
        # for callers that pass larger dt (e.g. the T0.5 gate test up
        # to 50 ms) we sub-step so Euler error stays bounded.
        n_sub = max(1, int(np.ceil(dt / MAX_INTERNAL_DT_S)))
        sub_dt = dt / n_sub
        for _ in range(n_sub):
            self._integrate_substep(motor_cmds, ext_force_world, ext_torque_world, sub_dt)

    def _integrate_substep(
        self,
        motor_cmds: np.ndarray,
        ext_force_world: np.ndarray,
        ext_torque_world: np.ndarray,
        sub_dt: float,
    ) -> None:
        # 1. Motor RPM dynamics (exact integration of first-order lag)
        self._motor_dynamics_step(motor_cmds, sub_dt)

        # 2. Body-frame force and torque from current motor RPMs
        F_body, tau_body = self._compute_motor_wrench()

        # 3. Body -> world rotation
        R = self._rotation_matrix(self._quat)

        # 4. Total world-frame force on body
        F_world = R @ F_body
        F_world += np.array([0.0, 0.0, -self._mass * GRAVITY])
        F_world += ext_force_world

        # 5. Total body-frame torque (external world torque rotated into body)
        tau_body_total = tau_body + R.T @ ext_torque_world

        # 6. Linear acceleration (world)
        a_world = F_world / self._mass

        # 7. Angular acceleration (body): I·alpha = tau - omega × (I·omega)
        I_omega = self.spec.inertia @ self._omega_body
        alpha_body = self._inv_inertia @ (
            tau_body_total - np.cross(self._omega_body, I_omega)
        )

        # 8. Semi-implicit Euler at sub_dt. RK4 would be more accurate;
        # left as an exercise for tier-1 sims.
        self._velocity = self._velocity + a_world * sub_dt
        self._position = self._position + self._velocity * sub_dt

        self._omega_body = self._omega_body + alpha_body * sub_dt
        self._quat = self._integrate_quaternion(self._quat, self._omega_body, sub_dt)

    def get_state(self) -> DroneState:
        return DroneState(
            position=self._position.copy(),
            velocity=self._velocity.copy(),
            quaternion=self._quat.copy(),
            angular_velocity_body=self._omega_body.copy(),
            motor_omegas=self._motor_omegas.copy(),
        )

    # ------------------------------------------------------------------ internals
    def _motor_dynamics_step(self, motor_cmds: np.ndarray, dt: float) -> None:
        """Tier 0 motor model: first-order lag toward commanded RPM.

        Exact integration of  dω/dt = (ω_cmd - ω) / τ  over dt:
            ω_new = ω + (1 - exp(-dt/τ)) * (ω_cmd - ω)
        """
        omega_cmd = motor_cmds * self._omega_max
        if self._tau_motor <= 0:
            self._motor_omegas = omega_cmd  # instant response
        else:
            alpha = 1.0 - np.exp(-dt / self._tau_motor)
            self._motor_omegas = self._motor_omegas + alpha * (omega_cmd - self._motor_omegas)

    def _compute_motor_wrench(self):
        """Sum thrust forces and reaction torques from current motor RPMs.

        Returns (F_body, tau_body), both (3,) arrays.
        """
        omega2 = self._motor_omegas ** 2                  # (N,)
        T = self._k_T * omega2                            # (N,) thrust magnitudes
        Q = self._k_Q * omega2                            # (N,) drag-torque magnitudes

        # Thrust force vectors: each motor pushes along its thrust_axis.
        F_motors = T[:, None] * self._motor_axes          # (N, 3)
        F_body = F_motors.sum(axis=0)                     # (3,)

        # Thrust moments: r × F per motor.
        tau_thrust = np.cross(self._motor_positions, F_motors)  # (N, 3)
        # Drag reaction: a CCW prop (spin=+1) creates CCW torque on air,
        # equal+opposite CW on body — i.e. the body torque is along
        # -spin * thrust_axis.
        tau_drag = (-self._motor_spins[:, None] * Q[:, None]) * self._motor_axes  # (N, 3)
        tau_body = tau_thrust.sum(axis=0) + tau_drag.sum(axis=0)
        return F_body, tau_body

    def _hover_omega(self) -> float:
        """Per-motor steady-state RPM at hover, assuming all motors equal
        and thrust axis aligned with world +z (body upright). For tilted
        motors or non-symmetric layouts this is only an approximation —
        the real solution comes from the inverse mixer, but the difference
        decays in the first few control steps."""
        T_per_motor = self.spec.hover_thrust_per_motor
        return float(np.sqrt(max(T_per_motor, 0.0) / max(self._k_T, 1e-12)))

    @staticmethod
    def _coerce_attitude(att: np.ndarray) -> np.ndarray:
        """Accept either RPY (3,) or quaternion (4,) and return (x,y,z,w)."""
        a = np.asarray(att, dtype=np.float64).reshape(-1)
        if a.shape == (4,):
            q = a / np.linalg.norm(a)
            return q
        if a.shape == (3,):
            r, p, y = float(a[0]), float(a[1]), float(a[2])
            cr, sr = np.cos(r / 2), np.sin(r / 2)
            cp, sp = np.cos(p / 2), np.sin(p / 2)
            cy, sy = np.cos(y / 2), np.sin(y / 2)
            # PyBullet RPY convention: R = Rz(yaw) Ry(pitch) Rx(roll)
            qw = cr * cp * cy + sr * sp * sy
            qx = sr * cp * cy - cr * sp * sy
            qy = cr * sp * cy + sr * cp * sy
            qz = cr * cp * sy - sr * sp * cy
            return np.array([qx, qy, qz, qw], dtype=np.float64)
        raise ValueError(f"attitude must be (3,) RPY or (4,) quat, got {a.shape}")

    @staticmethod
    def _rotation_matrix(q: np.ndarray) -> np.ndarray:
        """Body-to-world rotation matrix from quaternion (x, y, z, w)."""
        x, y, z, w = float(q[0]), float(q[1]), float(q[2]), float(q[3])
        return np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z),     2 * (x * z + w * y)],
            [2 * (x * y + w * z),     1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y),     2 * (y * z + w * x),     1 - 2 * (x * x + y * y)],
        ], dtype=np.float64)

    @staticmethod
    def _integrate_quaternion(q: np.ndarray, omega_body: np.ndarray, dt: float) -> np.ndarray:
        """Integrate q_dot = 0.5 * q ⊗ (omega_body, 0) over dt.

        Uses the closed-form rotation-by-axis-angle formulation, which is
        exact for constant angular velocity over the substep — no drift,
        no need to renormalize aggressively. Quaternion convention is
        (x, y, z, w) (PyBullet)."""
        omega_norm = float(np.linalg.norm(omega_body))
        if omega_norm < 1e-12:
            return q.copy()
        axis = omega_body / omega_norm
        angle = omega_norm * dt
        half = 0.5 * angle
        s, c = np.sin(half), np.cos(half)
        # Delta quaternion in body frame: Δq = (axis * sin(θ/2), cos(θ/2))
        dq = np.array([axis[0] * s, axis[1] * s, axis[2] * s, c], dtype=np.float64)
        # New q = q ⊗ Δq (Hamilton product, scalar-last convention)
        q_new = _quat_mul(q, dq)
        return q_new / np.linalg.norm(q_new)


def _quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product, both quaternions in (x, y, z, w) convention."""
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array([
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    ], dtype=np.float64)


# Module-level entry point. evaluation/evaluate.py looks for `make_drone_sim`
# (preferred) or `DroneSim` to instantiate the simulator.
DroneSim = BaselineDroneSimulator


def make_drone_sim(spec_path: str) -> BaselineDroneSimulator:
    return BaselineDroneSimulator(spec_path)
