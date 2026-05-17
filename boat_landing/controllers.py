"""Attitude controllers — utilities the agent can layer on top of a
motor-level DroneSimulator.

Why this exists
---------------
The DroneSimulator contract is motor-level: it accepts per-motor throttle
commands. That's the right contract for evaluating sim fidelity (motor
RPM dynamics + propeller model are first-class), but it forces every
agent to also carry an inner attitude controller — which is mostly
boilerplate for teams whose focus is perception/estimation.

`DefaultAttitudeController` solves that. It implements a textbook
PD-attitude + rate-loop yaw + inverse-mixer pipeline that converts
high-level setpoints (thrust, roll, pitch, yaw_rate) into per-motor
throttles. Gains are derived analytically from the drone spec's mass
and inertia for a target closed-loop natural frequency, so the same
controller works on any airframe whose YAML provides those fields.

Use it like this:

    from boat_landing.drone_interface import load_drone_spec
    from boat_landing.controllers import DefaultAttitudeController

    spec = load_drone_spec("drones/vtol.yaml")
    ctrl = DefaultAttitudeController(spec)

    def act(self, obs):
        # ... your perception + guidance produces:
        roll_des, pitch_des, yaw_rate_des = ...
        thrust_norm = ...           # in [-1, +1]; 0 = hover
        return ctrl(obs["state"], roll_des, pitch_des, yaw_rate_des, thrust_norm)

If you want to write your own attitude controller (e.g. to handle the
weak-yaw VTOL more aggressively, or to do gain scheduling on roll
phase), bypass this and emit per-motor throttles directly from your
agent. That earns the "control fidelity" half of the sim-quality bonus.
"""

from __future__ import annotations

from typing import Dict, Optional, Union

import numpy as np

from boat_landing.drone_interface import DroneSpec, DroneState


# Soft attitude envelope. Same value as the env's MAX_TILT historically.
DEFAULT_MAX_TILT = 0.30           # rad (~17 deg)
# Thrust action range: thrust_norm = -1 → 50% hover, +1 → 150% hover.
DEFAULT_THRUST_RANGE = 0.50


class DefaultAttitudeController:
    """PD on attitude + P on yaw rate + inverse mixer.

    Derives all gains from the spec's inertia tensor for a target
    closed-loop natural frequency and damping ratio. The same parameters
    work regardless of airframe scale because Kp / Kd scale with I.
    """

    def __init__(
        self,
        spec: DroneSpec,
        omega_n_roll: float = 8.0,
        omega_n_pitch: float = 6.0,
        omega_n_yaw_rate: float = 4.0,
        zeta: float = 0.7,
        max_tilt: float = DEFAULT_MAX_TILT,
        thrust_range: float = DEFAULT_THRUST_RANGE,
    ):
        self.spec = spec
        self.max_tilt = float(max_tilt)
        self.thrust_range = float(thrust_range)

        Ixx = float(spec.inertia[0, 0])
        Iyy = float(spec.inertia[1, 1])
        Izz = float(spec.inertia[2, 2])

        # Closed-loop attitude PD: chosen so I*alpha + Kd*omega + Kp*theta = 0
        # has natural frequency omega_n and damping zeta.
        self.kp_roll = Ixx * omega_n_roll ** 2
        self.kd_roll = 2.0 * zeta * omega_n_roll * Ixx
        self.kp_pitch = Iyy * omega_n_pitch ** 2
        self.kd_pitch = 2.0 * zeta * omega_n_pitch * Iyy
        # Yaw is a rate loop only (no attitude target): single-pole P.
        self.kp_yaw_rate = Izz * omega_n_yaw_rate

        # Mixer: (F_z, tau_x, tau_y, tau_z)_body = M @ T, where T is per-motor
        # thrust magnitude. Cached at construction.
        self._mixer = self._build_mixer(spec)
        self._mixer_pinv = np.linalg.pinv(self._mixer)
        self._k_T = float(spec.propeller.thrust_coefficient)
        self._omega_max = float(spec.motor.omega_max)
        self._T_max_per_motor = self._k_T * self._omega_max ** 2

        # Pre-extract per-motor geometry and inertia matrix for the
        # cross-coupling feedforward. The body's equation of motion is
        #   I·α = τ − ω × (I·ω) − ω × L_rotors
        # The PD law produces τ_pd assuming decoupled axes (I·α = τ_pd).
        # To make that assumption hold under significant cross-coupling
        # we add BOTH cancellation terms to the commanded torque:
        #   τ_commanded = τ_pd + ω × (I·ω) + ω × L_rotors
        # - ω × (I·ω) is the Euler / inertial cross-coupling. Significant
        #   whenever I is asymmetric (vtol: Ixx=2.54, Iyy=3.47, Izz=5.74).
        # - ω × L_rotors is the rotor-gyroscopic precession. At hover the
        #   opposite-spin layout makes L_rotors ≈ 0, so this term mostly
        #   matters during yaw transients. Both terms together cover the
        #   full Newton-Euler coupling the sim integrates.
        self._inertia = spec.inertia.copy()
        self._motor_axes = np.stack([m.thrust_axis for m in spec.motors])  # (N, 3)
        self._motor_spins = np.array(
            [m.spin for m in spec.motors], dtype=np.float64
        )
        self._rotor_inertia = float(spec.motor.rotor_inertia)
        # Fallback motor speed used when state lacks motor_omegas (legacy
        # dict callers without the field). Hover RPM is close enough for
        # the gyro term whose body-axis components are anyway ~0 at hover.
        self._hover_omega_approx = float(
            np.sqrt(max(spec.hover_thrust_per_motor, 0.0) / max(self._k_T, 1e-12))
        )

    # ------------------------------------------------------------------ public
    def __call__(
        self,
        state: Union[Dict, DroneState],
        roll_des: float,
        pitch_des: float,
        yaw_rate_des: float,
        thrust_norm: float,
    ) -> np.ndarray:
        """Map high-level setpoints to per-motor throttles in [0, 1].

        state: either obs["state"] (dict from the env) or a DroneState.
        roll_des, pitch_des: in [-1, +1]; mapped to ±max_tilt rad.
        yaw_rate_des:        in [-1, +1]; mapped to ±max_tilt * 5 rad/s
                             (~1.5 rad/s by default).
        thrust_norm:         in [-1, +1]; 0 = hover (mg), +1 = 150% hover.
        """
        rpy, omega_body, motor_omegas = self._extract_state(state)

        # Target attitudes in radians.
        roll_t = float(np.clip(roll_des, -1.0, 1.0)) * self.max_tilt
        pitch_t = float(np.clip(pitch_des, -1.0, 1.0)) * self.max_tilt
        yaw_rate_t = float(np.clip(yaw_rate_des, -1.0, 1.0)) * 1.5  # rad/s

        # Body-frame torques from PD.
        tau_x = self.kp_roll * (roll_t - rpy[0]) - self.kd_roll * omega_body[0]
        tau_y = self.kp_pitch * (pitch_t - rpy[1]) - self.kd_pitch * omega_body[1]
        tau_z = self.kp_yaw_rate * (yaw_rate_t - omega_body[2])

        # Cross-coupling feedforward. Add the Euler term ω×(I·ω) and the
        # rotor-gyro term ω×L_rotors to the commanded torque so the body
        # actually rotates at the rate the PD asked for. Without these,
        # asymmetric inertia (Ixx=2.54, Iyy=3.47, Izz=5.74) couples roll
        # into yaw and vice versa during fast manoeuvres — the chief
        # cause of cascade-controller instability when commanding
        # diagonal pos_err.
        I_omega = self._inertia @ omega_body
        tau_euler = np.cross(omega_body, I_omega)
        tau_x += tau_euler[0]
        tau_y += tau_euler[1]
        tau_z += tau_euler[2]
        if self._rotor_inertia > 1e-12:
            L_rotors = (
                self._rotor_inertia
                * (self._motor_spins * motor_omegas)[:, None]
                * self._motor_axes
            ).sum(axis=0)
            tau_gyro = np.cross(omega_body, L_rotors)
            tau_x += tau_gyro[0]
            tau_y += tau_gyro[1]
            tau_z += tau_gyro[2]

        # Body-z thrust target. Hover offset + commanded delta. Note that
        # this is BODY-frame thrust along the dominant motor axis (+z),
        # NOT world-frame; tilt compensation is the agent's job.
        thrust = float(np.clip(thrust_norm, -1.0, 1.0))
        F_z = self.spec.hover_thrust * (1.0 + thrust * self.thrust_range)

        # Inverse mixer: solve M @ T = wrench for T (least squares for
        # over-actuated configurations, exact for square M).
        wrench = np.array([F_z, tau_x, tau_y, tau_z], dtype=np.float64)
        T = self._mixer_pinv @ wrench

        # Clip and convert to throttle. T = k_T * omega^2 = k_T * (throttle*ω_max)^2
        # → throttle = sqrt(T / T_max_per_motor)
        T = np.clip(T, 0.0, self._T_max_per_motor)
        throttles = np.sqrt(T / self._T_max_per_motor)
        return throttles

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _build_mixer(spec: DroneSpec) -> np.ndarray:
        """Construct the body-wrench-from-per-motor-thrust matrix.

        Returns M of shape (4, N) where:
            wrench = [F_z_body, tau_x_body, tau_y_body, tau_z_body] = M @ T
        and T = (T_0, ..., T_{N-1}) is the per-motor thrust vector.
        """
        N = spec.num_motors
        c = spec.propeller.drag_coefficient / spec.propeller.thrust_coefficient
        M = np.zeros((4, N), dtype=np.float64)
        for i, motor in enumerate(spec.motors):
            axis = motor.thrust_axis
            # F_z contribution per unit T_i (z-component of force vector).
            M[0, i] = axis[2]
            # Moment per unit T_i: thrust moment + drag reaction moment.
            #   tau_thrust = r_i x (T_i * axis_i)
            #   tau_drag   = -spin_i * Q_i * axis_i,  Q_i = c * T_i
            #   total / T_i = (r_i x axis_i) - spin_i * c * axis_i
            moment = np.cross(motor.position, axis) - motor.spin * c * axis
            M[1, i] = moment[0]
            M[2, i] = moment[1]
            M[3, i] = moment[2]
        return M

    def _extract_state(self, state):
        """Return (rpy_world, angular_velocity_body, motor_omegas) regardless
        of whether the caller passed a dict (env obs) or a DroneState.

        motor_omegas falls back to the hover estimate if the dict path
        doesn't include it — keeps the controller working for legacy
        callers, just without the gyro feedforward's full accuracy.
        """
        n_motors = self.spec.num_motors
        if isinstance(state, DroneState):
            quat = state.quaternion
            rpy = _quat_to_rpy(quat)
            motor_omegas = np.asarray(state.motor_omegas, dtype=np.float64).reshape(-1)
            if motor_omegas.size != n_motors:
                motor_omegas = np.full(n_motors, self._hover_omega_approx)
            return rpy, np.asarray(state.angular_velocity_body, dtype=np.float64), motor_omegas
        # Dict path (env obs["state"]).
        rpy = np.asarray(state["attitude"], dtype=np.float64).reshape(3)
        ang_world = np.asarray(state["angular_velocity"], dtype=np.float64).reshape(3)
        R = _rpy_to_matrix(rpy)
        omega_body = R.T @ ang_world
        omegas_raw = state.get("motor_omegas")
        if omegas_raw is not None:
            motor_omegas = np.asarray(omegas_raw, dtype=np.float64).reshape(-1)
            if motor_omegas.size != n_motors:
                motor_omegas = np.full(n_motors, self._hover_omega_approx)
        else:
            motor_omegas = np.full(n_motors, self._hover_omega_approx)
        return rpy, omega_body, motor_omegas


# ---------------------------------------------------------------------------
# Small math helpers — duplicated from agents/agent_baseline.py on purpose
# so this module has no dependency on participant code.
# ---------------------------------------------------------------------------


def _rpy_to_matrix(rpy: np.ndarray) -> np.ndarray:
    """World-from-body rotation matrix using PyBullet's RPY convention:
    R = Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    r, p_, y = float(rpy[0]), float(rpy[1]), float(rpy[2])
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p_), np.sin(p_)
    cy, sy = np.cos(y), np.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _quat_to_rpy(q: np.ndarray) -> np.ndarray:
    """PyBullet (x, y, z, w) quaternion to RPY (Z-Y-X intrinsic)."""
    x, y, z, w = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    # Roll (x-axis rotation)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    # Pitch (y-axis rotation)
    sinp = 2.0 * (w * y - z * x)
    pitch = np.arcsin(np.clip(sinp, -1.0, 1.0))
    # Yaw (z-axis rotation)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return np.array([roll, pitch, yaw], dtype=np.float64)
