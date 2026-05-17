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
        rpy, omega_body = self._extract_attitude(state)

        # Target attitudes in radians.
        roll_t = float(np.clip(roll_des, -1.0, 1.0)) * self.max_tilt
        pitch_t = float(np.clip(pitch_des, -1.0, 1.0)) * self.max_tilt
        yaw_rate_t = float(np.clip(yaw_rate_des, -1.0, 1.0)) * 1.5  # rad/s

        # Body-frame torques from PD.
        tau_x = self.kp_roll * (roll_t - rpy[0]) - self.kd_roll * omega_body[0]
        tau_y = self.kp_pitch * (pitch_t - rpy[1]) - self.kd_pitch * omega_body[1]
        tau_z = self.kp_yaw_rate * (yaw_rate_t - omega_body[2])

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

    @staticmethod
    def _extract_attitude(state):
        """Return (rpy_world, angular_velocity_body) regardless of whether
        the caller passed a dict (env obs) or a DroneState."""
        if isinstance(state, DroneState):
            quat = state.quaternion
            rpy = _quat_to_rpy(quat)
            # angular_velocity_body is body frame already.
            return rpy, state.angular_velocity_body
        # Dict path (env obs["state"]).
        rpy = np.asarray(state["attitude"], dtype=np.float64).reshape(3)
        ang_world = np.asarray(state["angular_velocity"], dtype=np.float64).reshape(3)
        # Convert world ω to body ω.
        R = _rpy_to_matrix(rpy)
        omega_body = R.T @ ang_world
        return rpy, omega_body


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
