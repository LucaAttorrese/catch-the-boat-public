"""Kalman-filter-only estimator for the Catch the Boat challenge.

This file intentionally contains NO landing controller, NO MPC, NO PID loops,
NO phase/state machine, and NO motor-command generation.

What it does:
    1. Detect the ArUco marker in the drone camera image.
    2. Convert the marker pose from camera frame to world frame.
    3. Fuse marker position measurements with a 6-state constant-velocity
       Kalman filter for the boat:

           x = [px, py, pz, vx, vy, vz]

Usage inside another agent/controller:

    estimator = BoatKalmanEstimator(dt=0.02)

    # every simulation step
    estimate = estimator.step(obs)

    if estimate["position"] is not None:
        boat_pos_world = estimate["position"]
        boat_vel_world = estimate["velocity"]

The returned estimate can then be passed to a separate controller, for example
PID, MPC, or any custom landing strategy. This module only estimates the boat.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import cv2
import numpy as np

from boat_landing.camera import CAMERA_BODY_OFFSET_Z, get_intrinsics


MARKER_ID = 0
MARKER_SIZE_M = 0.8
DEFAULT_DT = 0.02  # BoatLandingEnv.DT in the original project


# ---------------------------------------------------------------------------
# Geometry helper
# ---------------------------------------------------------------------------
def rpy_to_matrix(rpy: np.ndarray) -> np.ndarray:
    """Return world-from-body rotation matrix from PyBullet roll-pitch-yaw.

    PyBullet uses R = Rz(yaw) @ Ry(pitch) @ Rx(roll).
    """
    roll, pitch, yaw = float(rpy[0]), float(rpy[1]), float(rpy[2])

    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    rx = np.array(
        [[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]],
        dtype=np.float64,
    )
    ry = np.array(
        [[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]],
        dtype=np.float64,
    )
    rz = np.array(
        [[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return rz @ ry @ rx


# ---------------------------------------------------------------------------
# Kalman filter
# ---------------------------------------------------------------------------
@dataclass
class KalmanEstimate:
    """Container for the current filtered boat state."""

    position: Optional[np.ndarray]
    velocity: Optional[np.ndarray]
    covariance: Optional[np.ndarray]
    fresh: bool
    stale_steps: int
    detected: bool
    initialized: bool
    measurement_world: Optional[np.ndarray] = None

    def as_dict(self) -> Dict:
        return {
            "position": None if self.position is None else self.position.copy(),
            "velocity": None if self.velocity is None else self.velocity.copy(),
            "covariance": None if self.covariance is None else self.covariance.copy(),
            "fresh": self.fresh,
            "stale_steps": self.stale_steps,
            "detected": self.detected,
            "initialized": self.initialized,
            "measurement_world": (
                None if self.measurement_world is None else self.measurement_world.copy()
            ),
        }


class BoatKalmanFilter:
    """6-state constant-velocity Kalman filter for boat pose estimation.

    State:
        x = [px, py, pz, vx, vy, vz]

    Measurement:
        z = [px, py, pz]

    The prediction model assumes constant velocity. Unknown boat acceleration is
    modeled as white-noise acceleration, which creates the standard process
    noise block:

        Q_axis = sigma_a^2 * [[dt^4/4, dt^3/2],
                              [dt^3/2, dt^2  ]]
    """

    def __init__(
        self,
        dt: float = DEFAULT_DT,
        sigma_accel: float = 2.0,
        sigma_meas: float = 0.10,
    ) -> None:
        self.sigma_accel = float(sigma_accel)
        self.sigma_meas = float(sigma_meas)
        self._initialized = False

        self.x = np.zeros(6, dtype=np.float64)
        self.P = np.eye(6, dtype=np.float64)

        self.H = np.zeros((3, 6), dtype=np.float64)
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0
        self.H[2, 2] = 1.0

        self.R = np.eye(3, dtype=np.float64) * self.sigma_meas**2
        self.F = np.eye(6, dtype=np.float64)
        self.Q = np.zeros((6, 6), dtype=np.float64)
        self.set_dt(dt)

    @property
    def initialized(self) -> bool:
        return self._initialized

    @property
    def position(self) -> np.ndarray:
        return self.x[:3].copy()

    @property
    def velocity(self) -> np.ndarray:
        return self.x[3:].copy()

    def reset(self) -> None:
        self._initialized = False
        self.x[:] = 0.0
        self.P = np.eye(6, dtype=np.float64)

    def set_dt(self, dt: float) -> None:
        """Update F and Q for a given sample time."""
        dt = max(float(dt), 1e-6)
        self.dt = dt
        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt2 * dt2

        self.F = np.eye(6, dtype=np.float64)
        self.F[0, 3] = dt
        self.F[1, 4] = dt
        self.F[2, 5] = dt

        q = self.sigma_accel**2
        self.Q = np.zeros((6, 6), dtype=np.float64)
        for i in range(3):
            self.Q[i, i] = q * dt4 / 4.0
            self.Q[i, i + 3] = q * dt3 / 2.0
            self.Q[i + 3, i] = q * dt3 / 2.0
            self.Q[i + 3, i + 3] = q * dt2

    def initialize(self, measured_position: np.ndarray) -> None:
        """Initialize the filter from the first valid ArUco world position."""
        z = np.asarray(measured_position, dtype=np.float64).reshape(3)
        self.x[:] = 0.0
        self.x[:3] = z

        # Position is known moderately well; velocity is initially unknown.
        self.P = np.diag([0.25, 0.25, 0.10, 4.0, 4.0, 0.25]).astype(np.float64)
        self._initialized = True

    def predict(self, dt: Optional[float] = None) -> None:
        """Prediction step: propagate boat position and velocity forward."""
        if dt is not None and abs(float(dt) - self.dt) > 1e-9:
            self.set_dt(float(dt))

        if not self._initialized:
            return

        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.P = 0.5 * (self.P + self.P.T)  # keep covariance symmetric

    def update(self, measured_position: np.ndarray) -> None:
        """Correction step with a world-frame marker position measurement."""
        z = np.asarray(measured_position, dtype=np.float64).reshape(3)

        if not self._initialized:
            self.initialize(z)
            return

        innovation = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.solve(S, np.eye(3))

        self.x = self.x + K @ innovation

        # Joseph covariance update is more numerically stable than (I-KH)P.
        I_KH = np.eye(6, dtype=np.float64) - K @ self.H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T
        self.P = 0.5 * (self.P + self.P.T)


# ---------------------------------------------------------------------------
# ArUco perception + Kalman estimator wrapper
# ---------------------------------------------------------------------------
class BoatKalmanEstimator:
    """ArUco perception plus boat-position Kalman filtering.

    This class deliberately has no `control()` method and does not output motor
    commands. Use `step(obs)` to update the estimate from an environment
    observation.
    """

    def __init__(
        self,
        dt: float = DEFAULT_DT,
        marker_size_m: float = MARKER_SIZE_M,
        marker_id: int = MARKER_ID,
        sigma_accel: float = 2.0,
        sigma_meas: float = 0.10,
    ) -> None:
        self.dt = float(dt)
        self.marker_size_m = float(marker_size_m)
        self.marker_id = int(marker_id)

        self.kf = BoatKalmanFilter(
            dt=self.dt,
            sigma_accel=sigma_accel,
            sigma_meas=sigma_meas,
        )

        self.intrinsics = get_intrinsics()
        self.dist_coeffs = np.zeros(5, dtype=np.float64)

        self._dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
        try:
            self._detector = cv2.aruco.ArucoDetector(
                self._dictionary,
                cv2.aruco.DetectorParameters(),
            )
        except AttributeError:
            # Older OpenCV versions use cv2.aruco.detectMarkers directly.
            self._detector = None

        h = self.marker_size_m / 2.0
        self._object_points = np.array(
            [[-h, h, 0.0], [h, h, 0.0], [h, -h, 0.0], [-h, -h, 0.0]],
            dtype=np.float32,
        )

        self._frames_since_detection = 0
        self._last_time: Optional[float] = None
        self._last_perception: Dict = {"detected": False}
        self._last_estimate = KalmanEstimate(
            position=None,
            velocity=None,
            covariance=None,
            fresh=False,
            stale_steps=0,
            detected=False,
            initialized=False,
        )

    def reset(self) -> None:
        self.kf.reset()
        self._frames_since_detection = 0
        self._last_time = None
        self._last_perception = {"detected": False}
        self._last_estimate = KalmanEstimate(
            position=None,
            velocity=None,
            covariance=None,
            fresh=False,
            stale_steps=0,
            detected=False,
            initialized=False,
        )

    def step(self, obs: Dict) -> Dict:
        """Update perception + Kalman filter from one simulator observation.

        Expected observation keys:
            obs["camera"]: RGB image from the drone camera
            obs["state"]["position"]: drone world position
            obs["state"]["attitude"]: drone roll-pitch-yaw attitude
            obs["time"]: optional simulation time in seconds
        """
        camera_image = obs["camera"]
        drone_state = obs["state"]
        time_s = obs.get("time", None)

        dt = self._compute_dt(time_s)
        perception = self.perceive(camera_image)
        estimate = self.estimate(perception, drone_state, dt=dt)

        self._last_perception = perception
        self._last_estimate = estimate
        return estimate.as_dict()

    def _compute_dt(self, time_s: Optional[float]) -> float:
        if time_s is None:
            return self.dt

        time_s = float(time_s)
        if self._last_time is None:
            self._last_time = time_s
            return self.dt

        dt = time_s - self._last_time
        self._last_time = time_s
        if not np.isfinite(dt) or dt <= 0.0:
            return self.dt
        return dt

    def perceive(self, camera_image: np.ndarray) -> Dict:
        """Detect the configured ArUco marker and estimate its camera pose."""
        if camera_image is None:
            return {"detected": False}

        gray = cv2.cvtColor(camera_image, cv2.COLOR_RGB2GRAY)
        if self._detector is not None:
            corners, ids, _ = self._detector.detectMarkers(gray)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(gray, self._dictionary)

        if ids is None or len(ids) == 0:
            return {"detected": False}

        target_corners = None
        for marker_corners, marker_id in zip(corners, ids.flatten()):
            if int(marker_id) == self.marker_id:
                target_corners = marker_corners
                break

        if target_corners is None:
            return {"detected": False}

        image_points = target_corners.reshape(4, 2).astype(np.float32)
        ok, rvec, tvec = cv2.solvePnP(
            self._object_points,
            image_points,
            self.intrinsics,
            self.dist_coeffs,
            flags=cv2.SOLVEPNP_IPPE_SQUARE,
        )

        if not ok:
            return {"detected": False}

        tvec = np.asarray(tvec, dtype=np.float64).reshape(3)
        rvec = np.asarray(rvec, dtype=np.float64).reshape(3)

        if not (np.all(np.isfinite(tvec)) and np.all(np.isfinite(rvec))):
            return {"detected": False}

        # Reject obviously invalid depths.
        if tvec[2] <= 0.1 or tvec[2] > 100.0:
            return {"detected": False}

        return {
            "detected": True,
            "tvec": tvec,
            "rvec": rvec,
            "image_points": image_points,
        }

    def camera_to_world(self, tvec: np.ndarray, drone_state: Dict) -> np.ndarray:
        """Convert marker translation from camera frame to world frame."""
        drone_pos = np.asarray(drone_state["position"], dtype=np.float64).reshape(3)
        drone_rpy = np.asarray(drone_state["attitude"], dtype=np.float64).reshape(3)

        R_world_body = rpy_to_matrix(drone_rpy)
        body_x = R_world_body[:, 0]
        body_y = R_world_body[:, 1]
        body_z = R_world_body[:, 2]

        camera_world = drone_pos + body_z * CAMERA_BODY_OFFSET_Z

        # This matches the coordinate mapping used in the original agent.
        R_cam_to_world = np.column_stack([-body_y, -body_x, -body_z])
        return camera_world + R_cam_to_world @ np.asarray(tvec, dtype=np.float64)

    def estimate(self, perception: Dict, drone_state: Dict, dt: Optional[float] = None) -> KalmanEstimate:
        """Run one predict/correct cycle and return the current boat estimate."""
        self.kf.predict(dt=dt)

        measurement_world = None
        detected = bool(perception.get("detected", False))

        if detected:
            measurement_world = self.camera_to_world(perception["tvec"], drone_state)
            self.kf.update(measurement_world)
            self._frames_since_detection = 0
            return KalmanEstimate(
                position=self.kf.position,
                velocity=self.kf.velocity,
                covariance=self.kf.P.copy(),
                fresh=True,
                stale_steps=0,
                detected=True,
                initialized=self.kf.initialized,
                measurement_world=measurement_world,
            )

        self._frames_since_detection += 1

        if self.kf.initialized:
            return KalmanEstimate(
                position=self.kf.position,
                velocity=self.kf.velocity,
                covariance=self.kf.P.copy(),
                fresh=False,
                stale_steps=self._frames_since_detection,
                detected=False,
                initialized=True,
                measurement_world=None,
            )

        return KalmanEstimate(
            position=None,
            velocity=None,
            covariance=None,
            fresh=False,
            stale_steps=self._frames_since_detection,
            detected=False,
            initialized=False,
            measurement_world=None,
        )

    def get_last_estimate(self) -> Optional[Dict]:
        """Return the last estimate in the format expected by estimation RMSE code."""
        if self._last_estimate.position is None:
            return None
        return {
            "position": self._last_estimate.position.copy(),
            "velocity": self._last_estimate.velocity.copy(),
        }

    def get_debug_state(self) -> Dict:
        """Return extra information useful for printing or plotting."""
        return {
            "kf_initialized": self.kf.initialized,
            "kf_position": None if not self.kf.initialized else self.kf.position,
            "kf_velocity": None if not self.kf.initialized else self.kf.velocity,
            "kf_covariance": None if not self.kf.initialized else self.kf.P.copy(),
            "frames_since_detection": self._frames_since_detection,
            "last_perception": self._last_perception,
            "last_estimate": self._last_estimate.as_dict(),
        }


# Backward-friendly names, but this is an estimator, not a motor-control agent.
KalmanAgent = BoatKalmanEstimator
Agent = BoatKalmanEstimator


def make_agent(*args, **kwargs) -> BoatKalmanEstimator:
    """Factory kept for compatibility with scripts that expect make_agent()."""
    # Ignore a possible drone_spec keyword from older runners; estimation does
    # not need the drone specification.
    kwargs.pop("drone_spec", None)
    return BoatKalmanEstimator(*args, **kwargs)
