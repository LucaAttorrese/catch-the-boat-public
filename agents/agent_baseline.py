"""Baseline agent for the Catch the Boat challenge.

What this agent does:
    1. Detect the ArUco marker with cv2.aruco.ArucoDetector.
    2. Estimate marker pose with cv2.solvePnP (IPPE_SQUARE).
    3. Convert the marker's camera-frame translation to a world-frame
       position using the drone's attitude.
    4. Run a small state machine: SEARCH -> APPROACH -> DESCEND -> LAND.
    5. Drive three independent PIDs (x, y, z) to produce high-level
       (thrust, roll, pitch, yaw_rate) setpoints.
    6. Map those setpoints to per-motor throttles via the stock
       `DefaultAttitudeController`. The env's `step()` consumes
       per-motor throttles, not high-level setpoints.

If you want to write your own attitude controller (and earn the control-
fidelity portion of the sim-quality bonus), bypass step 6 and emit motor
throttles directly from `control()`.

Deliberate weaknesses (improvement directions for teams):
    - No prediction of boat motion: we always aim at the *current* marker
      pose, so we lag behind a moving boat.
    - No detection-jitter filter (no Kalman, no smoothing): every
      detection is trusted as-is.
    - No real recovery when the marker is briefly lost during descent.
      We hold the last known position and hope it returns. With
      occlusions, this leads to drift.
    - Hardcoded thresholds for state transitions and a hardcoded prior
      (origin) for cold-start search.
    - No abort/retry logic if the descent is going wrong (e.g. drone
      heading off the platform).
    - Independent x/y/z PIDs ignore the cross-coupling caused by tilt.
    - No active yaw control to align with boat heading — the wing-
      perpendicular landing condition will fail unless the boat happens
      to be aligned with the world x-axis at touchdown.
    - LAND-phase descent is bang-bang (thrust pinned to its lower bound)
      to brute-force through ground effect. Descent velocity often
      saturates and forfeits the soft-landing bonus. A smooth descent
      profile + velocity feed-forward both recover and stack.

Don't optimize the baseline. Fork agent_template.py and build your own.
"""

from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np

from boat_landing.controllers import DefaultAttitudeController
from boat_landing.drone_interface import DroneSpec, load_drone_spec


REPO_ROOT = Path(__file__).resolve().parent.parent


# Must match BoatLandingEnv.MARKER_SIZE.
MARKER_SIZE_M = 0.8

# Camera parameters replicated locally (mirroring boat_landing/camera.py)
# to keep the agent free of any runtime dependency on the camera module
# beyond what is strictly observation-derived. These are static sensor
# specs (FOV 90 deg, 640x480), not scenario state.
_CAMERA_WIDTH = 640
_CAMERA_HEIGHT = 480
_CAMERA_FOV_DEG = 90.0
_CAMERA_BODY_OFFSET_Z = -0.115

def _build_intrinsics() -> np.ndarray:
    fov_rad = float(np.deg2rad(_CAMERA_FOV_DEG))
    fy = _CAMERA_HEIGHT / (2.0 * np.tan(fov_rad / 2.0))
    fx = fy
    cx = _CAMERA_WIDTH / 2.0
    cy = _CAMERA_HEIGHT / 2.0
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)

_CAMERA_INTRINSICS = _build_intrinsics()

PHASE_SEARCH = "SEARCH"
PHASE_APPROACH = "APPROACH"
PHASE_DESCEND = "DESCEND"
PHASE_LAND = "LAND"
PHASE_WIND_EST = "WIND_EST"


def rpy_to_matrix(rpy) -> np.ndarray:
    """World-from-body rotation matrix using PyBullet's RPY convention:
    R = Rz(yaw) @ Ry(pitch) @ Rx(roll). Used to transform the marker's
    camera-frame translation into world coordinates."""
    r, p_, y = float(rpy[0]), float(rpy[1]), float(rpy[2])
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p_), np.sin(p_)
    cy, sy = np.cos(y), np.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


class PID:
    """Single-axis PID with output clamping.

    Supports both 'D on error' (the textbook form) and 'D on measurement'
    (the form most controllers in robotics actually use). Pass
    ``measurement_velocity`` to ``__call__`` to use D on measurement —
    that avoids the derivative kick that hits when the setpoint jumps
    (e.g. when our boat estimate snaps to a fresh ArUco detection).
    """

    def __init__(
        self,
        kp: float,
        ki: float,
        kd: float,
        out_min: float = -1.0,
        out_max: float = 1.0,
        i_clip: float = 1.0,
    ):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.out_min, self.out_max = out_min, out_max
        self.i_clip = i_clip
        self.reset()

    def reset(self) -> None:
        self._i = 0.0
        self._prev_err: Optional[float] = None

    def __call__(
        self,
        err: float,
        dt: float,
        measurement_velocity: Optional[float] = None,
    ) -> float:
        self._i = float(np.clip(self._i + err * dt, -self.i_clip, self.i_clip))
        if measurement_velocity is not None:
            # err = setpoint - measurement, so d(err)/dt = -d(meas)/dt.
            d = -float(measurement_velocity)
        elif self._prev_err is None:
            d = 0.0
        else:
            d = (err - self._prev_err) / max(dt, 1e-6)
        self._prev_err = err
        out = self.kp * err + self.ki * self._i + self.kd * d
        return float(np.clip(out, self.out_min, self.out_max))


class BoatKalmanFilter:
    """6-state constant-velocity Kalman filter on boat (x, y, z, vx, vy, vz).

    Process model:
        white-noise acceleration with std `sigma_accel` (m/s^2). Builds the
        standard Q block per axis:
            Q_axis = sigma_a^2 * [[dt^4/4, dt^3/2],
                                  [dt^3/2, dt^2  ]]
    Measurement model:
        position-only (rows of H pick px, py, pz). Per-axis std `sigma_meas`.
    """

    def __init__(self, dt: float, sigma_accel: float = 2.0, sigma_meas: float = 0.10):
        self.sigma_accel = float(sigma_accel)
        self.sigma_meas = float(sigma_meas)
        self.initialized = False
        self.x = np.zeros(6, dtype=np.float64)
        self.P = np.eye(6, dtype=np.float64)
        self.H = np.zeros((3, 6), dtype=np.float64)
        self.H[0, 0] = self.H[1, 1] = self.H[2, 2] = 1.0
        self.R = np.eye(3, dtype=np.float64) * self.sigma_meas ** 2
        self.F = np.eye(6, dtype=np.float64)
        self.Q = np.zeros((6, 6), dtype=np.float64)
        self.set_dt(dt)

    def set_dt(self, dt: float) -> None:
        dt = max(float(dt), 1e-6)
        self.dt = dt
        self.F = np.eye(6, dtype=np.float64)
        self.F[0, 3] = self.F[1, 4] = self.F[2, 5] = dt
        q = self.sigma_accel ** 2
        dt2, dt3, dt4 = dt * dt, dt ** 3, dt ** 4
        self.Q = np.zeros((6, 6), dtype=np.float64)
        for i in range(3):
            self.Q[i, i] = q * dt4 / 4.0
            self.Q[i, i + 3] = q * dt3 / 2.0
            self.Q[i + 3, i] = q * dt3 / 2.0
            self.Q[i + 3, i + 3] = q * dt2

    @property
    def position(self) -> np.ndarray:
        return self.x[:3].copy()

    @property
    def velocity(self) -> np.ndarray:
        return self.x[3:].copy()

    def predict(self) -> None:
        if not self.initialized:
            return
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.P = 0.5 * (self.P + self.P.T)  # keep symmetric

    def update(self, z: np.ndarray) -> None:
        z = np.asarray(z, dtype=np.float64).reshape(3)
        if not self.initialized:
            # Bootstrap: trust position, leave velocity unknown (high P_vv).
            self.x[:] = 0.0
            self.x[:3] = z
            self.P = np.diag([0.25, 0.25, 0.10, 4.0, 4.0, 0.25]).astype(np.float64)
            self.initialized = True
            return
        innovation = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.solve(S, np.eye(3))
        self.x = self.x + K @ innovation
        # Joseph form: more numerically stable than (I - K H) P.
        I_KH = np.eye(6, dtype=np.float64) - K @ self.H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T
        self.P = 0.5 * (self.P + self.P.T)


class BaselineAgent:
    """ArUco + state machine + 3 independent PIDs."""

    DT = 0.02  # must match BoatLandingEnv.DT

    # Search prior: when the marker has never been detected, head toward
    # this world position at this altitude. The boats spawn near the
    # origin in all public scenarios.
    SEARCH_PRIOR_XY = np.array([0.0, 0.0])
    SEARCH_PRIOR_ALT = 5.5

    # Phase transition thresholds (m)
    APPROACH_HORIZ_OK = 0.6
    APPROACH_ALTITUDE_OK = 3.5
    DESCEND_HORIZ_OK = 0.4
    DESCEND_ALTITUDE_OK = 0.6

    # LAND-commit gate: only commit when the boat deck tilt (from rvec)
    # is below this threshold. If we have to wait too long for a level
    # window, commit anyway (safety timeout).
    LAND_TILT_MAX_RAD = float(np.deg2rad(5.0))
    LAND_WAIT_MAX_STEPS = 100   # 2.0 s at DT=0.02

    # WIND_EST phase: one-shot hover at high altitude to read drone drift
    # under wind. The resulting world-frame acceleration is reused as a
    # static feed-forward in APPROACH/DESCEND.
    WIND_EST_STEPS = 50         # 1.0 s window
    WIND_EST_MIN_ALT = 2.0      # only trigger when well above the marker

    # Target altitude above the marker per phase. Each phase's target must
    # be *below* the threshold for the next phase, otherwise the drone
    # settles at equilibrium and never transitions. With Kp/Kd ≈ 0.33 in
    # the z PID the drone settles at its target altitude with no
    # overshoot — no slack.
    PHASE_ALTITUDE = {
        PHASE_SEARCH: 5.5,
        PHASE_APPROACH: 3.0,   # below APPROACH_ALTITUDE_OK (3.5)
        PHASE_DESCEND: 0.4,    # below DESCEND_ALTITUDE_OK (0.6)
        PHASE_LAND: -0.5,      # below the marker so the drone keeps descending
    }

    def __init__(self, drone_spec: Optional[DroneSpec] = None):
        if drone_spec is None:
            drone_spec = load_drone_spec(REPO_ROOT / "drones" / "vtol.yaml")
        self.spec: DroneSpec = drone_spec
        self.attitude_ctrl = DefaultAttitudeController(drone_spec)
        # Pre-compute the per-motor hover throttle for safe-fallback recovery.
        self._hover_action = self.attitude_ctrl(
            {"attitude": np.zeros(3), "angular_velocity": np.zeros(3)},
            roll_des=0.0, pitch_des=0.0, yaw_rate_des=0.0, thrust_norm=0.0,
        )

        self.intrinsics = _CAMERA_INTRINSICS
        self.dist_coeffs = np.zeros(5, dtype=np.float64)

        self._dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
        params = cv2.aruco.DetectorParameters()
        # CONTOUR refinement fits the marker edges (complementary to the
        # corner-level cv2.cornerSubPix run after detection in perceive()).
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_CONTOUR
        # Default adaptive-threshold window (3-23-10) targets small markers;
        # widen it for the 0.8 m plate seen blurry/foggy from altitude.
        params.adaptiveThreshWinSizeMin = 5
        params.adaptiveThreshWinSizeMax = 35
        params.adaptiveThreshWinSizeStep = 6
        # Lower minMarkerPerimeterRate so the marker stays detectable
        # higher up (cold-start SEARCH altitude is 5.5 m).
        params.minMarkerPerimeterRate = 0.02
        # More tolerant polygonal approximation -> survives motion blur.
        params.polygonalApproxAccuracyRate = 0.05
        self._detector_params = params
        try:
            self._detector = cv2.aruco.ArucoDetector(self._dictionary, params)
        except AttributeError:
            self._detector = None  # OpenCV < 4.7 fallback path used in perceive()

        # CLAHE: restores local contrast washed out by Beer-Lambert fog at
        # altitude (see boat_landing/camera.py). Reused across frames.
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        # Marker corner ordering used by IPPE_SQUARE: TL, TR, BR, BL in the
        # marker's own local frame, marker plane = z=0.
        h = MARKER_SIZE_M / 2.0
        self._object_points = np.array(
            [
                [-h,  h, 0.0],
                [ h,  h, 0.0],
                [ h, -h, 0.0],
                [-h, -h, 0.0],
            ],
            dtype=np.float32,
        )

        # PID gains tuned for stability rather than aggressiveness — leave
        # plenty of room for teams to push performance up. With Kp/Kd
        # ratio fixed, the closed-loop velocity at error e is roughly
        # v_steady ≈ (Kp/Kd) * e — ~0.5 means descend at 0.5 m/s for a
        # 1 m altitude error, gentle enough to avoid free-fall before
        # the controller stabilizes.
        self.pid_x = PID(kp=0.15, ki=0.02, kd=0.40)
        self.pid_y = PID(kp=0.15, ki=0.02, kd=0.40)
        self.pid_z = PID(kp=0.10, ki=0.02, kd=0.30)
        self.pid_yaw = PID(kp=1.0, ki=0.0, kd=0.0)

        # Constant-velocity Kalman on (x, y, z, vx, vy, vz). Smooths solvePnP
        # jitter and gives us a non-zero velocity estimate the controller can
        # use for feed-forward.
        self.kf = BoatKalmanFilter(dt=self.DT)

        self.phase = PHASE_SEARCH
        self._prev_phase: Optional[str] = None
        self._last_marker_world: Optional[np.ndarray] = None
        self._frames_since_detection = 0
        self._last_estimate: Optional[Dict] = None
        # Last seen boat tilt (from marker rvec). Cached so decide() can
        # gate on the previous value during brief detection gaps.
        self._last_boat_roll: Optional[float] = None
        self._last_boat_pitch: Optional[float] = None

        # Cache of the most recent 4-DoF FC setpoint computed by control().
        # act_setpoint() returns this — keeping act() as the single
        # pipeline path and act_setpoint() as a thin delegate.
        self._last_setpoint: tuple = (0.0, 0.0, 0.0, 0.0)

        # ABORT / go-around state. When the descent is going wrong (lost
        # detection, drifted off platform), decide() pins the phase to
        # APPROACH for `_abort_until_step - _step_count` ticks so the
        # drone climbs and re-acquires before re-attempting LAND.
        self._step_count = 0
        self._abort_until_step = -1
        # LAND-wait state: step at which we first became "ready to LAND
        # but waiting for a level deck". None means not currently waiting.
        self._land_wait_since_step: Optional[int] = None

        # WIND_EST state. One-shot: hover at high altitude for
        # WIND_EST_STEPS ticks, snapshot velocity at start and end, derive
        # wind acceleration in world frame, reuse as static FF afterwards.
        self._wind_est_done = False
        self._wind_est_started_step: Optional[int] = None
        self._wind_est_v0: Optional[np.ndarray] = None
        self._wind_est_z0: Optional[float] = None
        self._wind_accel_w = np.zeros(2)

    # ------------------------------------------------------------------ public API
    def act(self, obs: Dict) -> np.ndarray:
        """Top-level orchestrator. The agent_template.py version is a
        copy of this; teams should override perceive/estimate/decide/
        control independently and leave act() alone."""
        camera = obs["camera"]
        state = obs["state"]
        battery = obs["battery"]
        time_s = obs["time"]

        perception = self.perceive(camera)
        boat_estimate = self.estimate(perception, state, history=None)
        self._last_estimate = boat_estimate
        self.phase = self.decide(state, boat_estimate, battery, time_s)
        action = self.control(state, boat_estimate, self.phase)
        if not np.all(np.isfinite(action)):
            import sys
            print(
                f"[BaselineAgent] non-finite action {action}; perception="
                f"{perception}; estimate={boat_estimate}; pos={state['position']}; "
                f"vel={state['velocity']}; rpy={state['attitude']}",
                file=sys.stderr,
            )
            # Safe fallback: hover throttle on every motor.
            action = self._hover_action.copy()
        return action

    def act_setpoint(self, obs: Dict) -> tuple:
        """FC-compatible 4-DoF setpoint (thrust_norm, roll, pitch, yaw_rate).

        Thin delegate: runs the full act() pipeline (the only actuator
        path in this agent), then returns the high-level setpoint that
        control() cached as a side effect. Motor throttles computed by
        act() are discarded by the evaluator in this mode. Keeping a
        single pipeline path guarantees setpoint and motor outputs
        cannot diverge.
        """
        _ = self.act(obs)
        sp = self._last_setpoint
        if not all(np.isfinite(v) for v in sp):
            import sys
            print(
                f"[BaselineAgent] non-finite setpoint {sp}",
                file=sys.stderr,
            )
            sp = (0.0, 0.0, 0.0, 0.0)
        return sp

    def get_last_estimate(self) -> Optional[Dict]:
        """Expose the most recent boat-position estimate so the scorer can
        compute the optional estimation-bonus RMSE. Returning None opts
        out of the bonus."""
        if self._last_estimate is None or self._last_estimate.get("position") is None:
            return None
        return {
            "position": np.asarray(self._last_estimate["position"]),
            "velocity": np.asarray(self._last_estimate.get("velocity", np.zeros(3))),
        }

    # ------------------------------------------------------------------ perception
    def perceive(self, camera_image: np.ndarray) -> Dict:
        gray = cv2.cvtColor(camera_image, cv2.COLOR_RGB2GRAY)
        gray = self._clahe.apply(gray)
        if self._detector is not None:
            corners, ids, _ = self._detector.detectMarkers(gray)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(
                gray, self._dictionary, parameters=self._detector_params
            )

        if ids is None or len(ids) == 0:
            return {"detected": False}

        # Only marker ID 0 matters for this challenge.
        target = None
        for c, i in zip(corners, ids.flatten()):
            if int(i) == 0:
                target = c
                break
        if target is None:
            return {"detected": False}

        image_points = target.reshape(4, 2).astype(np.float32)
        # Subpixel refinement: ArUco returns integer-pixel corners. solvePnP's
        # translation error scales with corner error, so spending ~0.05 ms per
        # frame here typically tightens tvec by 5-10x at altitude.
        cv2.cornerSubPix(
            gray,
            image_points,
            winSize=(5, 5),
            zeroZone=(-1, -1),
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01),
        )
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
        # IPPE_SQUARE silently emits NaN tvec when the four corners are
        # nearly collinear (near the FOV edge, extreme angles). Also
        # reject solutions where the marker is behind the camera or
        # absurdly far. Both manifest as garbage downstream.
        if not (np.all(np.isfinite(tvec)) and np.all(np.isfinite(rvec))):
            return {"detected": False}
        if tvec[2] <= 0.1 or tvec[2] > 100.0:
            return {"detected": False}

        return {
            "detected": True,
            "tvec": tvec,
            "rvec": rvec,
            "image_points": image_points,
        }

    # ------------------------------------------------------------------ estimation
    def estimate(self, perception: Dict, drone_state: Dict, history) -> Dict:
        # Always advance the KF one step. On the very first call this is a
        # no-op (filter not yet initialized); after the first update it
        # propagates position with the current velocity estimate, so gaps
        # in detection are bridged by predicted state instead of frozen
        # last-known position.
        self.kf.predict()

        if perception.get("detected", False):
            tvec = perception["tvec"]
            rvec = perception["rvec"]
            drone_pos = np.asarray(drone_state["position"], dtype=np.float64)
            R_world_body = rpy_to_matrix(drone_state["attitude"])
            body_x = R_world_body[:, 0]
            body_y = R_world_body[:, 1]
            body_z = R_world_body[:, 2]
            # Camera is mounted below the drone (under the fuselage),
            # offset by CAMERA_BODY_OFFSET_Z along body -z. tvec from
            # solvePnP is relative to the *camera*, not the drone COM,
            # so transform from camera frame and add to camera world pos.
            camera_world = drone_pos + body_z * _CAMERA_BODY_OFFSET_Z
            # Mapping derived from the env's view-matrix construction:
            # OpenCV camera +X (image right) = -body_y in world,
            # OpenCV camera +Y (image down) = -body_x in world,
            # OpenCV camera +Z (into scene) = -body_z in world.
            R_cam_to_world = np.column_stack([-body_y, -body_x, -body_z])
            marker_world = camera_world + R_cam_to_world @ tvec
            self.kf.update(marker_world)
            self._last_marker_world = marker_world
            self._frames_since_detection = 0

            # Boat tilt from marker rotation. rvec rotates marker-local
            # axes into the camera frame; reuse R_cam_to_world (already
            # built for tvec) to compose the marker frame in world.
            R_cam_marker, _ = cv2.Rodrigues(rvec)
            R_world_marker = R_cam_to_world @ R_cam_marker
            # Marker normal in world = R_world_marker[:, 2]. On a level
            # deck this is +z_world. Angle off vertical is the scalar
            # decide() actually needs for "wait for level" gating, and
            # is invariant to IPPE_SQUARE yaw flips.
            up = R_world_marker[:, 2]
            cos_tilt = float(np.clip(up[2], -1.0, 1.0))
            boat_tilt = float(np.arccos(cos_tilt))
            # Optional decomposed roll/pitch (PyBullet R = Rz Ry Rx),
            # exposed for axis-specific gating if useful.
            boat_pitch = float(np.arctan2(
                -R_world_marker[2, 0],
                np.sqrt(R_world_marker[2, 1] ** 2 + R_world_marker[2, 2] ** 2),
            ))
            boat_roll = float(np.arctan2(
                R_world_marker[2, 1], R_world_marker[2, 2]
            ))
            self._last_boat_roll = boat_roll
            self._last_boat_pitch = boat_pitch

            return {
                "position": self.kf.position,
                "velocity": self.kf.velocity,
                "fresh": True,
                "stale_steps": 0,
                "from_prior": False,
                "measurement_world": marker_world,
                "boat_tilt": boat_tilt,
                "boat_roll": boat_roll,
                "boat_pitch": boat_pitch,
            }

        # No fresh detection — KF coasts on its prediction.
        self._frames_since_detection += 1
        if self.kf.initialized:
            return {
                "position": self.kf.position,
                "velocity": self.kf.velocity,
                "fresh": False,
                "stale_steps": self._frames_since_detection,
                "from_prior": False,
                # boat_tilt is a SNAPSHOT measurement: a stale value tells
                # us nothing about whether the deck is level *now*, so we
                # omit it (None). Roll/pitch are still useful as a hint
                # over very short gaps, so we carry them forward.
                "boat_tilt": None,
                "boat_roll": self._last_boat_roll,
                "boat_pitch": self._last_boat_pitch,
            }
        # Cold start: filter never initialized -> use the search prior.
        return {
            "position": np.array(
                [self.SEARCH_PRIOR_XY[0], self.SEARCH_PRIOR_XY[1], 0.3]
            ),
            "velocity": np.zeros(3),
            "fresh": False,
            "stale_steps": self._frames_since_detection,
            "from_prior": True,
        }

    # ------------------------------------------------------------------ decision
    def decide(
        self, drone_state: Dict, boat_estimate: Dict, battery: float, time_s: float
    ) -> str:
        self._step_count += 1

        pos = np.asarray(drone_state["position"], dtype=np.float64)
        target = boat_estimate.get("position")
        if target is None or boat_estimate.get("from_prior", False):
            self._land_wait_since_step = None
            return PHASE_SEARCH

        horiz = float(np.linalg.norm(pos[:2] - np.asarray(target[:2])))
        z_above = float(pos[2] - target[2])

        # ABORT triggers: only meaningful while we are already committing
        # to land (DESCEND or LAND). If any of these fires, pin the next
        # 1.5 s to APPROACH so the drone climbs back, stabilizes, and
        # re-acquires before another LAND attempt.
        if self.phase in (PHASE_DESCEND, PHASE_LAND):
            stale = int(boat_estimate.get("stale_steps", 0)) > 25   # 0.5 s
            drifted = horiz > 0.8                                   # off-platform
            if stale or drifted:
                self._abort_until_step = self._step_count + 75      # 1.5 s

        # If inside an abort window, force APPROACH regardless of the
        # cascade below. Climbing to PHASE_ALTITUDE[APPROACH] (3 m) is
        # handled by the existing z-PID + target_z mapping.
        if self._step_count < self._abort_until_step:
            self._land_wait_since_step = None
            return PHASE_APPROACH

        # WIND_EST: keep the phase active for its full window. When the
        # window closes, mark it done and fall through to normal decision
        # logic; control() picks up _wind_est_v0 to compute the estimate.
        if self._wind_est_started_step is not None:
            elapsed = self._step_count - self._wind_est_started_step
            if elapsed < self.WIND_EST_STEPS:
                return PHASE_WIND_EST
            self._wind_est_done = True
            self._wind_est_started_step = None

        # Trigger WIND_EST once, when safe: KF locked on, well above the
        # marker, drone roughly stationary (no inherited velocity to coast
        # through the window), and currently approaching/searching.
        drone_vel = np.asarray(drone_state["velocity"], dtype=np.float64)
        drone_stable = bool(np.linalg.norm(drone_vel) < 4.0)
        if (not self._wind_est_done
                and self.kf.initialized
                and z_above > self.WIND_EST_MIN_ALT
                and drone_stable
                and self.phase in (PHASE_SEARCH, PHASE_APPROACH)):
            self._wind_est_started_step = self._step_count
            return PHASE_WIND_EST

        # TILT GATE: in the LAND-ready altitude band, only commit when
        # the deck is roughly level. Otherwise hold DESCEND (hover near
        # the marker) and wait for the next zero-crossing of the wave
        # oscillation. Safety timeout: if we've been waiting too long
        # (boat in heavy seas, noisy rvec, etc.), commit anyway so we
        # don't burn battery hovering forever.
        if z_above < self.DESCEND_ALTITUDE_OK:
            boat_tilt = boat_estimate.get("boat_tilt")
            # boat_tilt is None when info is unavailable (cold-start or
            # the post-detection branch). In that case don't block LAND.
            deck_level = (boat_tilt is None) or (boat_tilt < self.LAND_TILT_MAX_RAD)
            if deck_level:
                self._land_wait_since_step = None
                return PHASE_LAND
            if self._land_wait_since_step is None:
                self._land_wait_since_step = self._step_count
            waited = self._step_count - self._land_wait_since_step
            if waited > self.LAND_WAIT_MAX_STEPS:
                self._land_wait_since_step = None
                return PHASE_LAND                  # safety timeout
            return PHASE_DESCEND                   # hover, wait for level

        # Out of the LAND-ready band — clear the wait timer so the next
        # entry to the band starts fresh.
        self._land_wait_since_step = None

        if horiz < self.DESCEND_HORIZ_OK and z_above < self.APPROACH_ALTITUDE_OK:
            return PHASE_DESCEND
        if horiz < self.APPROACH_HORIZ_OK:
            return PHASE_APPROACH
        return PHASE_APPROACH if boat_estimate.get("fresh", False) else PHASE_SEARCH

    # ------------------------------------------------------------------ control
    def control(
        self, drone_state: Dict, boat_estimate: Dict, phase: str
    ) -> np.ndarray:
        # Reset PID integrators on phase transitions. Without this, the
        # z-PID's integrator saturates during a prolonged LAND (err is
        # always large-negative against the -0.5 m target) and carries
        # that bias into a subsequent APPROACH, e.g. when a recovery
        # scenario briefly loses the marker and forces re-search.
        if self._prev_phase is not None and phase != self._prev_phase:
            self.pid_x.reset()
            self.pid_y.reset()
            self.pid_z.reset()
            self.pid_yaw.reset()
        self._prev_phase = phase

        # Velocity is needed both for the WIND_EST snapshot and for the
        # main flow; hoist it before any phase-specific branching.
        vel_world = np.asarray(drone_state["velocity"], dtype=np.float64)

        # WIND_EST: level attitude + altitude-hold, short-circuit the rest
        # of control. Wind only acts horizontally, so a small z PD doesn't
        # contaminate the xy drift estimate. Without altitude hold, any
        # vertical velocity inherited from the prior phase persists
        # through hover (thrust_norm=0 only cancels gravity, not inertia)
        # and the drone crashes after WIND_EST ends.
        if phase == PHASE_WIND_EST:
            if self._wind_est_v0 is None:
                self._wind_est_v0 = vel_world.copy()
                self._wind_est_z0 = float(drone_state["position"][2])
            z_err = float(self._wind_est_z0 - drone_state["position"][2])
            v_z = float(vel_world[2])
            thrust_z = float(np.clip(0.4 * z_err - 0.8 * v_z, -1.0, 1.0))
            self._last_setpoint = (thrust_z, 0.0, 0.0, 0.0)
            return self.attitude_ctrl(
                drone_state,
                roll_des=0.0, pitch_des=0.0,
                yaw_rate_des=0.0, thrust_norm=thrust_z,
            )

        # First tick after WIND_EST: turn (v_now - v0) into the wind
        # acceleration estimate. One-shot: clear v0 so this never fires
        # again (unless WIND_EST is rearmed in a future iteration).
        if self._wind_est_v0 is not None:
            elapsed = self.WIND_EST_STEPS * self.DT
            wind = (vel_world[:2] - self._wind_est_v0[:2]) / elapsed
            self._wind_accel_w = np.clip(wind, -5.0, 5.0)
            self._wind_est_v0 = None

        pos = np.asarray(drone_state["position"], dtype=np.float64)
        target = boat_estimate.get("position")
        if target is None:
            target = np.array(
                [self.SEARCH_PRIOR_XY[0], self.SEARCH_PRIOR_XY[1], 0.3]
            )

        target_z = float(target[2]) + self.PHASE_ALTITUDE[phase]

        # World-frame errors -> body-frame errors so the controller works
        # regardless of yaw. The baseline never actively yaws, but this
        # keeps things correct if anything perturbs heading.
        R = rpy_to_matrix(drone_state["attitude"])
        vel_body = R.T @ vel_world
        err_world = np.array(
            [target[0] - pos[0], target[1] - pos[1], target_z - pos[2]],
            dtype=np.float64,
        )
        err_body = R.T @ err_world

        # Velocity feed-forward: subtract the boat's estimated velocity
        # from the drone's velocity before passing it to the PID D-term.
        # The PID then "sees" the RELATIVE velocity. At steady state with
        # zero position error, drone_vel = boat_vel makes the D-term zero —
        # i.e. the drone naturally matches the boat's motion instead of
        # lagging. Equivalent to err_dot = (boat_vel - drone_vel) in 'D on
        # error' form, but smoother (KF-filtered boat_vel, not noisy diff).
        # Clip as a safety against a transient KF blow-up at cold start.
        boat_vel_world = np.asarray(
            boat_estimate.get("velocity", np.zeros(3)), dtype=np.float64
        )
        boat_vel_world = np.clip(boat_vel_world, -5.0, 5.0)
        boat_vel_body = R.T @ boat_vel_world
        rel_vel_body = vel_body - boat_vel_body
        rel_vel_world_z = float(vel_world[2] - boat_vel_world[2])

        # Sign mapping: a positive forward (body +x) error means the target
        # is ahead, which calls for a positive pitch (which rotates body z
        # forward, accelerating +x). A positive left (body +y) error means
        # target is to the left, which calls for a negative roll
        # (-roll_cmd accelerates +y).
        #
        # Thrust uses err_world[2] (NOT err_body[2]): altitude is
        # controlled in the world frame, not the body frame.
        pitch_cmd = self.pid_x(err_body[0], self.DT, measurement_velocity=rel_vel_body[0])
        roll_cmd = -self.pid_y(err_body[1], self.DT, measurement_velocity=rel_vel_body[1])
        thrust_cmd = self.pid_z(err_world[2], self.DT, measurement_velocity=rel_vel_world_z)

        # WIND FEED-FORWARD: cancel the measured wind acceleration by
        # tilting into it. World -> body rotation uses yaw only (the
        # required tilt is a body-frame quantity). K_FF inverts the
        # attitude controller's pitch_norm -> acceleration gain;
        # 0.20 assumes max_tilt ~ 0.5 rad (verify with controllers.py).
        if self._wind_est_done and phase in (PHASE_APPROACH, PHASE_DESCEND):
            yaw = float(drone_state["attitude"][2])
            cy, sy = float(np.cos(yaw)), float(np.sin(yaw))
            wx_w, wy_w = float(self._wind_accel_w[0]), float(self._wind_accel_w[1])
            wind_body_x =  cy * wx_w + sy * wy_w
            wind_body_y = -sy * wx_w + cy * wy_w
            K_FF = 0.20
            pitch_ff = float(np.clip( K_FF * wind_body_x, -0.3, 0.3))
            roll_ff  = float(np.clip(-K_FF * wind_body_y, -0.3, 0.3))
            pitch_cmd = float(np.clip(pitch_cmd + pitch_ff, -1.0, 1.0))
            roll_cmd  = float(np.clip(roll_cmd  + roll_ff,  -1.0, 1.0))

        # SOFT DESCENT (LAND phase): instead of bang-bang thrust_cmd=-1.0,
        # track a velocity reference that decreases with height. Goal:
        # |v_z| < 1.0 m/s at touchdown (soft-landing bonus threshold) and
        # break through the ground-effect cushion of the reference sim.
        if phase == PHASE_LAND:
            h = max(float(pos[2] - target[2]), 0.0)        # height above marker
            # Schedule: -1.0 m/s far up, easing to -0.3 m/s near contact.
            v_des = -float(np.clip(0.3 + 0.5 * h, 0.3, 1.0))
            # Inner velocity loop: bias keeps the drone descending when
            # v_z == v_des; the gain (1.5) corrects deviations. Upper
            # clip is +1 (not 0) so that when v_z << v_des — i.e. drone
            # has built up a fast descent inherited from the prior phase
            # — the loop can apply upward thrust to brake. The formula
            # naturally settles around hover (or just below) once v_z
            # tracks v_des, so the drone does not bounce.
            v_z = float(vel_world[2])
            thrust_cmd = -0.3 + 1.5 * (v_des - v_z)
            thrust_cmd = float(np.clip(thrust_cmd, -1.0, 1.0))

        # ACTIVE YAW (wing-perpendicular landing condition).
        # Boat heading is derived from the KF velocity estimate (atan2 of
        # vx,vy); this avoids touching estimate(). Gate on a minimum speed
        # so we don't yaw on noise when the boat is essentially stationary.
        # Two valid alignments 180 deg apart -> collapse the error onto
        # [-pi/2, +pi/2] to always take the shortest rotation.
        yaw_rate_cmd = 0.0
        if phase in (PHASE_APPROACH, PHASE_DESCEND, PHASE_LAND):
            boat_v_xy = boat_vel_world[:2]
            if float(np.linalg.norm(boat_v_xy)) > 0.2:
                boat_heading = float(np.arctan2(boat_v_xy[1], boat_v_xy[0]))
                drone_yaw = float(drone_state["attitude"][2])
                yaw_err = (boat_heading - drone_yaw + np.pi) % (2 * np.pi) - np.pi
                if yaw_err > np.pi / 2:
                    yaw_err -= np.pi
                elif yaw_err < -np.pi / 2:
                    yaw_err += np.pi
                yaw_rate_cmd = self.pid_yaw(yaw_err, self.DT)

        # TILT COMPENSATION on thrust. The attitude controller produces
        # a body-z force F_body = hover * (1 + thrust_cmd * 0.5). The
        # *vertical* force in world is F_body * R[2,2], so a tilted
        # drone gets less vertical force than the z-PID asked for. While
        # chasing the boat horizontally the drone tilts a lot, vertical
        # force drops, and the drone falls. Solve for thrust_new such
        # that the vertical force matches the original PID command:
        #   (1 + thrust_new*0.5) * R[2,2] = (1 + thrust_orig*0.5)
        Rzz = max(float(R[2, 2]), 0.3)   # floor for extreme tilts
        thrust_cmd = ((1.0 + thrust_cmd * 0.5) / Rzz - 1.0) / 0.5
        thrust_cmd = float(np.clip(thrust_cmd, -1.0, 1.0))

        # HARD SAFETY BRAKE: even with tilt comp, motor headroom is
        # limited at high bank angles and the brake may not fully halt
        # the fall. Trigger earlier (at -1.5 m/s) so the brake has time
        # to bite before ground impact.
        if float(vel_world[2]) < -1.5:
            thrust_cmd = 1.0

        # Cache the 4-DoF FC setpoint just before the mixer. act_setpoint()
        # reads from here, so the setpoint and motor outputs come from the
        # same compute path and can never diverge.
        self._last_setpoint = (
            float(thrust_cmd), float(roll_cmd),
            float(pitch_cmd), float(yaw_rate_cmd),
        )

        # Hand the high-level (thrust, roll, pitch, yaw_rate) setpoints to
        # the stock attitude controller, which mixes them into per-motor
        # throttles using the drone spec's geometry and propeller params.
        return self.attitude_ctrl(
            drone_state,
            roll_des=roll_cmd,
            pitch_des=pitch_cmd,
            yaw_rate_des=yaw_rate_cmd,
            thrust_norm=thrust_cmd,
        )


# Module-level entry point. evaluation/evaluate.py imports the module
# dynamically and looks for a callable named `make_agent` (preferred) or a
# class named `Agent`. We support both for convenience.
Agent = BaselineAgent


def make_agent(drone_spec: Optional[DroneSpec] = None) -> BaselineAgent:
    return BaselineAgent(drone_spec)
