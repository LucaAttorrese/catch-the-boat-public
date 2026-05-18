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

from boat_landing.camera import CAMERA_BODY_OFFSET_Z, get_intrinsics
from boat_landing.controllers import DefaultAttitudeController
from boat_landing.drone_interface import DroneSpec, load_drone_spec


REPO_ROOT = Path(__file__).resolve().parent.parent


# Must match BoatLandingEnv.MARKER_SIZE.
MARKER_SIZE_M = 0.8

PHASE_SEARCH = "SEARCH"
PHASE_APPROACH = "APPROACH"
PHASE_DESCEND = "DESCEND"
PHASE_LAND = "LAND"


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

        self.intrinsics = get_intrinsics()
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
        self.pid_x = PID(kp=0.15, ki=0.00, kd=0.40)
        self.pid_y = PID(kp=0.15, ki=0.00, kd=0.40)
        self.pid_z = PID(kp=0.10, ki=0.02, kd=0.30)
        self.pid_yaw = PID(kp=1.0, ki=0.0, kd=0.0)

        self.phase = PHASE_SEARCH
        self._prev_phase: Optional[str] = None
        self._last_marker_world: Optional[np.ndarray] = None
        self._frames_since_detection = 0
        self._last_estimate: Optional[Dict] = None

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
        if perception.get("detected", False):
            tvec = perception["tvec"]
            drone_pos = np.asarray(drone_state["position"], dtype=np.float64)
            R_world_body = rpy_to_matrix(drone_state["attitude"])
            body_x = R_world_body[:, 0]
            body_y = R_world_body[:, 1]
            body_z = R_world_body[:, 2]
            # Camera is mounted below the drone (under the fuselage),
            # offset by CAMERA_BODY_OFFSET_Z along body -z. tvec from
            # solvePnP is relative to the *camera*, not the drone COM,
            # so transform from camera frame and add to camera world pos.
            camera_world = drone_pos + body_z * CAMERA_BODY_OFFSET_Z
            # Mapping derived from the env's view-matrix construction:
            # OpenCV camera +X (image right) = -body_y in world,
            # OpenCV camera +Y (image down) = -body_x in world,
            # OpenCV camera +Z (into scene) = -body_z in world.
            R_cam_to_world = np.column_stack([-body_y, -body_x, -body_z])
            marker_world = camera_world + R_cam_to_world @ tvec
            self._last_marker_world = marker_world
            self._frames_since_detection = 0
            return {
                "position": marker_world,
                "velocity": np.zeros(3),
                "fresh": True,
                "stale_steps": 0,
                "from_prior": False,
            }

        # No fresh detection — fall back to last known if we have one.
        self._frames_since_detection += 1
        if self._last_marker_world is not None:
            return {
                "position": self._last_marker_world,
                "velocity": np.zeros(3),
                "fresh": False,
                "stale_steps": self._frames_since_detection,
                "from_prior": False,
            }
        # Cold start: use the search prior. Deliberately wrong as a hint to
        # teams: a real solution should *search* rather than guess.
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
        pos = np.asarray(drone_state["position"], dtype=np.float64)
        target = boat_estimate.get("position")
        if target is None or boat_estimate.get("from_prior", False):
            return PHASE_SEARCH

        horiz = float(np.linalg.norm(pos[:2] - np.asarray(target[:2])))
        z_above = float(pos[2] - target[2])

        if z_above < self.DESCEND_ALTITUDE_OK:
            return PHASE_LAND
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
        self._prev_phase = phase

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
        vel_world = np.asarray(drone_state["velocity"], dtype=np.float64)
        vel_body = R.T @ vel_world
        err_world = np.array(
            [target[0] - pos[0], target[1] - pos[1], target_z - pos[2]],
            dtype=np.float64,
        )
        err_body = R.T @ err_world

        # Sign mapping: a positive forward (body +x) error means the target
        # is ahead, which calls for a positive pitch (which rotates body z
        # forward, accelerating +x). A positive left (body +y) error means
        # target is to the left, which calls for a negative roll
        # (-roll_cmd accelerates +y).
        #
        # Thrust uses err_world[2] (NOT err_body[2]): altitude is
        # controlled in the world frame, not the body frame.
        #
        # PID D-term uses drone velocity ('D on measurement'), which is
        # smooth, instead of d(err)/dt, which spikes when the noisy boat
        # estimate snaps to a new detection. With d(err)/dt the
        # controller bang-bangs ±1 every step and the simulation
        # numerically diverges.
        pitch_cmd = self.pid_x(err_body[0], self.DT, measurement_velocity=vel_body[0])
        roll_cmd = -self.pid_y(err_body[1], self.DT, measurement_velocity=vel_body[1])
        thrust_cmd = self.pid_z(err_world[2], self.DT, measurement_velocity=vel_world[2])
        # In LAND, force thrust to the controller's lower bound (50% hover
        # thrust). The z-PID alone outputs ~ -0.10 at this altitude — over
        # the platform the reference sim's ground-effect amplification cancels
        # that small negative bias and the drone hovers ~0.2 m above the deck
        # forever. Full negative thrust pushes through the GE cushion (cap is
        # 2× per-motor in the ref sim, so worst-case F_eff ≈ mg at h ≈ 0.05 m)
        # and the drone touches down. Crude — descent is uncontrolled and
        # max_descent_velocity often saturates, killing the soft-landing
        # bonus. Smooth descent profiles + velocity feed-forward are left
        # as exercises.
        if phase == PHASE_LAND:
            thrust_cmd = -1.0
        yaw_rate_cmd = 0.0

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
