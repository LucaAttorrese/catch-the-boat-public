"""Kalman + MPC agent for the Catch the Boat challenge.

Architecture
============
perceive()  – ArUco detection  (same as baseline)
estimate()  – 6-D constant-velocity Kalman filter for the boat
               predict() every tick  →  update() on each camera detection
decide()    – Phase state machine   (SEARCH → APPROACH → DESCEND → LAND)
control()   – 3-axis decoupled linear MPC replaces the baseline PIDs

MPC design
----------
Each world axis (x, y, z) is an independent discrete-time double integrator:

    p[k+1] = p[k] + v[k]*dt
    v[k+1] = v[k] + a[k]*dt
    state = [p, v],  input = a (acceleration)

`AxisMPC` precomputes the optimal gain matrix K offline using the standard
batch-form unconstrained QP solution, then clips the first input to hard
acceleration limits at runtime.  The online cost is two (N×N) × N
matrix-vector products — well under 1 ms for N = 20.

`DroneMPC` wraps three `AxisMPC` instances (one per axis) and:
  – Builds an N-step moving reference from the Kalman boat position + velocity.
  – Applies battery-aware vertical thrust limits.
  – Converts the optimal world-frame accelerations to the attitude-controller
    setpoints (roll_des, pitch_des, thrust_norm).

Physical limits are taken directly from DefaultAttitudeController's defaults:
  max_tilt    = 0.30 rad  →  A_XY_MAX = g * sin(0.30) ≈ 2.94 m/s²
  thrust_range = 0.50     →  A_Z_MAX  = 0.50 * g      ≈ 4.91 m/s²
"""

from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from boat_landing.camera import CAMERA_BODY_OFFSET_Z, get_intrinsics
from boat_landing.controllers import DefaultAttitudeController
from boat_landing.drone_interface import DroneSpec, load_drone_spec


REPO_ROOT = Path(__file__).resolve().parent.parent

MARKER_SIZE_M = 0.8

PHASE_SEARCH  = "SEARCH"
PHASE_APPROACH = "APPROACH"
PHASE_DESCEND = "DESCEND"
PHASE_LAND    = "LAND"

_G            = 9.81    # m/s²
_MAX_TILT     = 0.30    # rad  (DefaultAttitudeController.DEFAULT_MAX_TILT)
_THRUST_RANGE = 0.50    # fraction (DefaultAttitudeController.DEFAULT_THRUST_RANGE)


def rpy_to_matrix(rpy) -> np.ndarray:
    """World-from-body rotation matrix (PyBullet RPY: R = Rz @ Ry @ Rx)."""
    r, p_, y = float(rpy[0]), float(rpy[1]), float(rpy[2])
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p_), np.sin(p_)
    cy, sy = np.cos(y), np.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


# ══════════════════════════════════════════════════════════════════════════════
# Kalman filter
# ══════════════════════════════════════════════════════════════════════════════

class BoatKalmanFilter:
    """6-D constant-velocity Kalman filter for boat position + velocity.

    State:       x = [px, py, pz, vx, vy, vz]
    Measurement: z = [px, py, pz]  (ArUco world-frame detection)

    Process noise Q uses the DWNA model (discrete white-noise acceleration):
    treats unknown boat acceleration as zero-mean white noise, which gives the
    physically correct position/velocity cross-terms.
    """

    def __init__(self, dt: float, sigma_accel: float = 2.0, sigma_meas: float = 0.10):
        self._initialized = False
        dt2, dt3, dt4 = dt**2, dt**3, dt**4

        self.F = np.eye(6)
        self.F[0, 3] = self.F[1, 4] = self.F[2, 5] = dt

        self.H = np.zeros((3, 6))
        self.H[0, 0] = self.H[1, 1] = self.H[2, 2] = 1.0

        q = sigma_accel ** 2
        self.Q = np.zeros((6, 6))
        for i in range(3):
            self.Q[i,     i    ] = q * dt4 / 4.0
            self.Q[i,     i + 3] = q * dt3 / 2.0
            self.Q[i + 3, i    ] = q * dt3 / 2.0
            self.Q[i + 3, i + 3] = q * dt2

        self.R = np.eye(3) * sigma_meas**2
        self.x = np.zeros(6)
        self.P = np.eye(6)

    @property
    def initialized(self) -> bool:
        return self._initialized

    @property
    def position(self) -> np.ndarray:
        return self.x[:3].copy()

    @property
    def velocity(self) -> np.ndarray:
        return self.x[3:].copy()

    def initialize(self, pos: np.ndarray) -> None:
        self.x = np.zeros(6)
        self.x[:3] = pos
        self.P = np.diag([0.25, 0.25, 0.10, 4.0, 4.0, 0.25])
        self._initialized = True

    def predict(self) -> None:
        if not self._initialized:
            return
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z: np.ndarray) -> None:
        if not self._initialized:
            self.initialize(z)
            return
        y    = z - self.H @ self.x
        S    = self.H @ self.P @ self.H.T + self.R
        K    = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I_KH = np.eye(6) - K @ self.H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T   # Joseph form


# ══════════════════════════════════════════════════════════════════════════════
# MPC — single axis
# ══════════════════════════════════════════════════════════════════════════════

class AxisMPC:
    """Closed-form linear MPC for a 1-D discrete-time double integrator.

    All gain matrices are computed once in __init__.  The solve() call
    is two matrix-vector products → O(N²), negligible at N = 20.

    State   [p, v]  (position, velocity)
    Input   a       (acceleration)
    Cost    Σ_{k=1}^{N}  q*(p_k − r_k)²  +  r*a_{k−1}²

    The reference r_k can be time-varying (e.g. a moving boat prediction).
    Box constraints [a_min, a_max] are enforced by clipping U[0].
    """

    def __init__(
        self,
        dt:    float,
        N:     int,
        q:     float,
        r:     float,
        a_min: float,
        a_max: float,
    ) -> None:
        self.a_min = a_min
        self.a_max = a_max

        dt2 = dt * dt
        ks  = np.arange(1, N + 1, dtype=float)

        # Phi_p (N×2): free-response positions.  Row k → [1, (k+1)*dt]
        Phi_p = np.column_stack([np.ones(N), ks * dt])

        # G (N×N): input-to-position map.
        # G[k, j] = (0.5 + k − j) * dt²  for j ≤ k,  else 0
        # Derivation: A^n = [[1, n*dt],[0,1]]; B = [dt²/2, dt]
        # (A^(k−j) @ B)[0] = dt²/2 + (k−j)*dt² = (0.5 + k − j)*dt²
        G = np.zeros((N, N))
        for k in range(N):
            for j in range(k + 1):
                G[k, j] = (0.5 + k - j) * dt2

        Q_bar = np.eye(N) * q
        R_bar = np.eye(N) * r
        H     = G.T @ Q_bar @ G + R_bar
        # Optimal gain: U* = K @ (p_ref − p_free)
        self._K     = np.linalg.solve(H, G.T @ Q_bar)   # (N, N)
        self._Phi_p = Phi_p                              # (N, 2)

    def solve(self, pos: float, vel: float, p_ref: np.ndarray) -> float:
        """Return the optimal first acceleration, clipped to [a_min, a_max].

        pos, vel : current state scalars
        p_ref    : (N,) array of N future target positions
        """
        x0     = np.array([pos, vel])
        p_free = self._Phi_p @ x0             # free response (no input)
        U      = self._K @ (p_ref - p_free)   # optimal input sequence
        return float(np.clip(U[0], self.a_min, self.a_max))


# ══════════════════════════════════════════════════════════════════════════════
# MPC — 3-axis drone wrapper
# ══════════════════════════════════════════════════════════════════════════════

class DroneMPC:
    """3-axis decoupled linear MPC for drone guidance.

    One AxisMPC per axis per phase is precomputed (12 instances total).
    At runtime, solve() builds an N-step reference from the Kalman estimate
    and returns (roll_cmd, pitch_cmd, yaw_rate_cmd, thrust_cmd) ∈ [−1, 1]⁴
    for DefaultAttitudeController.

    Physical calibration
    --------------------
    DefaultAttitudeController maps:
      pitch_des ∈ [−1,+1] → ±max_tilt rad of nose pitch  → ax ≈ g*max_tilt m/s²
      roll_des  ∈ [−1,+1] → ±max_tilt rad of bank roll   → ay ≈ g*max_tilt m/s²
      thrust_norm ∈ [−1,+1] → hover ± thrust_range*hover  → az ≈ ±thrust_range*g

    So the inverse mapping from desired world-frame acceleration is:
      pitch_cmd  =  ax_body / (g * max_tilt)
      roll_cmd   = −ay_body / (g * max_tilt)
      thrust_cmd =  az_world / (thrust_range * g)

    Battery awareness
    -----------------
    When battery < 30 % the upward thrust budget is scaled back.
    When battery < 12 % the agent forces immediate descent.
    """

    DT = 0.02    # must match BoatLandingEnv.DT
    N  = 20      # horizon steps (= 0.4 s lookahead)

    # Physical acceleration limits derived from DefaultAttitudeController
    A_XY_MAX = _G * np.sin(_MAX_TILT)   # ≈ 2.94 m/s²
    A_Z_MAX  = _G * _THRUST_RANGE       # ≈ 4.91 m/s²

    # Target altitude above boat deck per phase
    PHASE_ALT = {
        PHASE_SEARCH:   5.5,
        PHASE_APPROACH: 3.0,
        PHASE_DESCEND:  0.4,
        PHASE_LAND:    -0.5,
    }

    # Desired descent speed during LAND phase (m/s, negative = downward)
    LAND_VEL = -0.7

    # MPC cost weights (q_xy, r_xy, q_z, r_z) per phase.
    # Higher q → tighter tracking.  Higher r → more conservative actuation.
    _W = {
        PHASE_SEARCH:  ( 40.0, 1.0,  50.0, 0.8),
        PHASE_APPROACH:(100.0, 0.5,  70.0, 0.5),
        PHASE_DESCEND: (150.0, 0.3, 120.0, 0.3),
        PHASE_LAND:    (180.0, 0.2, 160.0, 0.2),
    }

    def __init__(self) -> None:
        a_xy = self.A_XY_MAX
        a_z  = self.A_Z_MAX
        dt, N = self.DT, self.N

        self._mpc: Dict[str, Dict[str, AxisMPC]] = {}
        for phase, (q_xy, r_xy, q_z, r_z) in self._W.items():
            self._mpc[phase] = {
                "x": AxisMPC(dt, N, q_xy, r_xy, -a_xy,  a_xy),
                "y": AxisMPC(dt, N, q_xy, r_xy, -a_xy,  a_xy),
                "z": AxisMPC(dt, N, q_z,  r_z,  -a_z,   a_z),
            }

    def solve(
        self,
        drone_state:   Dict,
        boat_estimate: Dict,
        phase:         str,
        battery:       float,
    ) -> Tuple[float, float, float, float]:
        """Compute optimal attitude setpoints.

        Returns (roll_cmd, pitch_cmd, yaw_rate_cmd, thrust_cmd) each in [-1, 1].
        """
        pos = np.asarray(drone_state["position"], dtype=np.float64)
        vel = np.asarray(drone_state["velocity"], dtype=np.float64)
        R   = rpy_to_matrix(drone_state["attitude"])

        from_prior = boat_estimate.get("from_prior", True)
        boat_pos   = np.asarray(
            boat_estimate.get("position", [0.0, 0.0, 0.3]), dtype=np.float64
        )
        boat_vel   = np.asarray(
            boat_estimate.get("velocity", np.zeros(3)), dtype=np.float64
        )

        # ── Future time instants ─────────────────────────────────────────
        t = np.arange(1, self.N + 1, dtype=float) * self.DT   # (N,) seconds ahead

        # ── X/Y reference: predict the boat's position along its velocity ─
        # If we're on the cold-start prior the velocity is unknown → no FF.
        v_ff = np.zeros(3) if from_prior else boat_vel
        p_ref_x = boat_pos[0] + v_ff[0] * t
        p_ref_y = boat_pos[1] + v_ff[1] * t

        # ── Z reference: altitude target ─────────────────────────────────
        target_z = float(boat_pos[2]) + self.PHASE_ALT.get(phase, 3.0)
        p_ref_z  = np.full(self.N, target_z)

        # LAND: ramp downward from current altitude at LAND_VEL m/s.
        # The ramp is floored at target_z so we stop at the deck level.
        if phase == PHASE_LAND:
            ramp    = pos[2] + self.LAND_VEL * t
            p_ref_z = np.maximum(target_z, ramp)

        # ── Battery-aware thrust limit ───────────────────────────────────
        mpc_set = self._mpc.get(phase, self._mpc[PHASE_APPROACH])
        bat     = float(np.clip(battery, 0.25, 1.0))
        mpc_set["z"].a_max = self.A_Z_MAX * bat   # scale back upward thrust

        # Emergency: battery nearly exhausted → drop immediately
        if battery < 0.12:
            p_ref_z = np.full(self.N, pos[2] - 10.0)

        # ── Solve per-axis MPC (world frame) ────────────────────────────
        ax_w = mpc_set["x"].solve(pos[0], vel[0], p_ref_x)
        ay_w = mpc_set["y"].solve(pos[1], vel[1], p_ref_y)
        az_w = mpc_set["z"].solve(pos[2], vel[2], p_ref_z)

        # ── World → body for horizontal axes (only yaw matters at small tilt) ─
        a_body_h = R.T @ np.array([ax_w, ay_w, 0.0])
        ax_b = a_body_h[0]
        ay_b = a_body_h[1]

        # ── Map to normalised attitude commands ─────────────────────────
        pitch_cmd  = float(np.clip( ax_b / self.A_XY_MAX, -1.0, 1.0))
        roll_cmd   = float(np.clip(-ay_b / self.A_XY_MAX, -1.0, 1.0))
        thrust_cmd = float(np.clip( az_w / self.A_Z_MAX,  -1.0, 1.0))

        # LAND: enforce minimum downward thrust to push through ground effect
        if phase == PHASE_LAND:
            thrust_cmd = min(thrust_cmd, -0.5)

        return roll_cmd, pitch_cmd, 0.0, thrust_cmd


# ══════════════════════════════════════════════════════════════════════════════
# Agent
# ══════════════════════════════════════════════════════════════════════════════

class KalmanAgent:
    """ArUco detector → Kalman filter → phase state machine → MPC."""

    DT = 0.02

    SEARCH_PRIOR_XY   = np.array([0.0, 0.0])
    APPROACH_HORIZ_OK = 0.6
    APPROACH_ALT_OK   = 3.5
    DESCEND_HORIZ_OK  = 0.4
    DESCEND_ALT_OK    = 0.6

    def __init__(self, drone_spec: Optional[DroneSpec] = None):
        if drone_spec is None:
            drone_spec = load_drone_spec(REPO_ROOT / "drones" / "vtol.yaml")
        self.spec = drone_spec
        self.attitude_ctrl = DefaultAttitudeController(drone_spec)
        self._hover_action = self.attitude_ctrl(
            {"attitude": np.zeros(3), "angular_velocity": np.zeros(3)},
            roll_des=0.0, pitch_des=0.0, yaw_rate_des=0.0, thrust_norm=0.0,
        )

        self.intrinsics  = get_intrinsics()
        self.dist_coeffs = np.zeros(5, dtype=np.float64)

        self._dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
        try:
            self._detector = cv2.aruco.ArucoDetector(
                self._dictionary, cv2.aruco.DetectorParameters()
            )
        except AttributeError:
            self._detector = None

        h = MARKER_SIZE_M / 2.0
        self._object_points = np.array(
            [[-h,  h, 0.0], [ h,  h, 0.0], [ h, -h, 0.0], [-h, -h, 0.0]],
            dtype=np.float32,
        )

        # Estimation
        self.kf = BoatKalmanFilter(dt=self.DT)

        # Control
        self.mpc = DroneMPC()

        # Bookkeeping
        self.phase                = PHASE_SEARCH
        self._frames_since_detect = 0
        self._last_estimate: Optional[Dict] = None
        # Expose last MPC commands so the HUD can read them
        self.last_cmd = {"roll": 0.0, "pitch": 0.0, "thrust": 0.0}

    # ── public API ──────────────────────────────────────────────────────────

    def act(self, obs: Dict) -> np.ndarray:
        camera  = obs["camera"]
        state   = obs["state"]
        battery = obs["battery"]
        time_s  = obs["time"]

        perception    = self.perceive(camera)
        boat_estimate = self.estimate(perception, state)
        self._last_estimate = boat_estimate
        self.phase    = self.decide(state, boat_estimate, battery, time_s)
        action        = self.control(state, boat_estimate, self.phase, battery)

        if not np.all(np.isfinite(action)):
            import sys
            print(
                f"[KalmanAgent] non-finite action — phase={self.phase} "
                f"pos={state['position']} est={boat_estimate}",
                file=sys.stderr,
            )
            action = self._hover_action.copy()
        return action

    def get_last_estimate(self) -> Optional[Dict]:
        if self._last_estimate is None or self._last_estimate.get("position") is None:
            return None
        return {
            "position": np.asarray(self._last_estimate["position"]),
            "velocity": np.asarray(self._last_estimate.get("velocity", np.zeros(3))),
        }

    # ── perception ─────────────────────────────────────────────────────────

    def perceive(self, image: np.ndarray) -> Dict:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        if self._detector is not None:
            corners, ids, _ = self._detector.detectMarkers(gray)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(gray, self._dictionary)

        if ids is None or len(ids) == 0:
            return {"detected": False}

        target = None
        for c, i in zip(corners, ids.flatten()):
            if int(i) == 0:
                target = c
                break
        if target is None:
            return {"detected": False}

        pts = target.reshape(4, 2).astype(np.float32)
        ok, rvec, tvec = cv2.solvePnP(
            self._object_points, pts, self.intrinsics, self.dist_coeffs,
            flags=cv2.SOLVEPNP_IPPE_SQUARE,
        )
        if not ok:
            return {"detected": False}

        tvec = np.asarray(tvec, dtype=np.float64).reshape(3)
        rvec = np.asarray(rvec, dtype=np.float64).reshape(3)
        if not (np.all(np.isfinite(tvec)) and np.all(np.isfinite(rvec))):
            return {"detected": False}
        if tvec[2] <= 0.1 or tvec[2] > 100.0:
            return {"detected": False}

        return {"detected": True, "tvec": tvec, "rvec": rvec}

    # ── estimation ─────────────────────────────────────────────────────────

    def _cam_to_world(self, tvec: np.ndarray, drone_state: Dict) -> np.ndarray:
        drone_pos      = np.asarray(drone_state["position"], dtype=np.float64)
        R_wb           = rpy_to_matrix(drone_state["attitude"])
        camera_world   = drone_pos + R_wb[:, 2] * CAMERA_BODY_OFFSET_Z
        R_cw           = np.column_stack([-R_wb[:, 1], -R_wb[:, 0], -R_wb[:, 2]])
        return camera_world + R_cw @ tvec

    def estimate(self, perception: Dict, drone_state: Dict) -> Dict:
        self.kf.predict()   # time-update every tick

        if perception.get("detected", False):
            marker_world = self._cam_to_world(perception["tvec"], drone_state)
            self.kf.update(marker_world)   # measurement-update
            self._frames_since_detect = 0
            return {
                "position":    self.kf.position,
                "velocity":    self.kf.velocity,
                "fresh":       True,
                "stale_steps": 0,
                "from_prior":  False,
            }

        self._frames_since_detect += 1
        if self.kf.initialized:
            return {
                "position":    self.kf.position,
                "velocity":    self.kf.velocity,
                "fresh":       False,
                "stale_steps": self._frames_since_detect,
                "from_prior":  False,
            }

        return {
            "position":    np.array([self.SEARCH_PRIOR_XY[0], self.SEARCH_PRIOR_XY[1], 0.3]),
            "velocity":    np.zeros(3),
            "fresh":       False,
            "stale_steps": self._frames_since_detect,
            "from_prior":  True,
        }

    # ── decision ───────────────────────────────────────────────────────────

    def decide(
        self, drone_state: Dict, boat_estimate: Dict, battery: float, time_s: float
    ) -> str:
        if boat_estimate.get("from_prior", True):
            return PHASE_SEARCH

        pos    = np.asarray(drone_state["position"], dtype=np.float64)
        target = np.asarray(boat_estimate["position"], dtype=np.float64)
        horiz  = float(np.linalg.norm(pos[:2] - target[:2]))
        z_above = float(pos[2] - target[2])

        if z_above < self.DESCEND_ALT_OK:
            return PHASE_LAND
        if horiz < self.DESCEND_HORIZ_OK and z_above < self.APPROACH_ALT_OK:
            return PHASE_DESCEND
        if horiz < self.APPROACH_HORIZ_OK or boat_estimate.get("fresh", False):
            return PHASE_APPROACH
        return PHASE_SEARCH

    # ── control ────────────────────────────────────────────────────────────

    def control(
        self,
        drone_state:   Dict,
        boat_estimate: Dict,
        phase:         str,
        battery:       float = 1.0,
    ) -> np.ndarray:
        roll, pitch, yaw_rate, thrust = self.mpc.solve(
            drone_state, boat_estimate, phase, battery
        )
        self.last_cmd = {"roll": roll, "pitch": pitch, "thrust": thrust}
        return self.attitude_ctrl(
            drone_state,
            roll_des=roll,
            pitch_des=pitch,
            yaw_rate_des=yaw_rate,
            thrust_norm=thrust,
        )


Agent = KalmanAgent


def make_agent(drone_spec: Optional[DroneSpec] = None) -> KalmanAgent:
    return KalmanAgent(drone_spec)
