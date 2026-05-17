"""Template agent — your starting point.

Copy this file (or this whole repo) and start replacing the bodies of
`perceive`, `estimate`, `decide`, and `control`. Do NOT modify `act()` —
the orchestrator just chains those four methods together and is identical
to the baseline's. Keeping `act()` fixed makes A/B comparisons across
agents straightforward.

Action contract: `act()` returns a numpy array of shape
`(drone_spec.num_motors,)` with values in `[0, 1]` — one throttle per
motor. You can either:

    * Emit motor throttles directly from `control()` (richer for control-
      fidelity bonus), OR
    * Compute high-level setpoints (thrust, roll, pitch, yaw_rate) and
      let `DefaultAttitudeController` mix them into per-motor throttles
      (what the baseline does — easier path).

Each method below is documented with:
    1. What the baseline does.
    2. Why that's weak.
    3. Concrete improvement ideas, ordered roughly by impact.

Pick one improvement at a time, run `python evaluation/evaluate.py
--agent agents/agent_template.py --scenario medium` after each change,
and track the score.
"""

from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np

from boat_landing.camera import get_intrinsics
from boat_landing.controllers import DefaultAttitudeController
from boat_landing.drone_interface import DroneSpec, load_drone_spec


REPO_ROOT = Path(__file__).resolve().parent.parent


# Must match BoatLandingEnv.MARKER_SIZE.
MARKER_SIZE_M = 0.8

PHASE_SEARCH = "SEARCH"
PHASE_APPROACH = "APPROACH"
PHASE_DESCEND = "DESCEND"
PHASE_LAND = "LAND"


class TemplateAgent:
    DT = 0.02  # must match BoatLandingEnv.DT

    def __init__(self, drone_spec: Optional[DroneSpec] = None):
        # The env passes the active drone spec at construction. If you
        # instantiate this agent standalone (tests, notebooks), it falls
        # back to drones/vtol.yaml.
        if drone_spec is None:
            drone_spec = load_drone_spec(REPO_ROOT / "drones" / "vtol.yaml")
        self.spec: DroneSpec = drone_spec
        # Default attitude controller: maps (thrust, roll, pitch, yaw_rate)
        # setpoints to per-motor throttles. Replace with your own if you
        # want full motor-level control.
        self.attitude_ctrl = DefaultAttitudeController(drone_spec)

        self.intrinsics = get_intrinsics()
        self.dist_coeffs = np.zeros(5, dtype=np.float64)

        # Initialize anything you need across calls here:
        # - Detector(s)
        # - Filter state (mean, covariance, history buffer, ...)
        # - Phase/state-machine variables
        # - PIDs / controller objects
        self._dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
        try:
            self._detector = cv2.aruco.ArucoDetector(
                self._dictionary, cv2.aruco.DetectorParameters()
            )
        except AttributeError:
            self._detector = None

        self.phase = PHASE_SEARCH
        self._last_estimate: Optional[Dict] = None

    # ------------------------------------------------------------------ DO NOT MODIFY
    def act(self, obs: Dict) -> np.ndarray:
        """Top-level orchestrator. Identical to the baseline's. Leave
        this alone so the four pipeline stages stay easy to compare
        across agents."""
        camera = obs["camera"]
        state = obs["state"]
        battery = obs["battery"]
        time_s = obs["time"]

        perception = self.perceive(camera)
        boat_estimate = self.estimate(perception, state, history=None)
        self._last_estimate = boat_estimate
        self.phase = self.decide(state, boat_estimate, battery, time_s)
        action = self.control(state, boat_estimate, self.phase)
        return action

    def get_last_estimate(self) -> Optional[Dict]:
        """Optional: return your latest boat estimate for the scorer's
        estimation bonus. Should return a dict with keys 'position'
        (np.ndarray of shape (3,)) and optionally 'velocity'.

        Return None to opt out of the bonus."""
        return self._last_estimate

    # ------------------------------------------------------------------ OPTIONAL: FC-compatible output
    # If you implement `act_setpoint` (instead of, or in addition to,
    # `act`), the evaluator calls it INSTEAD of `act`. It must return
    # `(thrust_norm, roll, pitch, yaw_rate)` — the same 4-DoF setpoint
    # a real flight controller accepts via MAVLink SET_ATTITUDE_TARGET.
    # The runner converts those four numbers to per-motor throttles via
    # `DefaultAttitudeController` before calling env.step, so you do NOT
    # need to run the attitude controller yourself.
    #
    # Why bother:
    #   - Agents that emit setpoints port to PX4/Ardupilot OFFBOARD in an
    #     afternoon. Agents that emit motor throttles need a custom FC.
    #   - +5 HW-readiness points (see docs/AGENT_SCORING.md).
    #
    # Example (uncomment + adapt):
    #
    # def act_setpoint(self, obs: Dict) -> tuple:
    #     camera = obs["camera"]; state = obs["state"]
    #     perception   = self.perceive(camera)
    #     boat_est     = self.estimate(perception, state, history=None)
    #     self._last_estimate = boat_est
    #     self.phase   = self.decide(state, boat_est, obs["battery"], obs["time"])
    #     thrust_norm, roll, pitch, yaw_rate = self.control_setpoint(
    #         state, boat_est, self.phase
    #     )
    #     return (thrust_norm, roll, pitch, yaw_rate)

    # ------------------------------------------------------------------ STAGE 1: PERCEPTION
    def perceive(self, camera_image: np.ndarray) -> Dict:
        """Convert a (480, 640, 3) RGB uint8 frame into a perception dict.

        BASELINE: detect ArUco markers in grayscale and run cv2.solvePnP
        with IPPE_SQUARE for a single-marker pose. Returns
        {'detected': bool, 'tvec': (3,), 'rvec': (3,), 'image_points': (4,2)}.

        WHY IT'S WEAK:
        - Single detector. Brittle under motion blur or partial occlusion.
        - No subpixel refinement on corners.
        - Solves PnP only when the marker is fully in frame (4 corners).

        IMPROVEMENT IDEAS (rough impact order):
        1. Subpixel corner refinement: cv2.cornerSubPix on the four
           detected corners — small change, often noticeable PnP gain.
        2. Tune DetectorParameters (adaptiveThreshWinSizeMin/Max,
           cornerRefinementMethod = CORNER_REFINE_SUBPIX or CONTOUR).
        3. CLAHE / histogram equalization on the grayscale to be robust
           to varying lighting.
        4. Confidence score from reprojection error or detection size,
           which the estimator can use to weight measurements.
        5. Train a small CNN on synthetic frames to localize the marker
           when ArUco fails (motion blur, occlusion). The 0.8 m white
           plate is large and visually distinctive even without the code.
        """
        # TODO: replace with your perception pipeline.
        return {"detected": False}

    # ------------------------------------------------------------------ STAGE 2: ESTIMATION
    def estimate(
        self, perception: Dict, drone_state: Dict, history
    ) -> Dict:
        """Fuse the perception result with the drone state (and optionally
        a history buffer) to produce a boat estimate.

        Must return at minimum:
            {'position': np.ndarray shape (3,) | None,
             'velocity': np.ndarray shape (3,)}

        BASELINE: trusts every detection blindly; for missing detections
        it returns the last known position; with no history at all it
        returns a hardcoded prior (origin). No filtering, no velocity.

        WHY IT'S WEAK:
        - No measurement model: a noisy frame can produce a noisy world
          estimate that the controller will then chase.
        - No motion model: the boat is moving but we estimate velocity
          as zero, so the controller always lags.
        - Lost detections trigger no recovery behavior.

        IMPROVEMENT IDEAS:
        1. KALMAN FILTER on (x, y, vx, vy). State updates with constant-
           velocity model, measurements come from solvePnP. This single
           change typically dwarfs every other improvement.
        2. Use the drone-state velocity to subtract ego-motion when
           checking detection consistency.
        3. Outlier rejection: discard detections whose reprojection error
           or jump-from-previous is too large.
        4. Estimate boat heading and yaw rate too (from successive
           position estimates or from the marker's rvec).
        5. When detection is lost, propagate the filter forward (predict
           only) for a few frames before reverting to search.
        """
        # TODO: replace with your estimator.
        return {"position": None, "velocity": np.zeros(3)}

    # ------------------------------------------------------------------ STAGE 3: DECISION
    def decide(
        self,
        drone_state: Dict,
        boat_estimate: Dict,
        battery: float,
        time_s: float,
    ) -> str:
        """Pick a high-level phase based on the current state. Must
        return one of:
            PHASE_SEARCH, PHASE_APPROACH, PHASE_DESCEND, PHASE_LAND

        BASELINE: thresholds on horizontal distance and altitude above
        the marker, with a tiny hysteresis (returns SEARCH if the
        estimate is from the cold-start prior).

        WHY IT'S WEAK:
        - No timing awareness: descends regardless of whether the boat
          is at the worst moment of its roll oscillation.
        - No battery-aware decisions: doesn't abort if energy is low.
        - No 'go-around' behavior: once committed to LAND, can't bail
          and re-approach if drift takes it off the platform.

        IMPROVEMENT IDEAS:
        1. Wait until the boat's roll/pitch oscillation crosses zero
           before committing to LAND (estimate the phase from your
           filter or by tracking the marker's rvec over time).
        2. Add an ABORT phase: if descent is going wrong (e.g. drone is
           drifting off-platform, or the estimate becomes stale), climb
           and re-approach.
        3. Speed-vs-battery trade-off: if battery is low, accept a less
           precise landing rather than burning more time hovering.
        4. Time-budgeting: estimate remaining time-to-land and bail to
           a faster (less precise) path if you'd otherwise time out.
        """
        # TODO: replace with your decision policy.
        return PHASE_SEARCH

    # ------------------------------------------------------------------ STAGE 4: CONTROL
    def control(
        self, drone_state: Dict, boat_estimate: Dict, phase: str
    ) -> np.ndarray:
        """Produce the per-motor throttle action of shape
        `(self.spec.num_motors,)` in `[0, 1]`.

        Two valid pipelines:

        (A) Use the stock attitude controller (matches the baseline):

                roll_des = ...      # in [-1, +1] -> mapped to ±max_tilt
                pitch_des = ...
                yaw_rate_des = ...
                thrust_norm = ...   # in [-1, +1]; 0 = hover
                return self.attitude_ctrl(
                    drone_state, roll_des, pitch_des, yaw_rate_des, thrust_norm
                )

        (B) Emit motor throttles directly (more flexibility, possible
            sim-quality bonus):

                throttles = my_motor_mixing(...)
                return np.clip(throttles, 0.0, 1.0)

        BASELINE: three independent PIDs (x, y, z) on body-frame errors,
        feeding (A) above with the result. No yaw control, constant per-
        phase target altitudes, no feed-forward. In LAND phase the thrust
        PID is overridden with thrust_norm=-1 (bang-bang minimum) so the
        drone punches through the reference sim's ground-effect cushion
        — that's how the baseline lands EASY at all, but the descent is
        crude and forfeits the soft-landing bonus.

        WHY IT'S WEAK:
        - Independent axes ignore the cross-coupling caused by tilt.
        - No feed-forward on the boat's velocity, so we always chase.
        - No wind compensation.
        - LAND descent is bang-bang — fast but rough, no soft-landing
          bonus on touchdown.
        - No active yaw to align fuselage with boat heading — the
          wing-perpendicular landing condition will fail unless the boat
          happens to be aligned with world x at touchdown.

        IMPROVEMENT IDEAS:
        1. ACTIVE YAW CONTROL: command yaw_rate to drive your estimate
           of (boat_heading - drone_yaw) toward 0 or pi. On the VTOL the
           yaw response is slow — start aligning during APPROACH.
        2. FEED-FORWARD: command a target velocity equal to the boat's
           estimated velocity plus a correction term.
        3. Cascaded structure: outer loop on position -> velocity
           setpoint, inner loop on velocity -> attitude setpoint.
        4. Wind feed-forward: estimate wind from steady-state attitude
           offset and add a cancelation term to thrust direction.
        5. SOFT DESCENT PROFILE: replace baseline's bang-bang LAND with a
           target_z(t) that decreases monotonically and matches a
           smooth |v_z| < 1 m/s on touchdown — recovers the +5 soft
           landing bonus the baseline gives up.
        6. Light MPC: optimize the next 1-2 s over a low-DOF horizon.
        """
        # TODO: replace with your controller. Default = hover throttle.
        return self.attitude_ctrl(
            drone_state,
            roll_des=0.0, pitch_des=0.0, yaw_rate_des=0.0, thrust_norm=0.0,
        )


# evaluation/evaluate.py looks for `make_agent` first, then `Agent`.
Agent = TemplateAgent


def make_agent(drone_spec: Optional[DroneSpec] = None) -> TemplateAgent:
    return TemplateAgent(drone_spec)
