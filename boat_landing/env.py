"""Catch the Boat — main simulation environment.

A gym-style environment that orchestrates THREE actors:

    1. The participant's `DroneSimulator` (motor-level physics).
    2. The participant's `Agent` (perception + estimation + control).
    3. The organizer-owned env (boat dynamics, wind, camera, scoring,
       contact detection, scenarios).

The drone simulator owns the rigid-body physics of the airframe — the
env never integrates the drone's state. The env wraps a kinematic
PyBullet body around the simulator's pose for the sole purpose of
detecting contact with the boat hull, the platform, and the water plane.
PyBullet's dynamics solver is NOT used for the drone (no mass, no
forces); only its collision detector is.

Critical contracts:
    * `obs` NEVER contains the boat's ground-truth pose. It lives only
      in `info` for evaluation and debugging. Agents must infer boat
      position from the camera image.
    * `action` is the per-motor throttle vector of shape
      `(drone_sim.spec.num_motors,)` in `[0, 1]`. Agents that prefer
      the legacy 4-DoF (thrust/roll/pitch/yaw_rate) interface should
      wrap their high-level commands with
      `boat_landing.controllers.DefaultAttitudeController` BEFORE
      handing the action to `env.step`.
    * The drone simulator is sealed: it sees its own action and a
      world-frame external force (currently wind only). It does not see
      `obs`, the camera image, the boat pose, or the scenario YAML.

Physics timing:
    * Agent step (DT)          50 Hz   — `obs`/`step` cadence.
    * Physics substep          250 Hz  — `drone_sim.step` is called once
                                          per substep so collisions are
                                          checked at sub-DT resolution.
    * Wind force is sampled once per agent step and held constant across
      its substeps (matching the previous env's behavior).

Landing classification:
    A platform contact counts as LANDED iff:
        - vertical descent velocity at contact <= CRASH_VERT_VEL
        - drone is centered within the platform extents in xy
        - drone fuselage (body x-axis) is aligned with boat length axis
          within the scenario's `landing.yaw_alignment_tol_deg`
          tolerance ("wings perpendicular to boat length")
    Otherwise the contact counts as CRASHED.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import pybullet as p
import pybullet_data
import yaml

from boat_landing.battery import Battery
from boat_landing.boat import Boat
from boat_landing.camera import (
    CAMERA_HEIGHT,
    CAMERA_WIDTH,
    apply_fog,
    apply_motion_blur,
    apply_noise,
    apply_occlusion,
    get_intrinsics,
    get_projection_matrix,
    get_view_matrix,
)
from boat_landing.drone_interface import DroneSimulator, DroneSpec
from boat_landing.wind import Wind


REPO_ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = Path(__file__).resolve().parent / "assets"
ARUCO_DIR = ASSETS_DIR / "aruco"


def load_scenario(path) -> Dict:
    """Parse a scenario YAML file and return its contents as a dict."""
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Scenario file {path} did not parse to a dict.")
    return cfg


def _default_drone_sim() -> DroneSimulator:
    """Construct the reference drone simulator + VTOL spec.

    Used when callers (tests, legacy scripts) instantiate the env without
    explicitly passing a drone sim. Production CLIs (`evaluate.py`,
    `run_baseline.py`) always pass one.
    """
    # Lazy import: the reference sim lives under `agents/` and we do not
    # want a hard dependency cycle if the env is used standalone.
    from agents.drone_sim_baseline import BaselineDroneSimulator
    return BaselineDroneSimulator(str(REPO_ROOT / "drones" / "vtol.yaml"))


class BoatLandingEnv:
    """Drone-on-moving-boat landing environment.

    Action: np.ndarray of shape (drone_sim.spec.num_motors,) with values
    in [0, 1]. Each entry is the throttle for the corresponding motor.
    The mapping throttle -> RPM is the simulator's responsibility.

    Observation: dict with keys:
        camera   (uint8, HxWx3 RGB),
        state    (dict with position/velocity/attitude/angular_velocity),
        battery  (float in [0, 1]),
        time     (float seconds since reset).
    """

    # --- Timing ---
    DT = 0.02
    PHYSICS_DT = 1.0 / 250.0
    SUBSTEPS = 5

    # --- Boat ---
    # 6 m × 1.5 m hull with a small raised landing platform (1 m × 1 m)
    # on top. The drone targets the platform; landing on the hull (off
    # the platform) counts as a crash.
    BOAT_LENGTH = 6.0
    BOAT_WIDTH = 1.5
    BOAT_HEIGHT = 0.4

    PLATFORM_SIZE = 1.0
    PLATFORM_HEIGHT = 0.10

    # --- Marker ---
    MARKER_ID = 0
    MARKER_SIZE = 0.8
    MARKER_QUIET_ZONE = 0.1
    PLATE_SIZE = MARKER_SIZE + 2 * MARKER_QUIET_ZONE  # 1.0 m

    # --- Termination thresholds ---
    LANDING_VERT_VEL_TOL = 1.5
    CRASH_VERT_VEL = 3.0
    LANDING_HORIZ_TOL = 1.0
    DEFAULT_YAW_ALIGNMENT_TOL_DEG = 30.0   # used if scenario omits the field

    OUTCOMES = ("LANDED", "CRASHED", "TIMEOUT", "OUT_OF_BATTERY", "ABORTED")

    # ------------------------------------------------------------------ init
    def __init__(
        self,
        scenario_path,
        drone_sim: Optional[DroneSimulator] = None,
        gui: bool = False,
        record: bool = False,
    ):
        self.scenario_path = str(scenario_path)
        self.scenario = load_scenario(scenario_path)
        self.gui = bool(gui)
        self.record = bool(record)

        if drone_sim is None:
            drone_sim = _default_drone_sim()
        self.drone_sim: DroneSimulator = drone_sim
        self.drone_spec: DroneSpec = drone_sim.spec

        self.client = p.connect(p.GUI if gui else p.DIRECT)
        p.setAdditionalSearchPath(
            pybullet_data.getDataPath(), physicsClientId=self.client
        )
        p.setTimeStep(self.PHYSICS_DT, physicsClientId=self.client)
        # Gravity in PyBullet doesn't matter (everything we care about is
        # mass=0 / kinematic), but set it for any future dynamic bodies.
        p.setGravity(0, 0, -9.81, physicsClientId=self.client)
        if gui:
            p.configureDebugVisualizer(
                p.COV_ENABLE_GUI, 0, physicsClientId=self.client
            )
            p.resetDebugVisualizerCamera(
                cameraDistance=12.0,
                cameraYaw=45,
                cameraPitch=-30,
                cameraTargetPosition=[0, 0, 1],
                physicsClientId=self.client,
            )

        # Subsystems (created in reset)
        self.rng: Optional[np.random.Generator] = None
        self.boat: Optional[Boat] = None
        self.wind: Optional[Wind] = None
        self.battery: Optional[Battery] = None

        # Body IDs
        self.drone_id: Optional[int] = None        # kinematic, contact-only
        self.boat_id: Optional[int] = None         # hull
        self.platform_id: Optional[int] = None     # landing platform
        self.marker_plate_id: Optional[int] = None # ArUco visual
        self.plane_id: Optional[int] = None

        # Episode state
        self._t = 0.0
        self._step_count = 0
        self._terminated_flag = False
        self._termination_reason: Optional[str] = None
        self._max_descent_velocity = 0.0
        self._traj: list = []
        self._occlusion_t_remaining = 0.0
        self._wind_force_world = np.zeros(3, dtype=np.float64)

        # Camera frame cache (for fps-drop simulation)
        self._cached_camera: Optional[np.ndarray] = None
        self._last_camera_render_t: float = -np.inf
        # Deterministic occlusion window: [t_start, t_end] in sim seconds,
        # populated from scenario.camera.occlusion_window at reset. While
        # `_t` is inside the window the rendered frame is blacked out
        # regardless of the stochastic occlusion roll. Used by the recovery
        # scenario to test marker re-acquisition.
        self._scheduled_occlusion: Optional[Tuple[float, float]] = None

    # ------------------------------------------------------------------ reset
    def reset(self, seed: Optional[int] = None) -> Tuple[Dict, Dict]:
        if seed is None:
            seed = int(self.scenario.get("seed", 0))
        self.rng = np.random.default_rng(seed)

        p.resetSimulation(physicsClientId=self.client)
        p.setTimeStep(self.PHYSICS_DT, physicsClientId=self.client)
        p.setGravity(0, 0, -9.81, physicsClientId=self.client)
        p.setAdditionalSearchPath(
            pybullet_data.getDataPath(), physicsClientId=self.client
        )

        self.plane_id = self._create_water_plane()
        self.drone_id = self._create_drone_kinematic()
        self.boat_id, self.platform_id, self.marker_plate_id = self._create_boat()

        # Reset the participant simulator to its initial pose.
        start_pos = np.asarray(
            self.scenario["drone_start"]["position"], dtype=np.float64
        )
        start_att = np.asarray(
            self.scenario["drone_start"]["attitude"], dtype=np.float64
        )
        self.drone_sim.reset(start_pos, start_att)

        self.boat = Boat(self.scenario["boat"], self.rng)
        self.wind = Wind(self.scenario.get("wind", {}) or {}, self.rng)
        self.battery = Battery(self.scenario)

        self._t = 0.0
        self._step_count = 0
        self._terminated_flag = False
        self._termination_reason = None
        self._max_descent_velocity = 0.0
        self._traj = []
        self._occlusion_t_remaining = 0.0
        self._wind_force_world = np.zeros(3, dtype=np.float64)
        self._cached_camera = None
        self._last_camera_render_t = -np.inf

        cam_cfg = self.scenario.get("camera", {}) or {}
        window = cam_cfg.get("occlusion_window")
        if window is not None:
            t_start, t_end = float(window[0]), float(window[1])
            if t_end <= t_start:
                raise ValueError(
                    f"camera.occlusion_window must satisfy t_end > t_start; "
                    f"got [{t_start}, {t_end}]"
                )
            self._scheduled_occlusion = (t_start, t_end)
        else:
            self._scheduled_occlusion = None

        self._sync_drone_body()
        self._update_boat_pose()
        obs = self._get_observation()
        info = self._get_info()
        return obs, info

    # ------------------------------------------------------------------ step
    def step(self, action) -> Tuple[Dict, float, bool, bool, Dict]:
        if self._terminated_flag:
            raise RuntimeError(
                "step() called after termination — call reset() before stepping again."
            )
        action = np.asarray(action, dtype=np.float64).reshape(-1)
        n_motors = self.drone_spec.num_motors
        if action.shape != (n_motors,):
            raise ValueError(
                f"action must have shape ({n_motors},) "
                f"(per-motor throttle for this drone spec); got {action.shape}"
            )
        action = np.clip(action, 0.0, 1.0)

        # Wind: sampled once per agent step, held constant over the substeps.
        self._wind_force_world = self.wind.get_force(self.DT)

        early_terminate = False
        for _ in range(self.SUBSTEPS):
            # 1. Advance the participant's drone simulator by one physics
            #    substep. Wind enters here as the only external world force.
            #    ext_ground_z is the world z of the surface directly below
            #    the drone — used by sims that model ground effect.
            ground_z = self._ground_reference_z()
            self.drone_sim.step(
                action,
                self._wind_force_world,
                self.PHYSICS_DT,
                ext_ground_z=ground_z,
            )
            # 2. Step the boat at the substep rate so contact detection
            #    sees the latest pose for both bodies.
            self.boat.step(self.PHYSICS_DT)
            # 3. Mirror the simulator state into the kinematic PyBullet
            #    body, then update the boat's PyBullet pose.
            self._sync_drone_body()
            self._update_boat_pose()

            # 4. NaN guard. Diverging participant sims should fail loudly.
            state = self.drone_sim.get_state()
            if not (
                np.all(np.isfinite(state.position))
                and np.all(np.isfinite(state.velocity))
                and np.all(np.isfinite(state.quaternion))
            ):
                raise RuntimeError(
                    f"DroneSimulator produced non-finite state at step "
                    f"{self._step_count} substep: pos={state.position}, "
                    f"vel={state.velocity}, quat={state.quaternion}, "
                    f"action={action.tolist()}, wind={self._wind_force_world.tolist()}"
                )

            # 5. Check termination. Contact detection uses getClosestPoints
            #    instead of performCollisionDetection because PyBullet's
            #    broadphase skips static-vs-static (mass=0 vs mass=0) pairs;
            #    getClosestPoints explicitly queries each pair regardless.
            terminated, _, reason = self._check_termination()
            if terminated:
                self._termination_reason = reason
                early_terminate = True
                break

        # Battery drain uses the agent-step dt and the latest body angular vel.
        state = self.drone_sim.get_state()
        ang_vel_world = self._rotate_body_to_world(
            state.angular_velocity_body, state.quaternion
        )
        self.battery.step(self.DT, ang_vel_world)

        self._t += self.DT
        self._step_count += 1
        self._max_descent_velocity = max(
            self._max_descent_velocity, -float(state.velocity[2])
        )
        if self._occlusion_t_remaining > 0:
            self._occlusion_t_remaining = max(
                0.0, self._occlusion_t_remaining - self.DT
            )

        if not early_terminate:
            terminated, truncated, reason = self._check_termination()
        else:
            terminated, truncated, reason = True, False, self._termination_reason
        self._terminated_flag = bool(terminated or truncated)
        self._termination_reason = reason

        rpy = p.getEulerFromQuaternion(state.quaternion.tolist())
        self._traj.append(
            {
                "t": float(self._t),
                "drone_pos": state.position.copy(),
                "drone_rpy": np.asarray(rpy).copy(),
                "drone_vel": state.velocity.copy(),
                "boat_pos": self.boat.position.copy(),
                "boat_vel": self.boat.get_velocity().copy(),
            }
        )

        obs = self._get_observation()
        info = self._get_info()
        info["terminated"] = bool(terminated)
        info["truncated"] = bool(truncated)
        info["outcome"] = reason
        reward = 0.0  # rewards aren't used; agents are scored externally
        return obs, reward, bool(terminated), bool(truncated), info

    # ------------------------------------------------------------------ render/close
    def render(self):
        return None

    def close(self):
        try:
            p.disconnect(physicsClientId=self.client)
        except Exception:
            pass

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    # ------------------------------------------------------------------ scene creation
    def _create_water_plane(self) -> int:
        col = p.createCollisionShape(
            p.GEOM_BOX, halfExtents=[100, 100, 0.01], physicsClientId=self.client
        )
        vis = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[100, 100, 0.01],
            rgbaColor=[0.10, 0.25, 0.45, 1.0],
            physicsClientId=self.client,
        )
        return p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=col,
            baseVisualShapeIndex=vis,
            basePosition=[0, 0, -0.01],
            physicsClientId=self.client,
        )

    def _create_drone_kinematic(self) -> int:
        """Build a kinematic (mass=0) PyBullet body whose ONLY purpose is
        contact detection with the boat / platform / water plane.

        Geometry is taken from the drone spec (`spec.geometry.collision_box`).
        The drone's actual rigid-body state is owned by the participant's
        DroneSimulator; we mirror that state into this body each substep
        via `resetBasePositionAndOrientation`.
        """
        geom = self.drone_spec.geometry
        half = (geom.collision_box / 2.0).tolist()
        col = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=half,
            collisionFramePosition=[0.0, 0.0, geom.collision_offset_z],
            physicsClientId=self.client,
        )
        # Visual: a single box matching the collision shape, plus a thin
        # "wing" plate spanning body y so the drone is recognisable in the
        # GUI and the wing-perpendicular landing condition is easy to see
        # by eye. The wing plate is purely cosmetic (no collision).
        wing_half = [
            half[0] * 0.4,                    # short x (fuselage chord)
            max(half[1] * 1.8, half[1] + 0.3),  # extend wings beyond fuselage
            0.02,
        ]
        vis = p.createVisualShapeArray(
            shapeTypes=[p.GEOM_BOX, p.GEOM_BOX],
            halfExtents=[half, wing_half],
            visualFramePositions=[
                [0.0, 0.0, geom.collision_offset_z],
                [0.0, 0.0, geom.collision_offset_z + half[2] + 0.03],
            ],
            rgbaColors=[
                [0.20, 0.20, 0.25, 1.0],   # fuselage
                [0.45, 0.50, 0.60, 1.0],   # wings
            ],
            physicsClientId=self.client,
        )
        # baseMass=0 means PyBullet treats the body as kinematic: it does
        # not integrate dynamics, but it still participates in collision
        # detection. We move it via resetBasePositionAndOrientation each
        # substep to match the participant simulator's pose.
        drone_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=col,
            baseVisualShapeIndex=vis,
            basePosition=[0, 0, 0],
            baseOrientation=[0, 0, 0, 1],
            physicsClientId=self.client,
        )
        return drone_id

    def _create_boat(self) -> Tuple[int, int, int]:
        hull_half = [self.BOAT_LENGTH / 2, self.BOAT_WIDTH / 2, self.BOAT_HEIGHT / 2]
        hull_col = p.createCollisionShape(
            p.GEOM_BOX, halfExtents=hull_half, physicsClientId=self.client
        )
        hull_vis = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=hull_half,
            rgbaColor=[0.35, 0.25, 0.15, 1.0],
            physicsClientId=self.client,
        )
        boat_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=hull_col,
            baseVisualShapeIndex=hull_vis,
            basePosition=[0, 0, self.BOAT_HEIGHT / 2],
            physicsClientId=self.client,
        )

        plat_half = [
            self.PLATFORM_SIZE / 2,
            self.PLATFORM_SIZE / 2,
            self.PLATFORM_HEIGHT / 2,
        ]
        plat_col = p.createCollisionShape(
            p.GEOM_BOX, halfExtents=plat_half, physicsClientId=self.client
        )
        plat_vis = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=plat_half,
            rgbaColor=[0.85, 0.85, 0.85, 1.0],
            physicsClientId=self.client,
        )
        platform_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=plat_col,
            baseVisualShapeIndex=plat_vis,
            basePosition=[
                0,
                0,
                self.BOAT_HEIGHT + self.PLATFORM_HEIGHT / 2,
            ],
            physicsClientId=self.client,
        )

        h = self.PLATE_SIZE / 2.0
        plate_vis = p.createVisualShape(
            shapeType=p.GEOM_MESH,
            vertices=[[-h, -h, 0.0], [h, -h, 0.0], [h, h, 0.0], [-h, h, 0.0]],
            indices=[0, 1, 2, 0, 2, 3],
            uvs=[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
            normals=[[0.0, 0.0, 1.0]] * 4,
            rgbaColor=[1.0, 1.0, 1.0, 1.0],
            physicsClientId=self.client,
        )
        plate_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=plate_vis,
            basePosition=[
                0,
                0,
                self.BOAT_HEIGHT + self.PLATFORM_HEIGHT + 0.005,
            ],
            physicsClientId=self.client,
        )

        marker_path = ARUCO_DIR / f"marker_{self.MARKER_ID}.png"
        if not marker_path.is_file():
            self._auto_generate_marker(marker_path)
        try:
            tex_id = p.loadTexture(str(marker_path), physicsClientId=self.client)
            p.changeVisualShape(
                plate_id, -1, textureUniqueId=tex_id, physicsClientId=self.client
            )
        except Exception as exc:  # pragma: no cover — best-effort visual
            print(f"[BoatLandingEnv] Warning: failed to apply ArUco texture: {exc}")

        return boat_id, platform_id, plate_id

    def _auto_generate_marker(self, out_path: Path) -> None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        marker_px = 800
        pad = int(round(marker_px * self.MARKER_QUIET_ZONE / self.MARKER_SIZE))
        try:
            adict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
            inner = cv2.aruco.generateImageMarker(adict, self.MARKER_ID, marker_px)
        except AttributeError:
            adict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_100)
            inner = cv2.aruco.drawMarker(adict, self.MARKER_ID, marker_px)
        padded = np.full(
            (inner.shape[0] + 2 * pad, inner.shape[1] + 2 * pad), 255, dtype=np.uint8
        )
        padded[pad : pad + inner.shape[0], pad : pad + inner.shape[1]] = inner
        cv2.imwrite(str(out_path), padded)

    def _update_boat_pose(self) -> None:
        pos, rpy = self.boat.get_pose()
        quat = p.getQuaternionFromEuler(rpy)
        p.resetBasePositionAndOrientation(
            self.boat_id, pos.tolist(), quat, physicsClientId=self.client
        )
        rot = np.array(p.getMatrixFromQuaternion(quat)).reshape(3, 3)

        platform_offset_body = np.array(
            [0.0, 0.0, self.BOAT_HEIGHT / 2 + self.PLATFORM_HEIGHT / 2]
        )
        platform_pos = np.asarray(pos) + rot @ platform_offset_body
        p.resetBasePositionAndOrientation(
            self.platform_id,
            platform_pos.tolist(),
            quat,
            physicsClientId=self.client,
        )

        plate_offset_body = np.array(
            [0.0, 0.0, self.BOAT_HEIGHT / 2 + self.PLATFORM_HEIGHT + 0.005]
        )
        plate_pos = np.asarray(pos) + rot @ plate_offset_body
        p.resetBasePositionAndOrientation(
            self.marker_plate_id,
            plate_pos.tolist(),
            quat,
            physicsClientId=self.client,
        )

    def _sync_drone_body(self) -> None:
        """Mirror the participant simulator's pose into the kinematic
        PyBullet body. Called once per substep right before collision
        detection."""
        state = self.drone_sim.get_state()
        p.resetBasePositionAndOrientation(
            self.drone_id,
            state.position.tolist(),
            state.quaternion.tolist(),
            physicsClientId=self.client,
        )

    # ------------------------------------------------------------------ observation
    def _get_observation(self) -> Dict:
        state = self.drone_sim.get_state()
        cam_cfg = self.scenario.get("camera", {}) or {}

        # Frame-rate degradation: if the scenario specifies a target fps
        # below the env's effective rate, hold the previous frame between
        # rendering events. fps <= 0 or missing means render every step.
        target_fps = float(cam_cfg.get("fps", 0.0))
        render_new = True
        if target_fps > 0 and self._cached_camera is not None:
            min_dt = 1.0 / target_fps
            if (self._t - self._last_camera_render_t) < (min_dt - 1e-6):
                render_new = False

        if render_new:
            img = self._render_camera(state.position, state.quaternion)
            img = apply_noise(img, float(cam_cfg.get("noise_level", 0.0)), self.rng)
            mb_kernel = int(cam_cfg.get("motion_blur_kernel", 0))
            if mb_kernel >= 3:
                img = apply_motion_blur(img, mb_kernel)
            elif bool(cam_cfg.get("motion_blur", False)):
                # Legacy boolean — apply a default-strength horizontal blur.
                img = apply_motion_blur(img, 5)
            fog_density = float(cam_cfg.get("fog_density", 0.0))
            if fog_density > 0:
                # Altitude is the drone height above the boat platform.
                # Using altitude (not absolute z) so fog scales with the
                # camera-to-marker line-of-sight distance.
                altitude = max(state.position[2] - (self.BOAT_HEIGHT + self.PLATFORM_HEIGHT), 0.0)
                img = apply_fog(img, fog_density, altitude)
            # Stochastic occlusion: trigger a multi-frame blackout. Once
            # triggered, all subsequent renders are occluded until the
            # countdown reaches zero (env.step decrements it).
            if self._occlusion_t_remaining <= 0:
                prob = float(cam_cfg.get("occlusion_probability", 0.0))
                if prob > 0 and self.rng.random() < prob * self.DT:
                    self._occlusion_t_remaining = float(
                        cam_cfg.get("occlusion_duration", 1.0)
                    )
            # Deterministic scheduled blackout takes effect alongside the
            # stochastic one — either path forces the frame to be occluded.
            scheduled = (
                self._scheduled_occlusion is not None
                and self._scheduled_occlusion[0] <= self._t < self._scheduled_occlusion[1]
            )
            img = apply_occlusion(img, scheduled or self._occlusion_t_remaining > 0)
            self._cached_camera = img
            self._last_camera_render_t = self._t
        else:
            img = self._cached_camera

        rpy = p.getEulerFromQuaternion(state.quaternion.tolist())
        ang_vel_world = self._rotate_body_to_world(
            state.angular_velocity_body, state.quaternion
        )
        return {
            "camera": img,
            "state": {
                "position": state.position.copy(),
                "velocity": state.velocity.copy(),
                "attitude": np.asarray(rpy, dtype=np.float64),
                "angular_velocity": ang_vel_world,
            },
            "battery": float(self.battery.charge),
            "time": float(self._t),
        }

    def _render_camera(self, drone_pos: np.ndarray, drone_quat: np.ndarray) -> np.ndarray:
        view = get_view_matrix(drone_pos, drone_quat)
        proj = get_projection_matrix()
        renderer = (
            p.ER_BULLET_HARDWARE_OPENGL if self.gui else p.ER_TINY_RENDERER
        )
        try:
            _, _, rgba, _, _ = p.getCameraImage(
                CAMERA_WIDTH,
                CAMERA_HEIGHT,
                viewMatrix=view,
                projectionMatrix=proj,
                renderer=renderer,
                flags=p.ER_NO_SEGMENTATION_MASK,
                physicsClientId=self.client,
            )
        except Exception:
            rgba = np.full((CAMERA_HEIGHT, CAMERA_WIDTH, 4), 64, dtype=np.uint8)
        return np.asarray(rgba, dtype=np.uint8).reshape(
            CAMERA_HEIGHT, CAMERA_WIDTH, 4
        )[:, :, :3].copy()

    def _get_info(self) -> Dict:
        return {
            "boat_position": self.boat.position.copy(),
            "boat_velocity": self.boat.get_velocity().copy(),
            "boat_heading": float(self.boat.heading),
            "boat_roll": float(self.boat.roll),
            "boat_pitch": float(self.boat.pitch),
            "max_descent_velocity": float(self._max_descent_velocity),
            "wind_force": self._wind_force_world.copy(),
            "scenario_id": self.scenario.get("scenario_id", "unknown"),
            "step": int(self._step_count),
        }

    # ------------------------------------------------------------------ termination
    def _check_termination(self) -> Tuple[bool, bool, Optional[str]]:
        state = self.drone_sim.get_state()
        pos = state.position
        vel = state.velocity

        if self.battery.depleted:
            return True, False, "OUT_OF_BATTERY"
        if self._t >= self.scenario["duration_max"]:
            return False, True, "TIMEOUT"
        if pos[2] < 0:
            return True, False, "CRASHED"

        platform_contact = self._drone_in_contact_with(self.platform_id)
        hull_contact = self._drone_in_contact_with(self.boat_id)
        plane_contact = self._drone_in_contact_with(self.plane_id)

        if not (platform_contact or hull_contact or plane_contact):
            return False, False, None

        if plane_contact and not platform_contact:
            return True, False, "CRASHED"

        if hull_contact and not platform_contact:
            return True, False, "CRASHED"

        boat_pos = self.boat.position
        descent_vel = -float(vel[2])
        on_platform = (
            abs(pos[0] - boat_pos[0]) < self.PLATFORM_SIZE / 2
            and abs(pos[1] - boat_pos[1]) < self.PLATFORM_SIZE / 2
        )
        if descent_vel > self.CRASH_VERT_VEL:
            return True, False, "CRASHED"
        if not on_platform:
            return True, False, "CRASHED"

        # Wing-perpendicular alignment check: the drone's body x-axis
        # (along fuselage) must be aligned with the boat's heading axis
        # within the scenario's tolerance. Equivalently, the wings
        # (body y-axis) must be perpendicular to boat length.
        landing_cfg = self.scenario.get("landing", {}) or {}
        tol_deg = float(
            landing_cfg.get("yaw_alignment_tol_deg", self.DEFAULT_YAW_ALIGNMENT_TOL_DEG)
        )
        tol_rad = np.deg2rad(tol_deg)
        misalignment = self._yaw_misalignment_rad(state.quaternion, self.boat.heading)
        if misalignment > tol_rad:
            return True, False, "CRASHED"
        return True, False, "LANDED"

    def _ground_reference_z(self) -> float:
        """World-frame z [m] of the surface directly below the drone.

        Used to compute ground-effect altitude inside the participant's
        simulator (passed to `drone_sim.step` as `ext_ground_z`).
        Convention:
            * Drone laterally over the platform → platform top z
              (boat hull center + half hull height + platform height).
            * Drone laterally over the hull but not the platform →
              hull top z.
            * Drone elsewhere → water plane (z = 0).
        Boat roll/pitch are ignored — we use the level-pose reference
        so that ground effect is a smooth function of drone position
        and not modulated by oscillation. The platform is the dominant
        ground surface during the landing approach, so this matches the
        physical regime where ground effect actually matters.
        """
        state = self.drone_sim.get_state()
        boat_pos = self.boat.position
        lateral = state.position[:2] - boat_pos[:2]
        platform_top = float(boat_pos[2] + self.BOAT_HEIGHT / 2 + self.PLATFORM_HEIGHT)
        hull_top = float(boat_pos[2] + self.BOAT_HEIGHT / 2)
        if (
            abs(lateral[0]) < self.PLATFORM_SIZE / 2
            and abs(lateral[1]) < self.PLATFORM_SIZE / 2
        ):
            return platform_top
        if (
            abs(lateral[0]) < self.BOAT_LENGTH / 2
            and abs(lateral[1]) < self.BOAT_WIDTH / 2
        ):
            return hull_top
        return 0.0

    def _drone_in_contact_with(self, other_id: int, tol_m: float = 1e-3) -> bool:
        """True iff the drone collision box is interpenetrating or within
        `tol_m` of the named body. Uses getClosestPoints (works for
        static-vs-static pairs, unlike getContactPoints which relies on
        the broadphase populated by stepSimulation)."""
        pts = p.getClosestPoints(
            self.drone_id, other_id, distance=tol_m,
            physicsClientId=self.client,
        )
        return len(pts) > 0

    @staticmethod
    def _yaw_misalignment_rad(drone_quat: np.ndarray, boat_heading: float) -> float:
        """Smallest angle between the drone fuselage (body x) and the boat
        length axis, modulo pi (so heading + pi is also "aligned"). Result
        is in [0, pi/2]."""
        rpy = p.getEulerFromQuaternion(drone_quat.tolist())
        drone_yaw = float(rpy[2])
        delta = (drone_yaw - float(boat_heading)) % np.pi
        return float(min(delta, np.pi - delta))

    # ------------------------------------------------------------------ accessors
    def _get_position(self) -> np.ndarray:
        return self.drone_sim.get_state().position.copy()

    def _get_quaternion(self) -> np.ndarray:
        return self.drone_sim.get_state().quaternion.copy()

    def _get_velocity(self) -> np.ndarray:
        return self.drone_sim.get_state().velocity.copy()

    def _get_angular_velocity(self) -> np.ndarray:
        state = self.drone_sim.get_state()
        return self._rotate_body_to_world(state.angular_velocity_body, state.quaternion)

    @staticmethod
    def _rotate_body_to_world(omega_body: np.ndarray, quat: np.ndarray) -> np.ndarray:
        rot = np.array(p.getMatrixFromQuaternion(quat.tolist())).reshape(3, 3)
        return rot @ omega_body

    # ------------------------------------------------------------------ public introspection
    def get_trajectory(self) -> list:
        """Per-step (t, drone_pos, drone_vel, boat_pos, boat_vel) log."""
        return self._traj

    def get_intrinsics(self) -> np.ndarray:
        return get_intrinsics()

    @property
    def t(self) -> float:
        return self._t

    @property
    def terminated(self) -> bool:
        return self._terminated_flag

    @property
    def termination_reason(self) -> Optional[str]:
        return self._termination_reason
