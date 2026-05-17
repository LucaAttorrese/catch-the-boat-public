"""Drone simulator interface — the contract between the env and the
participant-supplied DroneSimulator.

Architecture overview
---------------------
Catch the Boat splits the simulation into THREE artifacts:

    1. Sealed env (THIS package): boat dynamics, wind, camera, scoring,
       contact detection, scenarios. Owned by the organizers. Read-only.
    2. Drone simulator (participant): physics that turns motor commands
       into rigid-body state evolution, parameterised by a drone spec
       YAML in `drones/`. Tier 0 is mandatory; tiers 1-2 add fidelity
       and earn bonus points.
    3. Agent (participant): perception + estimation + control. Sees the
       env's `obs` and emits actions. Whether actions are per-motor
       throttles (tier 0 raw) or higher-level setpoints (using the
       DefaultAttitudeController in `boat_landing.controllers`) is the
       agent's choice.

Data flow per step
------------------

    env.boat.step(dt)
    env.wind.compute(dt) -> F_wind_world
    obs = env.render(camera + drone state)         # uses sim.get_state()

    action = agent.act(obs)                        # shape: (num_motors,) in [0, 1]
                                                   #   OR whatever the agent's
                                                   #   chosen controller emits

    drone_sim.step(action, F_wind_world, dt)       # the sim integrates physics
    state = drone_sim.get_state()                  # env reads new state
    env.contact_check(state, geometry)             # env decides termination

Sealing rule
------------
The drone simulator NEVER sees the camera image, the boat pose, the
scenario YAML, or any other env-internal state. It receives only:
    * its own action (motor commands), and
    * external world-frame forces (wind + future disturbances).
Gravity is the simulator's own responsibility — it knows mass.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Protocol, runtime_checkable

import numpy as np
import yaml


# ---------------------------------------------------------------------------
# Spec dataclasses — typed mirror of the drones/*.yaml schema.
# ---------------------------------------------------------------------------


@dataclass
class MotorPlacement:
    """One motor's geometry."""

    id: int
    position: np.ndarray       # (3,) body frame [m]
    thrust_axis: np.ndarray    # (3,) unit vector, body frame
    spin: int                  # +1 = CCW from above, -1 = CW from above


@dataclass
class MotorParams:
    """Per-motor parameters, assumed identical across all motors."""

    omega_max: float           # [rad/s] full-throttle steady-state speed
    time_constant: float       # [s] tier-0 first-order RPM lag
    rotor_inertia: float       # [kg*m^2] used by tier-1 BLDC models


@dataclass
class PropellerParams:
    """Per-propeller parameters, assumed identical across all motors."""

    thrust_coefficient: float  # [N / (rad/s)^2] T = k_T * omega^2
    drag_coefficient: float    # [N*m / (rad/s)^2] Q = k_Q * omega^2
    diameter: float            # [m]


@dataclass
class BatteryParams:
    """Electrical battery model — used by tier-1 sims for voltage sag."""

    capacity_Wh: float
    voltage_nominal: float     # [V]
    internal_resistance: float # [ohm]


@dataclass
class AeroParams:
    """Optional aerodynamic drag (tier 1). Set arrays to 0 for vacuum."""

    drag_linear: np.ndarray    # (3,) [N / (m/s)] per body axis
    drag_quadratic: np.ndarray # (3,) [N / (m/s)^2] per body axis


@dataclass
class GeometryParams:
    """Collision box used by the env for contact detection. Full edge
    lengths (NOT half-extents). The box is centered at the drone COM
    and offset along body z by `collision_offset_z`."""

    collision_box: np.ndarray  # (3,) full edge lengths [m]
    collision_offset_z: float  # [m] offset of box center from COM


@dataclass
class DroneSpec:
    """Top-level drone specification — typed view of a drones/*.yaml file."""

    name: str
    type: str
    mass: float                # [kg]
    inertia: np.ndarray        # (3,3) body-frame inertia tensor
    geometry: GeometryParams
    motors: List[MotorPlacement]
    motor: MotorParams
    propeller: PropellerParams
    battery: BatteryParams
    aero: AeroParams
    description: str = ""

    @property
    def num_motors(self) -> int:
        return len(self.motors)

    @property
    def hover_thrust(self) -> float:
        """Total upward thrust required to hover at sea level."""
        return float(self.mass * 9.81)

    @property
    def hover_thrust_per_motor(self) -> float:
        """Per-motor thrust at hover assuming equal split. For non-symmetric
        motor placements, the actual split needs to be solved via the mixer."""
        return self.hover_thrust / self.num_motors


# ---------------------------------------------------------------------------
# State dataclass — what the simulator returns at each step.
# ---------------------------------------------------------------------------


@dataclass
class DroneState:
    """Snapshot of the drone's rigid-body state.

    Conventions:
        position, velocity:        world frame
        quaternion:                (x, y, z, w), PyBullet convention
        angular_velocity_body:     body frame (omega_body)
        motor_omegas:              optional per-motor RPM (rad/s) — set to
                                   None if your sim does not track motor
                                   dynamics explicitly (you'd score 0 on
                                   the motor-fidelity sim-quality test).
    """

    position: np.ndarray           # (3,)
    velocity: np.ndarray           # (3,)
    quaternion: np.ndarray         # (4,) (x, y, z, w)
    angular_velocity_body: np.ndarray  # (3,)
    motor_omegas: Optional[np.ndarray] = None  # (N,) [rad/s]


# ---------------------------------------------------------------------------
# Protocol — what participants must implement.
# ---------------------------------------------------------------------------


@runtime_checkable
class DroneSimulator(Protocol):
    """The contract between the env and the participant's drone simulator.

    Required attribute:
        spec: DroneSpec   — exposes mass, geometry, num_motors, etc.

    Required methods:
        reset(position, attitude)            initial conditions
        step(motor_cmds, ext_force_world, dt)   advance physics
        get_state() -> DroneState

    Action shape contract:
        motor_cmds is a numpy array of shape (spec.num_motors,) with
        values in [0, 1]. 0 = motor off, 1 = full throttle. The mapping
        from throttle to commanded RPM is part of YOUR motor model — the
        env makes no assumption (the natural choice is
        omega_cmd = throttle * spec.motor.omega_max).

    Force contract:
        ext_force_world is an ndarray of shape (3,) in world coordinates.
        It contains EXTERNAL forces (currently: wind only). GRAVITY is
        the simulator's own responsibility — apply -mass*g along world -z
        inside step(). The env never adds gravity to ext_force_world.

    Sealing rule:
        Your sim must not import `boat_landing.boat`, `boat_landing.wind`,
        `boat_landing.camera`, or read `scenarios/*.yaml`. It only sees
        what step() receives.
    """

    spec: DroneSpec

    def reset(self, position: np.ndarray, attitude: np.ndarray) -> None:
        """Reset to the given initial state.

        position: (3,) world frame.
        attitude: either (3,) RPY in PyBullet convention OR (4,) quaternion
                  (x, y, z, w). Implementations should accept both.
        """
        ...

    def step(
        self,
        motor_cmds: np.ndarray,
        ext_force_world: np.ndarray,
        dt: float,
        ext_torque_world: Optional[np.ndarray] = None,
        ext_ground_z: Optional[float] = None,
    ) -> None:
        """Advance the rigid-body state by dt seconds.

        motor_cmds:        (num_motors,) in [0, 1]
        ext_force_world:   (3,) world-frame external force [N] (wind etc.)
        dt:                [s]
        ext_torque_world:  (3,) world-frame external torque [N*m], default 0
        ext_ground_z:      world-frame z [m] of the surface directly below
                           the drone (boat platform top when the drone is
                           over the platform; boat hull top when over the
                           hull; water plane (z=0) otherwise). Used by
                           sims that model GROUND EFFECT — the altitude
                           above the relevant surface is
                           `state.position[2] - ext_ground_z`. Tier-0
                           sims may ignore this argument.
                           Default `None` is interpreted as "no ground
                           reference, treat as free flight".
        """
        ...

    def get_state(self) -> DroneState:
        """Return the current rigid-body state."""
        ...


# ---------------------------------------------------------------------------
# YAML loader.
# ---------------------------------------------------------------------------


def _as_array(x, shape=None) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if shape is not None and arr.shape != shape:
        raise ValueError(f"Expected shape {shape}, got {arr.shape}")
    return arr


def load_drone_spec(path) -> DroneSpec:
    """Parse a drones/*.yaml file into a typed DroneSpec.

    Raises ValueError on missing/malformed fields. The schema is defined
    by the comments in drones/vtol.yaml — keep them in sync.
    """
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"Drone spec {path} did not parse to a dict.")

    # Inertia: diagonal -> (3, 3) tensor.
    iner = raw.get("inertia", {})
    inertia = np.diag([
        float(iner["Ixx"]),
        float(iner["Iyy"]),
        float(iner["Izz"]),
    ])

    geom_raw = raw.get("geometry", {})
    cb = geom_raw.get("collision_box", {})
    geometry = GeometryParams(
        collision_box=np.array(
            [float(cb["length"]), float(cb["width"]), float(cb["height"])],
            dtype=np.float64,
        ),
        collision_offset_z=float(geom_raw.get("collision_offset_z", 0.0)),
    )

    motors_raw = raw.get("motors", [])
    if not motors_raw:
        raise ValueError(f"Drone spec {path} has no motors.")
    motors = []
    for m in motors_raw:
        axis = _as_array(m["thrust_axis"], shape=(3,))
        norm = float(np.linalg.norm(axis))
        if norm < 1e-9:
            raise ValueError(f"Motor {m.get('id')} has zero thrust axis.")
        motors.append(
            MotorPlacement(
                id=int(m["id"]),
                position=_as_array(m["position"], shape=(3,)),
                thrust_axis=axis / norm,
                spin=int(m["spin"]),
            )
        )

    mp = raw.get("motor", {})
    motor = MotorParams(
        omega_max=float(mp["omega_max"]),
        time_constant=float(mp.get("time_constant", 0.05)),
        rotor_inertia=float(mp.get("rotor_inertia", 0.0)),
    )

    pp = raw.get("propeller", {})
    propeller = PropellerParams(
        thrust_coefficient=float(pp["thrust_coefficient"]),
        drag_coefficient=float(pp["drag_coefficient"]),
        diameter=float(pp.get("diameter", 0.0)),
    )

    bp = raw.get("battery", {})
    battery = BatteryParams(
        capacity_Wh=float(bp.get("capacity_Wh", 0.0)),
        voltage_nominal=float(bp.get("voltage_nominal", 0.0)),
        internal_resistance=float(bp.get("internal_resistance", 0.0)),
    )

    ap = raw.get("aero", {}) or {}
    aero = AeroParams(
        drag_linear=_as_array(
            ap.get("drag_linear", [0.0, 0.0, 0.0]), shape=(3,)
        ),
        drag_quadratic=_as_array(
            ap.get("drag_quadratic", [0.0, 0.0, 0.0]), shape=(3,)
        ),
    )

    return DroneSpec(
        name=str(raw.get("name", "unnamed")),
        type=str(raw.get("type", "unknown")),
        description=str(raw.get("description", "")),
        mass=float(raw["mass"]),
        inertia=inertia,
        geometry=geometry,
        motors=motors,
        motor=motor,
        propeller=propeller,
        battery=battery,
        aero=aero,
    )
