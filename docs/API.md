# Environment API Reference

Three artifacts collaborate at runtime:

| Owner | File | Role |
| --- | --- | --- |
| Organizers | `boat_landing/env.py` (`BoatLandingEnv`) | Sealed env: boat, wind, camera, contact, scoring. |
| You (mandatory) | `drone_sim.py` (implements `DroneSimulator` Protocol) | Rigid-body physics: motor cmds → state. |
| You (mandatory) | `agent.py` (implements `Agent`) | Perception + estimation + control. |

This document covers the API surface for all three.

---

## `BoatLandingEnv`

```python
from boat_landing.env import BoatLandingEnv
from agents.drone_sim_baseline import BaselineDroneSimulator

sim = BaselineDroneSimulator("drones/quadcopter.yaml")
env = BoatLandingEnv("scenarios/easy.yaml", drone_sim=sim, gui=False)
```

| Argument        | Type                | Default | Notes                                          |
| --------------- | ------------------- | ------- | ---------------------------------------------- |
| `scenario_path` | `str / Path`        | —       | YAML file under `scenarios/` (or absolute).    |
| `drone_sim`     | `DroneSimulator`    | `None`  | If `None`, a `BaselineDroneSimulator(quadcopter.yaml)` is constructed automatically (legacy convenience). |
| `gui`           | `bool`              | `False` | Open a PyBullet GUI window.                    |
| `record`        | `bool`              | `False` | Reserved; visualizer handles demo capture.     |

After construction the env exposes:

```python
env.drone_sim    # the DroneSimulator instance you passed in
env.drone_spec   # convenience alias for env.drone_sim.spec
env.t            # simulated seconds since reset
env.terminated   # bool, True after termination
```

### `reset(seed: int | None = None) -> (obs, info)`

Resets the simulation. If `seed` is `None`, the scenario's `seed` field
is used. The same `(scenario, seed, drone_sim_implementation)` triple is
fully reproducible — provided your sim passes the determinism gate
(T0.3) in [`SIM_SCORING.md`](SIM_SCORING.md).

### `step(action: np.ndarray) -> (obs, reward, terminated, truncated, info)`

Advances the simulation by one agent step (1/50 s). The signature
follows the `gymnasium` 5-tuple convention. `reward` is always `0.0` —
agents are scored externally by `evaluation/scorer.py`.

`action` must be `np.ndarray` of shape **`(env.drone_spec.num_motors,)`**
(4 for both shipped specs), dtype convertible to `float64`, with values
in `[0, 1]`. Out-of-range values are clipped silently.

Each entry is the **throttle command for one motor**. The mapping from
throttle to commanded RPM is your simulator's responsibility (the
natural choice is `omega_cmd = throttle * spec.motor.omega_max`).

If you'd rather emit high-level setpoints than per-motor throttles, use
the stock attitude controller:

```python
from boat_landing.controllers import DefaultAttitudeController

ctrl = DefaultAttitudeController(env.drone_spec)
action = ctrl(obs["state"],
              roll_des=0.0, pitch_des=0.0,
              yaw_rate_des=0.0, thrust_norm=0.0)  # hover throttle
```

The controller derives PD gains from `spec.inertia` so it works for
both the quadcopter and VTOL specs without retuning.

Internally the env runs **5 physics substeps per agent step** at
`PHYSICS_DT = 1/250 s`. Your `drone_sim.step` is called once per
substep with `dt = PHYSICS_DT`. Wind force is sampled once per agent
step and held constant across the substeps.

### `close() -> None`

Disconnects the PyBullet client. Call this when done; the env also
closes itself on garbage-collection but explicit cleanup is recommended.

### Observation contract

```python
obs = {
    "camera":  np.ndarray,  # (480, 640, 3) uint8 RGB — possibly degraded
    "state": {
        "position":         np.ndarray,  # (3,) world frame, m
        "velocity":         np.ndarray,  # (3,) world frame, m/s
        "attitude":         np.ndarray,  # (3,) roll, pitch, yaw, rad
        "angular_velocity": np.ndarray,  # (3,) world frame, rad/s
    },
    "battery": float,
    "time":    float,
}
```

Notes:

- **The camera frame is degraded** by the scenario. Scenarios with
  `camera.fps < 50` return the **previous** frame in between renders;
  scenarios with `camera.fog_density > 0` blend with marine haze
  proportional to drone altitude (Beer-Lambert); scenarios with
  `camera.motion_blur_kernel >= 3` apply a horizontal box blur;
  scenarios with `camera.occlusion_probability > 0` may briefly black
  out the frame entirely. See [`CHALLENGE.md`](CHALLENGE.md#scenarios)
  for per-tier values.
- The camera is mounted **under** the drone (not at its centre of
  mass) at body offset `(0, 0, CAMERA_BODY_OFFSET_Z)` with
  `CAMERA_BODY_OFFSET_Z = -0.115 m` along body `-z`. It looks straight
  down along body `-z`. Image "up" is body `+x` (drone forward).
  `solvePnP`'s `tvec` is relative to the *camera*, not the drone COM —
  add the offset when transforming to world.
- `attitude` follows PyBullet's `getEulerFromQuaternion` convention:
  `R_world_body = Rz(yaw) @ Ry(pitch) @ Rx(roll)`.
- `angular_velocity` is in **world frame** (PyBullet default). Convert
  to body frame via `R_world_body.T @ omega_world` if you need it.
- `battery` decreases with time; faster when angular velocity is high.

### Info contract (evaluation only)

```python
info = {
    "boat_position":         np.ndarray,  # (3,) ground truth, m
    "boat_velocity":         np.ndarray,  # (3,) ground truth, m/s
    "boat_heading":          float,       # rad
    "boat_roll":             float,       # rad
    "boat_pitch":            float,       # rad
    "max_descent_velocity":  float,       # cumulative max -vz over episode
    "wind_force":            np.ndarray,  # (3,) world frame, N (this step)
    "scenario_id":           str,
    "step":                  int,
    # After every step():
    "terminated":            bool,
    "truncated":             bool,
    "outcome":               str | None,  # one of OUTCOMES
}
```

`info` is intended for the scorer. **Reading `info` from your agent
or your sim is disqualifying.**

### Termination outcomes

| Outcome           | Trigger                                                   |
| ----------------- | --------------------------------------------------------- |
| `LANDED`          | All four hold: drone contacts the platform, drone center within ±0.5 m of platform xy, descent velocity < `CRASH_VERT_VEL` (3 m/s), and **fuselage axis aligned with boat heading mod π** within `landing.yaw_alignment_tol_deg` (default 25°). |
| `CRASHED`         | Drone hits the hull (off-platform), or the water (`z < 0`), or the platform with descent ≥ 3 m/s, or the platform with yaw misalignment > tolerance. |
| `TIMEOUT`         | Episode wall time ≥ `duration_max`.                       |
| `OUT_OF_BATTERY`  | `battery <= 0`.                                           |

The wing-perpendicular condition (third bullet of `LANDED`) is the
new one this year. Yaw planning is part of the agent's job.

### Camera intrinsics

```python
from boat_landing.camera import get_intrinsics, CAMERA_BODY_OFFSET_Z
K = get_intrinsics()  # (3, 3)
```

The camera is pinhole, no distortion. Resolution 640×480, vertical FOV
90°. Mount offset along body z is `CAMERA_BODY_OFFSET_Z = -0.115 m`.

---

## `DroneSimulator` Protocol

The full source of truth is
[`boat_landing/drone_interface.py`](../boat_landing/drone_interface.py).
Summary:

```python
from typing import Protocol, Optional
from boat_landing.drone_interface import DroneSpec, DroneState

class DroneSimulator(Protocol):
    spec: DroneSpec

    def reset(self, position: np.ndarray, attitude: np.ndarray) -> None: ...

    def step(self,
             motor_cmds:        np.ndarray,            # (num_motors,) in [0, 1]
             ext_force_world:   np.ndarray,            # (3,) world-frame N
             dt:                float,                 # seconds, ~ 1/250
             ext_torque_world:  Optional[np.ndarray] = None,   # (3,) world-frame N·m
             ext_ground_z:      Optional[float] = None,        # world z of surface below
             ) -> None: ...

    def get_state(self) -> DroneState: ...
```

### What the env passes you

* `motor_cmds`: directly from the agent's `act()`, clipped to `[0, 1]`.
* `ext_force_world`: wind force for this agent step (sampled once,
  applied to every substep). Currently world-frame only; future
  scenarios may add disturbances here.
* `dt`: always `1/250 s` (`PHYSICS_DT`). Don't assume a fixed value
  forever — read it at runtime.
* `ext_torque_world`: currently always zero. Reserved.
* `ext_ground_z`: world-frame z of the **surface directly below the
  drone**. The env computes:
  - drone laterally over the **platform** → `boat.z + BOAT_HEIGHT/2 + PLATFORM_HEIGHT`
  - drone laterally over the **hull** but not the platform → `boat.z + BOAT_HEIGHT/2`
  - elsewhere → `0.0` (water plane)
  Used by sims modelling **ground effect** (T1.F). Sims that ignore it
  receive any value safely (default `None`).

### Gravity is your responsibility

The env never adds gravity to `ext_force_world`. Your sim knows the
mass (from `spec.mass`), so apply `-m·g` along world `-z` inside
`step()`. The convention is `g = 9.81 m/s²`.

### State you must return

```python
@dataclass
class DroneState:
    position:                np.ndarray  # (3,) world frame [m]
    velocity:                np.ndarray  # (3,) world frame [m/s]
    quaternion:              np.ndarray  # (4,) (x, y, z, w) — PyBullet convention
    angular_velocity_body:   np.ndarray  # (3,) body frame [rad/s]
    motor_omegas:            Optional[np.ndarray]  # (num_motors,) [rad/s], or None
```

* `quaternion` must be **unit norm** (verified by T0.1 gate).
  PyBullet uses scalar-last `(x, y, z, w)`.
* `angular_velocity_body` is **body frame**. The env rotates it to
  world frame internally for the obs and battery model.
* `motor_omegas` is optional but **required** if you want to claim
  T1.A (motor lag) or T1.B (battery sag) in `submission.yaml`. Both
  tests read it.

### `DroneSpec` dataclass

The typed view of `drones/*.yaml`:

```python
spec.name                                   # str
spec.type                                   # "quadrotor" | "heavy_multirotor"
spec.mass                                   # float [kg]
spec.inertia                                # (3, 3) [kg·m²], diagonal in our shipped specs
spec.num_motors                             # int
spec.hover_thrust                           # mass * 9.81 [N]
spec.hover_thrust_per_motor                 # spec.hover_thrust / num_motors
spec.motors[i].position                     # (3,) body-frame [m]
spec.motors[i].thrust_axis                  # (3,) unit vector body-frame
spec.motors[i].spin                         # +1 (CCW) | -1 (CW) viewed from above
spec.motor.omega_max                        # [rad/s]
spec.motor.time_constant                    # [s] first-order RPM lag
spec.motor.rotor_inertia                    # [kg·m²] for tier-1 BLDC
spec.propeller.thrust_coefficient           # k_T  [N / (rad/s)²]
spec.propeller.drag_coefficient             # k_Q  [N·m / (rad/s)²]
spec.propeller.diameter                     # [m]
spec.battery.capacity_Wh                    # for tier-1 sag
spec.battery.voltage_nominal                # [V]
spec.battery.internal_resistance            # [ohm]
spec.aero.drag_linear                       # (3,) [N / (m/s)] per body axis
spec.aero.drag_quadratic                    # (3,) [N / (m/s)²]
spec.geometry.collision_box                 # (3,) full edge lengths [m]
spec.geometry.collision_offset_z            # [m] box center vs COM
```

Load with:

```python
from boat_landing.drone_interface import load_drone_spec
spec = load_drone_spec("drones/quadcopter.yaml")
```

### Simulator dynamic-loading contract

The CLI and `sim_scorer.py` load your drone-sim module and look for,
in order:

1. `make_drone_sim(spec_path: str)` — preferred; returns a fresh
   sim instance bound to `spec_path`.
2. `DroneSim` — a class taking the spec path as its sole `__init__`
   argument.

---

## Agent contract

The CLI and scorer load your agent module and look for, in order:

1. `make_agent(drone_spec)` — preferred; `drone_spec` is the typed
   spec the env will use, passed by the loader. Functions that take
   no arguments still work (legacy compatibility).
2. `Agent` — a class; the loader calls `Agent(drone_spec)` if the
   constructor accepts it, else `Agent()`.

Your agent must implement:

```python
def act(self, obs: dict) -> np.ndarray: ...
```

The return value must be shape `(drone_spec.num_motors,)` with values
in `[0, 1]`. The env clips out-of-range values silently.

Optionally:

```python
def get_last_estimate(self) -> dict | None: ...
```

Returns the most recent boat-pose estimate (or `None` to opt out of
the estimation bonus). Required keys: `"position"` (`(3,)` ndarray).
Recommended: `"velocity"` (`(3,)` ndarray).

---

## CLI

### Score the agent on a scenario

```bash
python evaluation/evaluate.py \
    --agent      teams/myteam/agent.py        \    # default: agents/agent_baseline.py
    --drone-sim  teams/myteam/drone_sim.py    \    # default: agents/drone_sim_baseline.py
    --drone      quadcopter                   \    # default: quadcopter; "vtol" or path to .yaml
    --scenario   easy                         \    # required
    --headless --seed 42
```

Prints the agent score JSON to stdout (with the score breakdown). See
`--help` for `--gui`, `--save-traj`, `--output`.

### Score the simulator on the validation suite

```bash
python evaluation/sim_scorer.py \
    --drone-sim   teams/myteam/drone_sim.py        \
    --drone       quadcopter                       \    # or vtol
    --submission  teams/myteam/submission.yaml     \
    --output      score.json
```

Runs the four Tier 0 gate tests + each Tier 1 feature declared in
`submission.yaml`. Prints a JSON breakdown with per-test pass/fail and
the metric that decided each one. See [`SIM_SCORING.md`](SIM_SCORING.md)
for the rubric.

---

## Scenario YAML

```yaml
scenario_id: "medium"
duration_max: 60.0
battery_initial: 1.0
battery_drain_rate: 0.008
battery_drain_aggressive: 0.02
drone_start:
  position: [10.0, 10.0, 8.0]
  attitude: [0, 0, 0]
boat:
  trajectory_type: "linear"  # static | linear | curve | random_walk
  start_position: [0.0, 0.0, 0.15]
  speed: 1.5
  heading_initial: 0.5
  curve_radius: 18.0          # only used by 'curve'
  curve_direction: 1          # only used by 'curve'
  oscillation:
    roll_amplitude: 0.05
    pitch_amplitude: 0.03
    frequency: 0.4
wind:
  enabled: true
  mean_force: [0.3, 0, 0]
  gust_amplitude: 0.2
  gust_frequency: 0.1
camera:
  noise_level: 0.02            # additive Gaussian sigma in [0, 1]
  motion_blur_kernel: 0        # 0 = off; odd integer >= 3 = horizontal blur of N px
  fog_density: 0.05            # 1/m, Beer-Lambert
  fps: 20                      # 0 = render every env step (50 Hz)
  occlusion_probability: 0.0
  occlusion_duration: 1.0
landing:
  yaw_alignment_tol_deg: 20.0  # max yaw misalignment between drone fuselage and boat length
seed: 42
```

See `scenarios/eval_template.yaml` for the full annotated schema. The
private eval scenarios use the same schema with values sampled from
documented ranges (do not assume specific numbers — see
[`CHALLENGE.md`](CHALLENGE.md#forbidden-things)).
