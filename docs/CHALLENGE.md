# The Challenge

## Scenario

You're flying over the Mediterranean. A research vessel — the only
platform within hundreds of nautical miles — has just radioed in: it's
running, and it can't slow down. You have about a minute of battery
left, and the drone has to land on the boat's deck.

The boat is 6 m × 1.5 m. Its hull is mostly off-limits — what you have
to land on is a small **1 m × 1 m raised platform** built on top of the
deck. The platform pitches and rolls with the swell, and there's wind
besides. Painted on the platform is a single ArUco marker
(DICT_5X5_100, ID = 0, edge length 0.8 m). That marker is your only
way of localizing the platform — no GPS handoff from the ship, no
radar, no lidar.

The drone is a single airframe shipped as a YAML spec:

* **`drones/vtol.yaml`** — 10 kg heavy multirotor with a 1.7 m
  wingspan, asymmetric inertia (`Ixx=2.54`, `Iyy=3.47`, `Izz=5.74`),
  and weak yaw authority (small `k_Q / k_T` ratio). Always operated as
  a multirotor — no forward-flight transition, no tilt servos.
  Reaching steady-state yaw rate takes ~2 s, so you cannot use yaw to
  bail out of a late misalignment.

**Your job is twofold:**

1. **Build a `DroneSimulator`** for this airframe. The env owns
   boat / wind / camera / contact / scoring; you own the rigid-body
   physics that turns motor commands into state evolution.
2. **Build an `Agent`** that lands the drone on the platform.

Both are graded — see [`SIM_SCORING.md`](SIM_SCORING.md) for the sim
rubric, [`AGENT_SCORING.md`](AGENT_SCORING.md) for the agent rubric.
Total score is the sum.

## Landing condition

A platform contact counts as **LANDED** iff all four hold:

| Condition | Default threshold |
| --- | --- |
| `descent_velocity` at touchdown | `< 3.0 m/s` |
| Drone center within platform footprint | `\|xy_err\| < 0.5 m` per axis |
| Fuselage axis aligned with boat heading (mod π) | `< 30°` (scenario-dependent) |
| Platform contact (not hull, not water) | `getClosestPoints` |

Fuselage alignment is the trickiest of the four: the drone's body x-axis
(forward) must be aligned with the boat's heading axis modulo π. Land
sideways and you crash. The tolerance is set per scenario in
`landing.yaw_alignment_tol_deg` (easy: 35°, medium: 25°, hard: 20°).

The VTOL has weak yaw authority by design — start aligning yaw
**early** in the approach phase or you won't make the tolerance.

## Architecture overview

Three artifacts collaborate at every env step:

```
┌─────────────────────────────────────────────────────────────────┐
│              env (sealed: boat + wind + camera + scoring)       │
│                                                                 │
│   obs = {camera, drone_state, battery, time}                    │
│              │                                                  │
│              ▼                                                  │
│        ┌───────────────────┐                                    │
│        │   AGENT (you)     │  perceive → estimate → decide →    │
│        │                   │  control                           │
│        └───────────────────┘                                    │
│              │ motor_cmds (per-motor throttle, [0, 1])          │
│              ▼                                                  │
│        ┌───────────────────┐                                    │
│        │ DRONE_SIM (you)   │  motor → RPM → thrust →            │
│        │                   │  Newton-Euler → new state          │
│        │  (reads spec YAML)│                                    │
│        └───────────────────┘                                    │
│              │ new state                                        │
│              ▼                                                  │
│   env: contact check + termination + scoring                    │
└─────────────────────────────────────────────────────────────────┘
```

The simulator never sees the camera, the boat, or the scenario YAML —
it gets only its own action and a world-frame external force (wind +
optionally an `ext_ground_z` for ground-effect models). The agent
never sees `info` (which has the boat ground truth for evaluation
only).

## What your agent receives

Each environment step (50 Hz) you receive an observation:

```python
obs = {
    "camera":  np.ndarray,        # (480, 640, 3) RGB uint8 from a downward
                                  # camera mounted on the drone (~90° FOV)
    "state": {
        "position":         np.ndarray,  # (3,) world frame, m
        "velocity":         np.ndarray,  # (3,) world frame, m/s
        "attitude":         np.ndarray,  # (3,) roll/pitch/yaw, rad
        "angular_velocity": np.ndarray,  # (3,) world frame, rad/s
    },
    "battery": float,              # 0..1, monotonically non-increasing
    "time":    float,              # simulated seconds since reset
}
```

Note the camera frame may be **degraded** by the scenario:
* **`fps`** — fewer than 50 frames per second arrive; in between, the
  agent sees the *previous* frame. Hard scenarios run at 10 Hz.
* **`fog_density`** — Beer-Lambert blend toward marine haze, scales
  with altitude.
* **`motion_blur_kernel`** — horizontal box blur, simulates camera
  shake.
* **`occlusion_probability` / `occlusion_duration`** — the frame is
  briefly blacked out (occluding spray, sea-bird, etc.).

You return an action — **per-motor throttle**, shape determined by
your drone spec:

```python
action = np.ndarray  # (drone_spec.num_motors,) in [0, 1]
                     # one throttle command per motor
```

If you don't want to write your own attitude controller, the repo
ships `boat_landing.controllers.DefaultAttitudeController`: a tuned PD
attitude + rate-loop yaw + inverse mixer. It maps the legacy
`(thrust_norm, roll, pitch, yaw_rate)` setpoints to motor throttles
using your drone spec. The baseline agent uses it. See
[`API.md`](API.md) for the full contract.

## What your DroneSimulator must implement

The contract is documented in
[`boat_landing/drone_interface.py`](../boat_landing/drone_interface.py)
(the `DroneSimulator` Protocol). Minimum viable simulator (Tier 0,
mandatory):

* Read the spec YAML and use mass, inertia, motor placements, propeller
  coefficients, etc.
* Motor RPM dynamics (first-order lag is enough).
* Static propeller model: `T = k_T · ω²`, `Q = k_Q · ω²`.
* Body-frame mixer (sum thrust forces + reaction torques).
* Newton-Euler integration with quaternion attitude.
* Apply gravity internally (you know the mass).

Beyond Tier 0, six **auto-tested fidelity features** (5 points each)
let you push your simulator score up to 30. See
[`SIM_SCORING.md`](SIM_SCORING.md) for the full rubric and the
exact pass criteria.

The reference implementation at
[`agents/drone_sim_baseline.py`](../agents/drone_sim_baseline.py)
implements Tier 0 + first-order motor lag + cross-axis Newton-Euler
coupling. Fork it as your starting point.

## What your agent and sim CANNOT access

The boat's ground-truth pose is **never** present in `obs`. It lives
only in `info`, which is intended for evaluation and debugging —
touching it in your agent or sim is cheating and disqualifying.

In particular, you are **not allowed** to:

- Read `info['boat_position']`, `info['boat_velocity']`, or any other
  ground-truth field from the env.
- Read the scenario YAML directly to extract the boat's trajectory
  parameters, oscillation, wind force, or seed.
- Construct or import a `BoatLandingEnv` to introspect its private state.
- Monkey-patch the env, the boat module, the wind module, or any other
  internal class.
- Have the `drone_sim` import `boat_landing.boat`, `boat_landing.wind`,
  `boat_landing.camera`, or anything else from the env. The sim sees
  *only* what `step(...)` receives.

You **are** allowed to:

- Use any pip-installable Python library (within reason — heavy GPU
  models that won't run on the eval machines won't help you).
- Cache data you compute from `obs` across steps.
- Train a model offline (on your own scenarios) and load weights at
  runtime.
- Use the env in your dev workflow with public scenarios — it's the
  *eval-time* sealing that matters.

## What you submit

Three files (or a directory containing them):

1. **`drone_sim.py`** — your `DroneSimulator` implementation. Must
   expose `make_drone_sim(spec_path)` or a `DroneSim` class.
2. **`agent.py`** — your `Agent`. Must expose `make_agent(drone_spec)`
   (preferred) or an `Agent` class.
3. **`submission.yaml`** — file paths and which Tier 1 simulator
   features you implemented. Template at
   [`evaluation/submission.yaml.template`](../evaluation/submission.yaml.template).

The agent class must implement:

```python
class Agent:
    def act(self, obs: dict) -> np.ndarray: ...
    # Optional, but unlocks the estimation bonus:
    def get_last_estimate(self) -> dict | None: ...
```

`get_last_estimate()`, when present, must return either `None` (opting
out of the estimation bonus) or a dict with at least:

```python
{
    "position": np.ndarray,  # shape (3,), best estimate of boat pose
    "velocity": np.ndarray,  # shape (3,), optional but recommended
}
```

The scorer calls `get_last_estimate()` after each `act()` to log the
`(true_boat_position, your_estimate)` pair and compute an RMSE bonus.

## Scenarios

Three public scenarios ship with this repo (`scenarios/`):

| Scenario | Boat motion       | Wind            | Oscillation | Camera fps | Fog density | Yaw tol |
| -------- | ----------------- | --------------- | ----------- | ---------- | ----------- | ------- |
| EASY     | static            | none            | none        | 50 (none)  | 0           | 35°     |
| MEDIUM   | linear @ 1.5 m/s  | mild (0.3 N)    | mild        | 20         | 0.05        | 25°     |
| HARD     | curved @ 2.5 m/s  | gusty (0.6 N)   | strong      | 10         | 0.15        | 20°     |

The real evaluation scenarios (kept private until the event ends) use
the same schema. They are tuned to be no harder than HARD but to
reward agents that handle a different *style* of boat motion than the
public ones — bring perception/estimation that generalizes, not
parameters that overfit.

## Forbidden things

- **Hardcoding scenario parameters** (boat speed, oscillation
  amplitude, wind force). Your agent must work without knowing these
  in advance.
- **Reading the scenario YAML.** This counts as cheating.
- **Reaching into `info`.** It is for evaluation only.
- **Modifying `act()` in your fork** of `agent_template.py`. The four
  pipeline stages (perceive / estimate / decide / control) are how we
  compare submissions; keep them visible.
- **Hardcoding drone spec parameters** in your `drone_sim.py`. The
  simulator must read mass, inertia, motor placement, propeller
  coefficients, etc. from the spec — values baked into the code fail
  the Tier 1 tests that read those fields.
- **Bluffing in `submission.yaml`**. Declaring a Tier 1 feature you
  didn't implement runs the test, fails it, and visibly flags your
  submission. See [`SIM_SCORING.md`](SIM_SCORING.md#anti-bluff).
- **Reverse-engineering the reference simulator binary.** It ships as
  a `.so`/`.pyd` deliberately. Time spent decompiling is time not
  spent on perception/control — the qualitative description in
  `boat_landing/reference_sim/physics_notes.md` is all you need.
- **Submitting `agents/drone_sim_baseline.py` verbatim.** The AST
  similarity check caps your sim-track score at 0 if you do.
- **Introspection / sandbox-escape attempts**: `import inspect` to
  walk the call stack and reach env internals, `import gc` to scan
  the heap for the env reference, reading `info` directly, importing
  `BoatLandingEnv` inside your agent to construct another env,
  reading files outside your own folder (we don't ship the eval
  scenarios alongside your code — but if you discover any, reading
  them is cheating). Any of these patterns triggers
  `evaluation/code_audit.py` and gets your submission flagged for
  manual review. Confirmed cheats are **disqualified**.

## Hardware track (optional, +60 pts)

In addition to the agent + sim tracks, you may design and present an
auxiliary **physical mechanism** that extends the boat-landing system
beyond what bare flight control can handle (nets, electromagnets,
gripper claws, retractable feet, magnetic plates, anything else you
can defend). The HW track is **all rubric, no automation** and is
worth up to **60 points** on top of the agent + sim totals.

Three components:

| Item | What | Max pts |
| --- | --- | --- |
| **HW.1 Concept pitch** | Video (1–3 min) + supporting materials (slides, Blender/SolidWorks renders, hand-drawn schematics) that explain the scenario you're solving, the mechanism, and why this mechanism vs alternatives | 20 |
| **HW.2 CAD / BOM / schema** | Engineering drawings, electrical schematic, parts list with real part numbers | 10 |
| **HW.3 Working implementation** | A runnable simulation/prototype of the mechanism in any framework — Python, Simulink, Gazebo, ROS, custom — plus a short demo video/screencast showing it operating, plus a 1-page writeup with quantitative results. NO integration into this repo required. | 30 |

HW.3 explicitly does **not** require forking this repo. Use whatever
toolchain best demonstrates the mechanism in action — a Simulink model
of a magnetic gripper, a Python simulation of a net deployment, a
small Gazebo world for a retractable-feet mechanism, anything that
shows the mechanism doing useful work and producing measurable
behavior. The deliverable is the trio: working code + demo evidence +
short writeup with numbers.

HW.3 is scored 0/10/20/30 by a human reviewer based on: implementation
runs end-to-end without hand-waving, the demo makes the mechanism's
benefit visible, and the writeup ties the design choices to the
numbers.

## Total score

| Track | Max | Who scores |
| --- | --- | --- |
| Agent — landing performance | 70 | Automatic ([`AGENT_SCORING.md`](AGENT_SCORING.md)) |
| Agent — HW-readiness bonuses | 45 | Automatic |
| Drone simulator — fidelity | 30 | Automatic ([`SIM_SCORING.md`](SIM_SCORING.md)) |
| Hardware track | 60 | Human rubric |
| **Total** | **205** | |

The agent track and the drone-sim track are **independent**: your sim's
score doesn't carry over to the agent score, because the agent is run
against the organizer reference sim (see
[`SIM_SCORING.md`](SIM_SCORING.md#reference-simulator-organizer-owned)).

## Time budget

12 hours. Two of those go to setup, demos, and meals — so plan on ~10
hours of build time. Suggested split:

- **1 h**: read this doc + `SIM_SCORING.md` + `AGENT_SCORING.md` + `API.md`.
- **1 h**: get the baseline agent + baseline sim running on EASY locally.
- **2 h**: simulator. Add the Tier 1 features that matter most for
  landing fidelity (motor lag, ground effect, aero drag are the high-
  value ones). Each is worth +5; pick what you can defend.
- **4 h**: perception + estimation + boat-motion estimator. This is
  where the biggest *agent* score gains are.
- **2 h**: control: fuselage-axis yaw alignment (the VTOL is sluggish
  in yaw), velocity feed-forward on the boat estimate, descent profile.
- **1 h**: run on MEDIUM and HARD; debug failure modes; tune.

If you're attempting the HW track too, redistribute: 1–2 people on
HW (concept pitch + implementation), the rest on agent + sim. The HW
track is fully decoupled from the code repo — pick the toolchain that
fits your mechanism best.

Good luck.
