# Catch the Boat

Hackathon challenge by **Project Europe** and **SkyEU** — Milan, 2026.

Build an autonomous agent **and** a drone-physics simulator that
together land a multirotor on a moving boat using only a downward
camera, an ArUco marker on the deck, and the drone's own state.
12 hours, two scored artifacts, three difficulty tiers.

---

## The Challenge

You ship **two** Python files:

* `drone_sim.py` — your physics: motor commands → RPM → thrust →
  Newton-Euler → new state. Reads `drones/vtol.yaml` (10 kg heavy
  multirotor with weak yaw — the only airframe shipped).
* `agent.py` — your perception + estimation + control. Given a 50 Hz
  stream of `(camera, drone state, battery, time)` observations, it
  outputs a per-motor throttle vector.

The env (boat, wind, camera, contact, scoring) is sealed and provided
by the organizers. Your agent never sees the boat's true position — it
has to infer it from the ArUco marker. Landing is "soft AND on-platform
AND fuselage aligned with boat heading (mod π)" — yaw alignment is
part of the landing condition.

Three public scenarios (`easy`, `medium`, `hard`) ship with this repo.
The real evaluation scenarios live with the organizers and are
revealed after the event.

**Four independent scores, summed (max 205 pts):**
* Agent landing performance (0–70) — see [`docs/AGENT_SCORING.md`](docs/AGENT_SCORING.md).
* Agent HW-readiness bonuses (0–45) — FC-compatible output, latency budget, marker-loss recovery.
* Simulator physical fidelity (0–30) — see [`docs/SIM_SCORING.md`](docs/SIM_SCORING.md).
* Optional hardware track (0–60) — concept pitch + CAD + working implementation in any framework.

Your **agent** is always scored against the organizer reference
simulator (a compiled binary in `boat_landing/reference_sim/`), not
your own drone sim. The two tracks are decoupled.

Full spec: [`docs/CHALLENGE.md`](docs/CHALLENGE.md). Evaluation runs
inside a fixed Docker container; see [`docs/DOCKER.md`](docs/DOCKER.md).

---

## Quick Start

> **Use Python 3.10.** PyBullet's only Windows-compatible PyPI builds
> target ≤ cp310 (see [Windows install notes](#windows-install-notes)
> below). On Linux/macOS any 3.10–3.12 works.

```bash
git clone <this-repo>
cd catch-the-boat

# Linux / macOS:
python3.10 -m venv .venv
source .venv/bin/activate

# Windows (PowerShell) — the launcher exposes 3.10 as `py -3.10`:
#   py -3.10 -m venv .venv
#   .venv\Scripts\activate

pip install -r requirements.txt
python scripts/test_setup.py
python scripts/run_baseline.py --scenario easy --visualize --gui
```

You should see a pygame window with the chase view, the drone's
downward camera (with the ArUco marker visible), and a HUD. The
baseline lands within ~12 simulated seconds. `--gui` is optional but
makes the simulation 3–5× faster on machines without a beefy CPU
(uses GPU rendering); see [Performance](#performance) below.

### Windows install notes

PyBullet ships **no binary wheels for Windows on PyPI** — pip will fall
back to a source build, which needs the MSVC compiler. Pick whichever
of these is least painful:

1. **Install Microsoft C++ Build Tools** (one-time, ~7 GB):
   ```powershell
   winget install --id Microsoft.VisualStudio.2022.BuildTools --override "--passive --wait --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
   ```
   Then `pip install -r requirements.txt` works as on Linux/macOS.
2. **Use Miniconda + conda-forge** (no compiler required):
   ```powershell
   winget install --id Anaconda.Miniconda3
   conda create -n catch-the-boat python=3.10 -y
   conda activate catch-the-boat
   conda install -c conda-forge pybullet -y
   pip install -r requirements.txt
   ```
3. **Use WSL2** and follow the Linux instructions inside the WSL shell.

`uv venv --python 3.10` works as a drop-in for `python3.10 -m venv`
(and is much faster).

---

## Repository Structure

```
catch-the-boat/
├── boat_landing/          # Sealed env: boat + wind + camera + scoring
│   ├── env.py             #   - kinematic drone body, contact detection
│   ├── drone_interface.py #   - DroneSimulator Protocol + DroneSpec
│   └── controllers.py     #   - DefaultAttitudeController utility
├── drones/                # Drone physical spec (read-only YAML)
│   └── vtol.yaml          #   - 10 kg heavy multirotor, weak yaw
├── agents/                # Reference + template for your submission
│   ├── drone_sim_baseline.py   # tier-0 reference simulator
│   ├── agent_baseline.py       # ArUco + PIDs reference agent
│   └── agent_template.py       # empty skeleton to fork
├── scenarios/             # Public scenarios + eval template
├── evaluation/            # Two scorers: agent (evaluate.py) + sim (sim_scorer.py)
│   ├── sim_validation/    #   - 4 gate + 6 auto-tested feature tests
│   └── submission*.yaml   #   - submission manifest (template + baseline)
├── docs/                  # CHALLENGE, AGENT_SCORING, SIM_SCORING, API, TIPS, COMMANDS
├── visualizer/            # Pygame demo viewer
├── scripts/               # Launchers, setup checker, ArUco generator
└── tests/                 # pytest suite (env + baseline + scoring + sim_validation)
```

---

## The two APIs

Your **agent** implements one method:

```python
class Agent:
    def __init__(self, drone_spec): ...    # drone_spec is optional
    def act(self, obs: dict) -> np.ndarray: ...
        # returns shape (drone_spec.num_motors,), values in [0, 1]
        # one throttle command per motor
```

Your **drone simulator** implements three:

```python
class DroneSim:
    def __init__(self, spec_path: str): ...
    def reset(self, position, attitude): ...
    def step(self, motor_cmds, ext_force_world, dt,
             ext_torque_world=None, ext_ground_z=None): ...
    def get_state(self) -> DroneState: ...
```

The agent ships per-motor throttles (motor-level action space). If you
don't want to write your own attitude controller, the repo provides
`boat_landing.controllers.DefaultAttitudeController` — a tuned PD
attitude + rate-loop yaw + inverse mixer that maps high-level
`(thrust, roll, pitch, yaw_rate)` setpoints to motor throttles using
your drone spec. The baseline agent uses it.

Optionally, expose `get_last_estimate()` on the agent to unlock the
estimation bonus (0–15 points). See [`docs/API.md`](docs/API.md) for
the full contract: observation/action shapes, intrinsics, termination
conditions, simulator protocol.

**Start your work** by copying:

```bash
cp agents/agent_template.py    teams/<your_team>/agent.py
cp agents/drone_sim_baseline.py teams/<your_team>/drone_sim.py
cp evaluation/submission.yaml.template teams/<your_team>/submission.yaml
```

Replace the four agent pipeline stages (perceive / estimate / decide /
control). Replace whichever parts of the simulator you want to enrich
for fidelity bonuses.

---

## Scenarios

| Scenario | Boat motion       | Wind            | Oscillation | Camera fps | Fog | Yaw tol |
| -------- | ----------------- | --------------- | ----------- | ---------- | --- | ------- |
| EASY     | static            | none            | none        | 50         | 0   | 35°     |
| MEDIUM   | linear @ 1.5 m/s  | mild (0.3 N)    | mild        | 20         | 0.05 | 25°    |
| HARD     | curved @ 2.5 m/s  | gusty (0.6 N)   | strong      | 10         | 0.15 | 20°    |

The baseline lands on EASY, times-out on MEDIUM, and never lands on
HARD — there's lots of headroom for a Kalman filter on the boat
estimate, velocity feed-forward, **active yaw alignment**, and
motion-blur / fog robustness. See [`docs/TIPS.md`](docs/TIPS.md) for
the priority order.

Run a scenario with the baseline (defaults to VTOL spec):

```bash
python evaluation/evaluate.py --scenario easy --headless --seed 42
```

You can swap each piece independently:

```bash
python evaluation/evaluate.py \
    --agent      teams/myteam/agent.py \
    --drone-sim  teams/myteam/drone_sim.py \
    --scenario   hard --headless --seed 42
```

The CLI prints a JSON with score and breakdown, plus a one-line-per-
sim-second progress trace on stderr. Use it to A/B your changes.

**Score your simulator** with the dedicated CLI:

```bash
python evaluation/sim_scorer.py \
    --drone-sim   agents/drone_sim_baseline.py \
    --drone       vtol \
    --submission  evaluation/submission_baseline.yaml
```

Output is the simulator-quality breakdown (Tier 0 gate, Tier 1 features
declared in `submission.yaml`, total). The baseline scores 15/30
(motor lag + cross-coupling + substepping). See
[`docs/SIM_SCORING.md`](docs/SIM_SCORING.md) for the rubric.

---

## Performance

PyBullet's default TINY (CPU) renderer makes each step ~50 ms wall on
typical laptops, so a 60 s scenario takes 3–5 minutes wall-clock.

If your machine has a GPU and an OpenGL context, pass `--gui` to switch
to PyBullet's hardware renderer (3–5× faster):

```bash
python scripts/run_baseline.py --scenario easy --visualize --gui
python evaluation/evaluate.py  --agent agents/agent_baseline.py \
                               --scenario easy --gui --seed 42
```

`--gui` opens PyBullet's debug window in addition to whatever else
you're showing. For batch evaluation in CI, stick to `--headless` and
accept the wall-clock cost — it keeps the runtime deterministic and
display-free.

---

## Scoring

Four scores, summed (max 205 pts):

```text
total = agent_landing_score (0..70)             # AGENT_SCORING.md
      + agent_hw_readiness_bonus (0..45)        # AGENT_SCORING.md (FC compat + latency + recovery)
      + simulator_quality_score (0..30)         # SIM_SCORING.md
      + hardware_track_score (0..60)            # CHALLENGE.md HW track
```

The agent track is always scored against the organizer reference
simulator (compiled binary), not against your own `drone_sim.py`. So
"less realistic sim → easier landing" is structurally impossible.

**Agent score** rewards landing precision, speed, battery preservation,
soft touchdown, and (optionally) good boat-state estimation. Formula:

```text
score = base * precision_factor * time_factor * battery_factor
        + soft_landing_bonus + estimation_bonus
```

with multiplicative factors floored so a slow but successful landing
is still meaningfully rewarded. Detail and worked examples in
[`docs/AGENT_SCORING.md`](docs/AGENT_SCORING.md).

| Outcome           | Agent score range |
| ----------------- | ----------------- |
| Crash             | `-20`             |
| Soft-fail (timeout, battery, wall-cap, error, OOM) | `0`             |
| Land (typical)    | `~25–65`          |
| Land (great)      | `~65–70` + HW-readiness bonuses |

The shipped `agents/agent_baseline.py` lands EASY at ~69/70 in
~8.5 s (latency + estimation bonuses earned; soft-landing bonus
forfeited because the LAND descent is bang-bang). It times out on
MEDIUM and HARD — that's where most of the headroom lives.

**Simulator score** rewards physical fidelity. Four mandatory gate
tests (incl. T0.5 hidden category) + six auto-tested fidelity features:

| Component                                       | Max points |
| ----------------------------------------------- | ---------- |
| Tier 0 gate (mandatory, incl. T0.5 hidden)      | gate       |
| Tier 1 auto features (5 × 6)                    | 30         |
| **Total**                                       | **30**     |

The code-quality and `SIMULATOR.md` rubrics from earlier drafts were
retired — the track is now 100% automated, and the human-judged half
of the challenge moved to the agent (HW-readiness) and hardware
tracks instead.

Full rubric and per-feature pass criteria in
[`docs/SIM_SCORING.md`](docs/SIM_SCORING.md). The reference baseline
sim scores 15/30 (motor_lag + cross_coupling + substepping).

---

## Submission

Three files (or a directory containing them):

1. **`drone_sim.py`** — exposes `make_drone_sim(spec_path)` or a
   `DroneSim` class.
2. **`agent.py`** — exposes `make_agent(drone_spec)` or an `Agent`
   class.
3. **`submission.yaml`** — file paths and which Tier 1 simulator
   features you implemented. Template at
   [`evaluation/submission.yaml.template`](evaluation/submission.yaml.template),
   detailed schema in [`docs/SIM_SCORING.md`](docs/SIM_SCORING.md#submission-manifest).

The scorers dynamically load all three and run them against the
private eval scenarios + the validation suite.

The exact submission mechanics (Slack channel, GitHub repo, USB drop)
will be announced at the opening keynote.

---

## Tips & Gotchas

The single biggest **agent** lever is a **Kalman filter on the boat's
(x, y, vx, vy)**. The baseline trusts every ArUco detection blindly —
that's noise the controller will chase forever. Fix that first. The
second biggest is **active yaw control** to satisfy the fuselage-aligned
landing condition (the baseline doesn't yaw and it shows).

The single biggest **simulator** lever is implementing the cheap Tier 1
features first: motor lag is already in the baseline, so you get to 5
points easily. Adding ground effect, aero drag, and battery sag are
each +5 and reasonably straightforward physics — that's another +15
without exotic modelling.

Other ideas, by module, in [`docs/TIPS.md`](docs/TIPS.md).

---

## FAQ

**Q. Can I use any pip-installable library?**
Yes. Bring your own perception stack, RL framework, MPC solver — as
long as it runs on CPU within reasonable wall-time and doesn't read
`info` from the env.

**Q. Can I read the scenario YAML to find the boat's parameters?**
No. The eval scenarios are private and tuned differently. Your agent
must work without knowing the parameters in advance.

**Q. Can I monkey-patch the env to expose ground truth?**
No. The same reason. The judges run your code against private scenarios
and will catch this.

**Q. The PyBullet GUI window is empty / weird on macOS / Linux.**
Drop `--gui` and use the pygame visualizer alone:
`python scripts/run_baseline.py --scenario easy --visualize`. It
renders offscreen and displays via pygame, which is more portable.
The trade-off is the slower TINY (CPU) renderer; see
[Performance](#performance). For batch evaluation always use
`--headless`.

**Q. The simulation is much slower than real-time.**
That's expected on CPU-only rendering. Add `--gui` to opt into the
GPU OpenGL renderer (3–5× faster). See [Performance](#performance).

**Q. My ArUco detection is jittery near the edges of the FOV.**
Expected. Tune `cv2.aruco.DetectorParameters` (especially
`cornerRefinementMethod = CORNER_REFINE_SUBPIX`), or build a small
filter — see [`docs/TIPS.md`](docs/TIPS.md).

**Q. Do I have to yaw the drone?**
**Yes.** The landing condition requires the drone's fuselage to be
aligned with the boat's heading axis (modulo π) within a tolerance
that ranges from 35° (easy) to 20° (hard). The baseline doesn't yaw
and fails this on every non-trivial scenario. Yaw authority on the
VTOL is weak (slow `Izz`, small `k_Q/k_T`) — start aligning during
APPROACH, not at touchdown.

**Q. Do I have to write a `drone_sim.py`?**
Yes — but only for the sim-track score. The agent itself is run
against the organizer **reference simulator** (a compiled binary in
the `:full` Docker image), not against your sim. Your sim is judged
independently by the sim-track validation suite. You can fork
`agents/drone_sim_baseline.py` and submit it unchanged — but the
AST similarity check caps verbatim resubmissions at 0; **rename
the file and modify at least one feature** to score the baseline's
15/30 floor.

**Q. Can I yaw the drone if I just want to point the camera?**
Yes. Yaw is also useful for keeping the marker in the center of frame
during APPROACH. The cost is energy (battery drain scales with angular
velocity).

**Q. Where do I print debug info?**
Use stderr (`print(..., file=sys.stderr)`). The CLI captures stdout
for the JSON result.

---

## License & Credits

Released under the [MIT License](LICENSE).

Built for the Milan 2026 hackathon by Project Europe and SkyEU.

Inspired by [`gym-pybullet-drones`](https://github.com/utiasDSL/gym-pybullet-drones)
(the action/observation contract here is gymnasium-style, and the
Crazyflie 2.X scale and PD controller follow that project's lead — but
this starter runs on PyBullet directly, no extra dependency required).
