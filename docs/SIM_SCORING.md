# Simulator Scoring

This document explains how the **drone simulator** is scored, separately
from your agent. The agent is judged on landing performance (see
[`AGENT_SCORING.md`](AGENT_SCORING.md), 0–70 points + 0–45 HW-readiness bonuses).
The simulator is judged on **physical fidelity** — how seriously you
modelled the airframe — and is worth up to **30 points** on top of
your agent score. The optional hardware track (rubric, see
[`CHALLENGE.md`](CHALLENGE.md)) adds another 60.

```
total_score = agent_landing_score (0..70)
            + agent_hw_readiness  (0..45)
            + simulator_score     (0..30)
            + hardware_track      (0..60)        # optional, rubric
```

The agent and simulator tracks are **independent**: a sloppy participant
sim does NOT make agent landing easier, because the agent is always run
against the organizer-owned reference simulator (see
[Reference simulator](#reference-simulator-organizer-owned)).

---

## TL;DR

| Tier | Points | What it is | Who scores it |
| --- | --- | --- | --- |
| **0 — Gate** | PASS / FAIL | 4 mandatory tests (incl. hidden T0.5). Fail any → simulator score = 0. | Automatic |
| **1 — Auto features** | up to **30** (5 × 6) | Six physically-motivated features. Each is verified by an automated test. Declare in `submission.yaml`. | Automatic |
| **Total sim track** | **30** | | |

> The track is 100% automated. The human-judged half of the challenge
> lives in the HW track and the agent's HW-readiness bonuses. Your
> sim's score does NOT carry over to the agent score; the two are
> independent.

## How the sim track interacts with the agent track

Two important rules:

1. **The agent is always scored against the organizer reference
   simulator**, NOT against your sim. So a sim that's "favourable"
   (less realistic) does not give your agent an advantage. Both tracks
   are independent.
2. **Submitting the public baseline verbatim is detected** by an AST
   similarity check; if matched, the sim-track total is forced to 0
   regardless of how many tests passed. Fork it, change it, ship it —
   don't paste it.

---

## What your simulator must implement (refresher)

You ship a Python file that conforms to the
[`DroneSimulator`](../boat_landing/drone_interface.py) protocol. The env
calls it as follows during every physics substep (250 Hz):

```
drone_sim.step(motor_cmds, ext_force_world, dt,
               ext_torque_world=None,
               ext_ground_z=None)
```

* `motor_cmds`: `(num_motors,)` array, throttle ∈ [0, 1] per motor.
* `ext_force_world`: world-frame external force `(3,)` — wind today,
  possibly more in future.
* `dt`: substep duration, seconds.
* `ext_torque_world`: optional world-frame disturbance torque, default 0.
* `ext_ground_z`: world-frame z of the surface directly below the drone
  (boat platform top when over the platform, hull top when over the
  hull, water plane = 0 elsewhere). **Used by ground-effect models.**
  Free to ignore for tier-0 sims.

Your `get_state()` returns a `DroneState` with `position`, `velocity`,
`quaternion (x,y,z,w)`, `angular_velocity_body`, and optionally
`motor_omegas` — see [`drone_interface.py`](../boat_landing/drone_interface.py).

The reference baseline implementation lives in
[`agents/drone_sim_baseline.py`](../agents/drone_sim_baseline.py). Fork
it, replace what you want, keep the contract.

---

## Tier 0 — Gate (PASS / FAIL)

Four mandatory tests. **If any fails, your simulator score is 0.** No
exceptions, no partial credit. These verify the contract is honoured
and your sim is even fit for evaluation.

### T0.1 — Protocol contract

Your sim must:
- expose `.spec` (a parsed `DroneSpec`)
- support `reset(position, attitude)` with `(3,)` arrays
- support `step(motor_cmds, ext_force_world, dt, ...)` without raising
- return a `DroneState` from `get_state()` with the documented shapes
- emit a unit-norm quaternion in `state.quaternion`

Failure mode: typo in the API, returning a tuple instead of a
`DroneState`, missing `motor_omegas` field on the dataclass, etc.

### T0.2 — Hover steady-state

With per-motor throttle equal to the **analytic hover throttle**
(`throttle_hover = sqrt(mg/4 / (k_T · ω_max²))` for symmetric specs),
no wind, no disturbance, no PID — the drone must stay within **1 cm of
its initial position over 5 simulated seconds**.

This catches: wrong `T = k_T · ω²` constant, motors not initialised to
hover RPM at `reset()`, broken gravity sign.

### T0.3 — Determinism

Two fresh sims, same spec, same seed, same sequence of motor commands
must produce final states that match to **|Δpos| < 1e-10 m**. No
wall-clock dependencies, no unseeded RNGs. (If you want randomness for
ablation, use a seed parameter in your sim's constructor.)

### T0.5 — Variable-dt robustness (hidden category)

Your simulator is called with `dt` values drawn from a non-constant
pattern in **[0.001, 0.050] s**. The same motor-command sequence is
also re-played against your sim with each step broken into ≤ 1 ms
chunks (so the integrator sees only small `dt`). Your sim must
produce final positions that agree between the two runs.

| Item | Value |
| --- | --- |
| Range of `dt` | [0.001, 0.050] s (announced) |
| Exact dt pattern | private |
| Pass threshold | private |
| Public default for local testing | 5 cm position diff, seed 1337 |

**Strategies that pass**:
- Internal sub-stepping that caps each integration tick at ≤ 5 ms
  regardless of incoming `dt`.
- RK4 or higher-order integrators.
- Adaptive timestep with error control.

**Strategies that fail**:
- Plain forward Euler with no sub-stepping.
- Hardcoded `internal_substeps = N` (assumes a specific outer `dt`).
- Stiff motor lag (τ ≈ 50 ms) integrated as one Euler step at
  `dt = 50 ms`.

The visibility policy is "category public, details private": you know
what shape of robustness we're testing, you don't know the exact dt
sequence or the threshold. Practising against the public default in
`evaluation/sim_validation/t0_variable_dt.py` puts you in the right
direction; passing it does not guarantee passing the eval run, but
failing it locally guarantees failing in eval.

> The IDs jump from T0.3 to T0.5: earlier drafts had a "T0.4
> spec-driven physics" test that fed the simulator two different drone
> YAMLs and verified the climb rates differed. With a single shipped
> airframe (`drones/vtol.yaml`) that test no longer applies; the
> Tier-1 tests (T1.A, T1.D) already exercise spec-driven values.

---

## Tier 1 — Auto-tested features (5 points each, max 30)

Six features. Each declared in `submission.yaml` triggers an automated
test. **Pass = 5 points, fail = 0 points.** Declaring a feature you
haven't implemented is a bluff — the test catches it (see [Anti-bluff](#anti-bluff)).

### T1.A — Motor RPM lag

The motor doesn't reach commanded RPM instantaneously. The simplest
honest model is a **first-order lag**:

```
dω/dt = (ω_cmd - ω) / τ_motor
```

with `τ_motor = spec.motor.time_constant`.

**Test**: command a step from 0 to full throttle, measure the time for
ω to reach **63.2 % of steady-state**. Pass criterion:

```
0.5 · τ_spec  ≤  τ_measured  ≤  2.0 · τ_spec
```

The wide tolerance lets richer models (full BLDC electromechanical with
`J_rotor·dω/dt = K_t·I − k_Q·ω²`) pass without exactly matching the
first-order spec.

**Required**: your sim must expose `state.motor_omegas`. A sim that
treats throttle → thrust instantly fails this test.

### T1.B — Battery voltage sag

Real batteries droop under load. Internal resistance `R_int` causes
voltage to fall as current rises:

```
V_bat(t) = V_nominal − R_int · I(t)
```

Lower voltage → lower `ω_max = V_bat / K_v` → lower achievable thrust.

**Test**: 20 seconds of full throttle on every motor. Measure the final
ω. Pass criterion: `ω_final / ω_max < 0.95` — at least 5 % sag.

**Required**: your sim must couple battery voltage (or some equivalent
"capacity remaining" signal) into either max RPM or per-motor thrust.
Hint: `spec.battery.internal_resistance` and `spec.battery.capacity_Wh`.

### T1.C — Aerodynamic drag

In vacuum, `v_z(t) = -g·t`. Real airframes have body drag — at minimum
linear (`F = -b·v`), often quadratic (`F = -c·|v|·v`).

**Test**: zero-throttle free fall for 4 seconds. Measure final vertical
velocity. Pass criterion:

```
|v_z_measured − v_no_drag| / |v_no_drag| > 0.02       # at least 2 % slower
```

with `v_no_drag = -g·t = -39.24 m/s`.

**Required**: your sim must apply *some* velocity-dependent damping
force. The test does not verify the exact formula — `aero.drag_linear`
times body velocity is enough; `aero.drag_quadratic` is icing. The 2 %
threshold is well above floating-point noise once the residual
motor-spin-down impulse has amortised over 4 s.

### T1.D — Cross-axis inertial coupling

Newton-Euler in the body frame is:

```
I · dω_body/dt + ω_body × (I · ω_body) = τ_body
```

The `ω × (I·ω)` term creates **cross-axis coupling**: if you spin a
body with asymmetric inertia, its angular velocity *precesses* — even
with zero applied torque. A sim that just does `I · dω/dt = τ` (drops
the gyro term) misses this.

**Test**: applies an asymmetric thrust burst on the VTOL spec
(asymmetric inertia: `Ixx=2.54`, `Iyy=3.47`, `Izz=5.74`) to imprint a
multi-axis ω, then commands hover throttle for 1 second. Pass criterion:
**ω rotates by more than 3°** during the free-precession second.

A sim that drops the cross term keeps ω constant → 0° rotation → fail.

### T1.E — Internal sub-stepping

Stiff motor dynamics (`τ ≈ 50 ms`) want a finer integration step than
the env's 250 Hz. A robust sim **sub-steps internally** when given a
big `dt`, instead of doing a single Euler step.

**Test**: run the same physical interval twice — once as one outer step
of `dt = 0.020 s`, once as 5 outer steps of `dt = 0.004 s`. Pass
criterion: position difference between the two runs **< 2 cm** at the
end. A single-Euler-step sim accumulates noticeable error and fails.

**Required**: detect `dt` and break it into smaller integration steps
internally (RK4 also helps). The threshold is loose so you don't have
to over-engineer.

### T1.F — Ground effect

Near a surface, rotor downwash recirculates and effectively amplifies
thrust. Standard textbook approximation (Cheng-Frantz):

```
T_eff = T · (1 + a · (R / h)²)        for h ≲ 4·R
T_eff = T                              otherwise
```

with `R = propeller radius`, `h = altitude above the surface`, `a ≈ 0.25–0.5`.

**Test**: places the drone at `z = ground_z + R` (one rotor radius
above a virtual surface at `ground_z = 0.5 m`, matching the env's
platform top), commands the **analytic free-flight hover throttle**, and
passes `ext_ground_z = 0.5` to every `step()` call. Integrates for
0.5 s. Pass criterion: **drone climbs > 5 cm** during that half-second.

A sim that ignores `ext_ground_z` produces `T = mg` exactly and
hovers (Δz ≈ 0 → fail). A sim that amplifies thrust by even a modest
factor at `h = R` shows visible climb (≈ 30 cm with `a = 0.25`).

**Required**: read `ext_ground_z` and use `state.position[2] - ext_ground_z`
as the altitude. Apply the multiplier to per-motor thrust. The boat
platform is the relevant surface during the landing approach — the env
passes the platform top z whenever the drone is laterally over it.

---

## Reference simulator (organizer-owned)

Your **agent** is never scored against your own drone simulator. It is
scored against the organizer-owned reference simulator shipped as a
compiled binary (`boat_landing.reference_sim`). The reference
implements all six Tier-1 features plus a few additional low-amplitude
effects — see `boat_landing/reference_sim/physics_notes.md` for the
qualitative description.

Why this matters:

- A "favourable" (less realistic) participant simulator does NOT help
  the agent score. Cheating-by-omission (skip motor lag in your sim so
  your agent loop is easier) is structurally impossible — your sim and
  the agent eval are decoupled.
- You CAN call `boat_landing.reference_sim` locally to dev your agent
  against the same physics it will face in eval. Pass
  `--use-reference-sim` to `evaluation/evaluate.py`.
- You CANNOT read the reference's source. It is distributed as
  `.so`/`.pyd`. Do not try to reverse-engineer it — the time is better
  spent on perception / estimation / control.

### Anti-copy check on your `drone_sim.py`

When `evaluation/sim_scorer.py` runs, the submission is first compared
against the public baseline (`agents/drone_sim_baseline.py`) at the AST
level (comments and whitespace normalised). If the AST matches
verbatim, the **sim-track total is capped at 0**, regardless of which
tests passed. This catches "I submitted the baseline unchanged"
submissions cleanly. Fork it, rename it, do not paste it.

---

## Submission manifest

Every team ships a `submission.yaml` declaring what's in their drop.
Template at [`evaluation/submission.yaml.template`](../evaluation/submission.yaml.template).
Minimal example:

```yaml
team:           "Team Sky"
drone_sim_path: "drone_sim.py"
agent_path:     "agent.py"

simulator_features:
  tier_1:
    motor_lag:        true
    battery_sag:      false
    aero_drag:        true
    cross_coupling:   true
    substepping:      false
    ground_effect:    true

notes: |
  We implemented Cheng-Frantz ground effect with a=0.30 and a quadratic
  body drag model fitted to wind-tunnel coefficients from [reference].
  Skipped battery sag and sub-stepping due to time budget.
```

* `simulator_features.tier_1.<feature>: true` — request that the
  feature's automated test be run for you. **Set `false` (or omit) if
  you didn't implement the feature.**

---

## Anti-bluff

> Declaring `motor_lag: true` when your sim doesn't model lag = the
> automated test runs and fails. You earn **0 for that feature** AND
> the breakdown JSON shows `"claimed": true, "passed": false` — visible
> to the reviewers and to the public scoreboard.

We deliberately do **not** apply a multiplicative penalty for bluffing
(an earlier draft did `score *= 0.5` for any false claim — we dropped
it as too punitive). The reputational signal is enough: judges see
exactly which features you bluffed on.

If you're unsure whether your model is "good enough" to claim a
feature: **run the test locally before submitting** (see below). The
test code is not secret.

---

## Running the scorer locally

Everything in `evaluation/sim_validation/` is part of the repo. You can
run the same scorer the organizers will use:

```bash
python evaluation/sim_scorer.py \
    --drone-sim    path/to/your/drone_sim.py \
    --drone        vtol \
    --submission   path/to/your/submission.yaml
```

Output is a JSON breakdown to stdout — every test, pass/fail, the
metric that decided it, and the running tally:

```
{
  "tier_0_gate": [ {...}, {...}, {...}, {...} ],
  "tier_1": [ ... ],
  "tier_0_passed": true,
  "tier_1_score": 15,
  "total_score": 15,
  "max_score_possible": 30,
  ...
}
```

You can iterate as fast as you want. Each test is also runnable on its
own as a Python module — see
[`tests/test_sim_validation.py`](../tests/test_sim_validation.py) for
how the organizers smoke-test the baseline.

---

## Baseline reference

The public baseline at `agents/drone_sim_baseline.py` is what ships
with the repo. It implements:

* Tier 0: all four gate tests pass, including T0.5 (the baseline does
  light internal sub-stepping at a 5 ms cap so plain-Euler error stays
  bounded for `dt` up to the env's 4 ms cadence and beyond).
* T1.A motor lag (first-order, `τ = spec.motor.time_constant`).
* T1.D cross-axis coupling (Newton-Euler with the `ω × Iω` term).
* T1.E internal sub-stepping (5 ms cap — required by T0.5 anyway).

It does NOT implement: battery sag, aero drag, ground effect.

Its declared `submission_baseline.yaml` honestly claims only the three
features it implements. Score:

```
TIER 0 PASSED: True
TIER 1 SCORE: 15/30
  motor_lag         claimed=True  passed=True  score= 5
  cross_coupling    claimed=True  passed=True  score= 5
  substepping       claimed=True  passed=True  score= 5
```

A submission needs to **beat 15/30 on simulator quality** to be ahead
of the baseline. With careful work, all six Tier 1 features are
achievable in the 12-hour window.

**Do NOT submit this file verbatim.** The similarity check in
`evaluation/sim_scorer.py` matches its AST and caps the sim-track
score at 0 if you do. Fork it, rename it, modify any one feature, and
you're past the cap.

---

## Questions

If a test feels unfair or its threshold seems wrong, **open an issue
on the repo before the event starts** — we'd rather adjust before
people are submitting than litigate after. Once the event is running,
the rubric is frozen.
