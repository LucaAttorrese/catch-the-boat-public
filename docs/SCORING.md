# Scoring

The scorer turns each episode into a single number. Higher is better.

## Formula

```text
if outcome == "CRASHED":           score = -20
elif outcome in {"TIMEOUT",
                 "OUT_OF_BATTERY"}: score =   0
elif outcome == "LANDED":
    base                = 100
    precision_factor    = max(0,    1 - landing_position_error / 1.5)
    time_factor         = max(0.5,  1 - time_to_land / duration_max)
    battery_factor      = max(0.3,  battery_remaining)
    soft_landing_bonus  = 10  if  max_descent_velocity < 1.0  else 0
    estimation_bonus    = compute_estimation_bonus(rmse)   # 0..20

    landing_score = (base * precision_factor * time_factor * battery_factor
                     + soft_landing_bonus + estimation_bonus)

    # HW-readiness add-ons (each independent, only on LANDED):
    fc_compat_bonus  = 5  if agent.act_setpoint is used
    latency_bonus    = 5  if act() p95 latency <= 20 ms (in eval container)
    recovery_bonus   = 5  if scenario is recovery AND drift+reacquire pass

    score = landing_score + fc_compat_bonus + latency_bonus + recovery_bonus
```

The **multiplicative factors** all sit on `[0, 1]`. Their floors (`time_factor
≥ 0.5`, `battery_factor ≥ 0.3`) are there so a slow but successful landing
is still meaningfully rewarded — we don't want the scoreboard to look
identical to "TIMEOUT" just because you took your time.

The **additive bonuses** (soft landing + estimation + HW-readiness) sit on
top. HW-readiness bonuses are independent — claim 0, 1, 2, or all 3.

## Score range

| Case                                                       | Score range  |
| ---------------------------------------------------------- | ------------ |
| Crash                                                      | `-20`        |
| Timeout / out of battery                                   | `0`          |
| Lands at the platform edge, full battery, slow             | `~10–60`     |
| Lands centered, fast, full battery                         | `~80–110`    |
| Centered, fast, soft, with a good Kalman estimate          | `~120–130`   |
| Theoretical max landing                                    | `100 + 10 + 20 = 130` |
| Add: FC-compatible output                                  | `+ 5`        |
| Add: act() p95 latency ≤ 20 ms                             | `+ 5`        |
| Add: scenario-`recovery` drift + reacquire                 | `+ 5`        |
| Theoretical max total                                      | `130 + 15 = 145` |

## Components in detail

### `landing_position_error` (m)

Horizontal (xy) distance between the drone center and the boat center at
the moment of touchdown. Capped: errors `≥ 1.5 m` zero out the precision
factor entirely (you landed in the corner of the platform — barely a
landing).

### `time_to_land` (s)

`env.t` at the touchdown step. The faster you land, the higher the
`time_factor` — but it's floored at `0.5`, so a 60-second landing is still
worth half a fast one.

### `battery_remaining` (0..1)

Whatever's in `obs['battery']` at termination. Battery drains:

- A constant `battery_drain_rate` per second.
- Plus `battery_drain_aggressive * min(1, |ω| / 5)` per second when the
  drone is rotating fast.

Floor at `0.3` — you don't get nothing for landing on fumes.

### `max_descent_velocity` (m/s)

The maximum downward speed observed during the episode. Below `1.0 m/s`
on touchdown earns the `soft_landing_bonus` of 10. Real drones break when
they slam into things; we reward gentle.

### `estimation_bonus` (0..20)

Computed from the 2D RMSE between your `get_last_estimate()['position']`
and the ground-truth boat position over the whole episode:

```text
rmse <= 0  m   ->  20.0
rmse >= 2  m   ->   0.0
elsewhere      ->  20 * (1 - rmse / 2)        # linear
```

Agents that don't implement `get_last_estimate()` (or return `None`) just
forfeit this bonus. There's no penalty for opting out, but on a moving
boat you'll almost certainly want a Kalman or similar — and once you have
one, exposing it is free 0–20 points.

## Worked examples

### Example 1: perfect run on EASY

```text
outcome:                LANDED
landing_position_error: 0.05 m
time_to_land:           18.0 s   (out of 60)
battery_remaining:      0.86
max_descent_velocity:   0.4 m/s
estimation_rmse:        0.10 m

precision_factor   = 1 - 0.05 / 1.5  = 0.967
time_factor        = 1 - 18 / 60     = 0.700
battery_factor     = 0.86
soft_landing_bonus = 10
estimation_bonus   = 20 * (1 - 0.10 / 2) = 19

base * factors = 100 * 0.967 * 0.700 * 0.86 = 58.2
score          = 58.2 + 10 + 19            = 87.2
```

### Example 2: scrappy MEDIUM landing

```text
outcome:                LANDED
landing_position_error: 0.95 m   (just barely on platform)
time_to_land:           45 s
battery_remaining:      0.32
max_descent_velocity:   1.4 m/s  (no soft bonus)
estimation_rmse:        0.6 m

precision_factor = 1 - 0.95 / 1.5 = 0.367
time_factor      = 1 - 45 / 60    = 0.250  -> floored at 0.5
battery_factor   = 0.32
estimation_bonus = 20 * (1 - 0.6 / 2) = 14

score = 100 * 0.367 * 0.5 * 0.32 + 0 + 14 = 5.87 + 14 = 19.87
```

### Example 3: crash

```text
outcome:                CRASHED  (descended too fast)
score:                  -20
```

## HW-readiness bonuses (deployment realism)

Three independent bonuses (5 pt each, max 15 pt total) reward agents
that would actually port to real hardware with minimal rework. All
bonuses are awarded ONLY when `outcome == "LANDED"` — a fancy interface
that doesn't land earns nothing.

### `fc_compat_bonus` — FC-compatible output (+5)

Implement `act_setpoint(obs) -> (thrust_norm, roll, pitch, yaw_rate)`
on your agent instead of (or in addition to) `act(obs) -> motor_throttles`.
The runner detects this and routes setpoints through the stock
`DefaultAttitudeController` before calling `env.step`. The same four
numbers are what PX4/Ardupilot accept in OFFBOARD mode via MAVLink
`SET_ATTITUDE_TARGET`.

Why it's worth 5 pt rather than 20: the work to implement it is small
(your existing outer/mid loop is most of the way there). The pay-off
shows up at HW porting time, which is out of scope for the challenge
itself — so we reward the architecture choice, not the volume of code.

### `latency_bonus` — p95 act() ≤ 20 ms (+5)

The runner times `agent.act` (or `agent.act_setpoint`) on every step
and reports the episode-wide p95. If it's ≤ 20 ms, you earn +5.

Measured inside the eval Docker container with `--cpus=6 --memory=8g`
(see [`DOCKER.md`](DOCKER.md)). The host machine the organizers run
on is fixed across all submissions. Use
`./docker/run-local.sh python docker/benchmark.py` to calibrate your
local timing against the eval target.

Failure modes the budget catches: re-running expensive perception
(big CNN, exhaustive ArUco params) on every step instead of caching;
allocating numpy arrays in the hot path; doing image preprocessing
that should run once.

### `recovery_bonus` — marker re-acquisition (+5, recovery scenario only)

On `scenarios/recovery.yaml`, the camera is blacked out deterministically
for 3 simulated seconds during the approach phase. To claim the bonus:

1. Horizontal drift while blacked out: ≤ `recovery_check.drift_tolerance_m`
   (2.0 m by default).
2. After the blackout ends: `get_last_estimate()['position']` agrees
   with the true boat position within
   `recovery_check.reacquire_tolerance_m` (0.30 m by default), within
   `recovery_check.reacquire_window_s` (1.0 s by default).

Both must pass. An agent that diverges during the blackout (no
prediction model) or never recovers (no detection-reset logic) fails.

This bonus is only awarded on scenarios that explicitly enable
`recovery_check.enabled: true`. Public + private eval scenarios that
do NOT enable it simply don't contribute this 5 pt slot.

## Implementation

`evaluation/scorer.py` has the canonical implementation.
`evaluation/evaluate.py` is the runner; it detects the agent mode,
profiles latency, and tracks the recovery check, then passes the
results to the scorer as `hw_readiness`. The scoring formula is
**frozen** for the duration of the event — no late-night "oh let's
reweight precision" surprises. If a clarification or bug fix becomes
necessary, it'll be communicated in writing.
