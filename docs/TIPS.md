# Tips & Gotchas

Improvement directions, debug tricks, and the mistakes that the people
who built this repo made first. Skim this once at the start; come back
to the section that matches whichever stage of your pipeline is
hurting.

The challenge has two scored artifacts (agent + simulator) — tips are
split accordingly.

---

## The airframe — VTOL

You're flying a single airframe: `drones/vtol.yaml`. The defining
characteristics that will shape your agent are:

| Property | Value | Implication |
| --- | --- | --- |
| Mass | 10 kg | High control authority needed; thrust-to-weight ~2.0 |
| Inertia | asymmetric (Ixx=2.54, Iyy=3.47, Izz=5.74) kg·m² | Cross-axis coupling matters; T1.D rewards modelling it |
| Yaw authority | **~0.85 rad/s² max angular accel** | Sluggish in yaw — start aligning EARLY |
| Pitch responsiveness | Sluggish (long fuselage, Iy=3.47) | Bigger lookahead for forward acceleration |
| Roll responsiveness | Moderate (Ix=2.54) | Snappier than pitch |
| Aero drag | High along body y (1.7 m wings catch air sideways) | Drag affects lateral approach trajectories |
| Fuselage-aligned landing | Must commit to yaw setpoint **early** | The single biggest failure mode of a naive agent |

The challenge isn't "easier" or "harder" airframe — it's the only
airframe, and the rubric is calibrated to it. The control problem is
yaw-dominated: agents that don't actively yaw fail the landing
condition on every non-trivial scenario.

---

## Where the baseline breaks

Run the baseline on all three scenarios so you know your floor:

```bash
python evaluation/evaluate.py --scenario easy   --headless --seed 42
python evaluation/evaluate.py --scenario medium --headless --seed 42
python evaluation/evaluate.py --scenario hard   --headless --seed 42
```

Empirical results on a clean install (seed 42, default VTOL):

| Scenario | Outcome | Why |
| -------- | ------- | --- |
| EASY     | LANDED ~69/70 (+15 latency bonus, no soft-landing bonus) | Stationary boat, no wind, no oscillation, fps=50, fog=0. Baseline lands in ~8.5 s but its bang-bang LAND descent (thrust pinned to -1 to punch through ground effect) saturates descent velocity → no soft-landing bonus. |
| MEDIUM   | TIMEOUT | Boat moves at 1.5 m/s + 20 fps camera + 0.05 fog density + 25° yaw tolerance. The baseline doesn't predict boat motion (lags behind) AND doesn't yaw-align (fails landing condition). |
| HARD     | TIMEOUT/CRASHED | Curved trajectory + 10 fps + 0.15 fog + 5 px motion blur + 20° yaw tol + 5 %/s occlusions. Baseline can't track a moving target through fog with stale frames. |

The shortest path to better numbers is listed below in priority order.
Doing **#1 (Kalman) + #2 (yaw alignment)** alone typically pulls
MEDIUM into the LANDED column.

---

## General strategy

- **Read `agents/agent_baseline.py` end to end before touching
  anything.** The state machine and PIDs are simple; the only delicate
  piece is the camera-frame → world-frame conversion in `estimate()`.
  Make sure you understand why it works before changing it.
- **Read `agents/drone_sim_baseline.py` end to end too.** It's <250
  LOC. It implements Tier 0 + motor lag + Newton-Euler with the
  ω×(I·ω) coupling term. Decide upfront which Tier 1 features you'll
  add and why.
- **Don't optimize the controller before fixing the estimator.** If
  your estimate jitters by 0.5 m, no PID tune will save you.
- **Iterate on EASY first, MEDIUM second, HARD last.** If your changes
  break EASY you've broken something foundational.
- **Run with `--seed` fixed during development.** Otherwise you'll
  chase flaky regressions caused by RNG.
- **Run the sim scorer locally before submitting.** It's the same
  scorer the organizers use:
  ```bash
  python evaluation/sim_scorer.py \
      --drone-sim teams/myteam/drone_sim.py \
      --drone     vtol \
      --submission teams/myteam/submission.yaml
  ```

---

# Agent improvements

## Improvement #1 — Estimation (BIGGEST WIN)

The baseline does *no* filtering: every detection is the new estimate.
Random ArUco corner jitter therefore propagates straight into the
controller, which then bang-bangs its commands. With fps drops and fog
on MEDIUM/HARD, this gets much worse — the controller is fed stale or
hazy detections.

- **Kalman filter on `(x, y, vx, vy)`** with a constant-velocity
  process model and the solvePnP output as the measurement. This
  single change typically pulls MEDIUM from TIMEOUT to LANDED with no
  other modifications. Set R (measurement noise) larger when the
  marker is far / fog is dense / framerate is low.
- **A 6D filter** that adds `(z, vz)` is overkill for a flat platform
  but ~30 minutes more work if the 4D filter is already in.
- **Add boat heading and yaw rate to the state** — necessary if you
  want to predict the curved trajectory in HARD. Estimate them from
  successive position measurements (atan2 of velocity) or from the
  marker's rvec. **You'll need this anyway for yaw alignment** (see
  Improvement #2).
- **Predict-only during occlusions and fps gaps.** When the marker
  disappears (occlusion) or the env returns a stale frame (fps drop),
  run the Kalman predict step (no update) instead of reverting to
  search mode. The boat doesn't teleport.
- **Outlier rejection.** Drop detections that disagree with the
  prediction by more than ~3σ. Otherwise a single bad frame knocks the
  filter sideways and the controller follows.

## Improvement #2 — Active yaw alignment

The landing condition requires the drone's fuselage axis to be aligned
with the boat's heading axis (modulo π) within tolerance (35° on easy,
25° on medium, 20° on hard). The baseline commands `yaw_rate_cmd = 0`
always — it crashes the alignment check on every non-trivial scenario.

- **Estimate boat heading.** From two consecutive position estimates,
  `boat_heading ≈ atan2(Δy, Δx)`. From the marker's rvec, you can also
  recover boat yaw directly (more accurate when the boat is moving
  slowly). Combine both; the rvec is noisier but instantaneous.
- **Pick the closer alignment target** between `boat_heading` and
  `boat_heading + π`. Both satisfy the landing condition (the
  fuselage-alignment check is mod π). You don't have to commit to a
  specific drone-forward direction.
- **Set yaw setpoint EARLY.** On the VTOL, achieving 20° yaw alignment
  from a 90° error takes ~3 seconds because of the slow yaw authority
  (~0.85 rad/s² max angular accel). Start aligning during APPROACH,
  not LAND.
- **Yaw rate command, not yaw angle command.** The
  `DefaultAttitudeController` uses a P-loop on yaw rate (no integral),
  so command `yaw_rate_des = K_yaw * (yaw_target - drone_yaw)` and
  saturate at ±1. Don't try to feed it a yaw-angle setpoint.
- **Don't yaw during DESCEND/LAND if you're already aligned** — the
  yaw torque costs battery and a slight residual ω_z is fine within
  tolerance.

## Improvement #3 — Velocity feed-forward in control

The baseline commands a target *position* and lets the PID error
chase. Against a moving boat this is wrong by construction: the
controller is always trying to move toward where the boat *was* a
frame ago.

- **Feed-forward**: command `v_drone_des = v_boat + Kp_pos * (target -
  pos)`. The drone matches the boat's velocity (so it stays alongside)
  and uses Kp only to close the residual gap.
- Pair this with the Kalman filter from #1 — without a clean velocity
  estimate, feed-forward injects noise straight into the command.
- **Wind feed-forward** is a smaller second-order win. When hovering
  steady, the drone's pitch/roll encodes the wind force. Estimate the
  steady-state tilt offset and cancel it explicitly.

## Improvement #4 — Roll-phase-aware LAND

The platform pitches/rolls at 0.4–0.6 Hz on MEDIUM/HARD. The baseline
commits to LAND whenever the geometric thresholds are satisfied, with
no awareness of *when* in the oscillation cycle it's committing. On
HARD (roll amplitude 0.15 rad ≈ 8.6°) this means a meaningful chance
of touching down with the deck tilted maximally — the drone slides
off-platform.

- **Track the marker's tilt** over time (from rvec or from successive
  estimate positions in 3D). Estimate phase and amplitude with a
  one-line lock-in or even just sign-of-derivative.
- **Wait for the zero crossing** (deck level) before transitioning
  DESCEND → LAND. A few hundred ms of patience typically saves the
  landing.
- **Add an ABORT phase.** If during DESCEND you find yourself drifting
  off-platform (estimated horizontal error > 0.4 m and still growing),
  climb 2 m and re-approach. The baseline is committed once it starts
  descending — that's wrong.

## Improvement #5 — Smarter perception against fps drop + fog

ArUco is fast and correct when it works, and gives nothing when it
doesn't. The baseline detects, runs IPPE_SQUARE solvePnP, and uses
the result raw. With degraded frames there's free precision waiting:

- **Subpixel corner refinement.** Either set
  `DetectorParameters.cornerRefinementMethod = CORNER_REFINE_SUBPIX`
  on the `ArucoDetector`, or run `cv2.cornerSubPix` manually on the
  detected corners before solvePnP. Visible improvement at no
  detection-rate cost.
- **CLAHE on the grayscale image** (`cv2.createCLAHE(...).apply(gray)`)
  is **the** trick for fog-heavy frames. Adaptive histogram
  equalization restores marker contrast that Beer-Lambert washing
  destroyed. Cheap, high-impact on HARD.
- **Defog (dark-channel prior)** is a step beyond CLAHE — cheap
  enough to run at 50 Hz and recovers more colour information.
  Overkill if CLAHE is enough.
- **Deconvolve motion blur** if the kernel size is known (it's in the
  scenario YAML which you can't read, but you CAN estimate it from the
  blur signature in detected edges). Wiener filter or Lucy-Richardson
  with 5–10 iterations is enough.
- **Detect that the frame is stale** — when fps < 50, you'll see
  identical consecutive frames. Hash a small region of the image and
  detect repeats; treat duplicates as "no new measurement" rather than
  feeding them to your filter twice.
- **Confidence score.** Detection size in pixels, solvePnP
  reprojection error, and consistency with the previous frame are all
  useful signals — pass them downstream so your filter can weight
  them.
- **A small CNN** trained on synthetic frames (you can render your own
  dataset by spinning up the env and saving `obs['camera']` to disk)
  can pick up the platform when ArUco fails on motion-blurred /
  partially-occluded frames. Eval machines are CPU — keep it tiny (a
  few-layer U-Net or HRNet, not ResNet-50).

---

# Drone simulator improvements

The simulator score is up to 30 from auto-tested Tier 1 features. The
baseline ships with two of them (motor lag, cross-coupling). Here's
what to add and the order I'd add them in.

## Recommended ROI ranking

| Feature | Effort | Risk | Why |
| --- | --- | --- | --- |
| **T1.C aero drag** | LOW (15 min) | LOW | Just multiply velocity by `spec.aero.drag_linear` and subtract from F_world. The threshold is "2% slower than vacuum" — very forgiving. |
| **T1.F ground effect** | LOW (30 min) | LOW | One formula, one altitude check. `T_eff = T · (1 + a·(R/h)²)` for `h < 4R` is enough to pass. The env passes you `ext_ground_z` already. |
| **T1.E sub-stepping** | LOW (30 min) | LOW | Detect dt at the start of `step()`, internally loop with smaller dt (e.g. dt/4). Keep the rest of the math identical. |
| **T1.B battery sag** | MEDIUM (1 h) | LOW | Track current draw per motor (`I = K_t · ω` or proportional), accumulate, drop voltage by `R_int · I_total`, scale `omega_max ← V/Kv`. Pass criterion is "5% sag after 20 s full throttle" — generous. |
| **T1.A motor lag** | (already in baseline) | — | Don't break it. |
| **T1.D cross-coupling** | (already in baseline) | — | Same. |

Doing the first three (aero, ground effect, sub-stepping) on top of
the baseline gets you to **25/30** in maybe 2 hours of work. Adding
battery sag pushes to 30/30.

## Common simulator pitfalls

- **Don't add gravity to `ext_force_world`.** The env never does. You
  do, internally, knowing `spec.mass`. Symptom: drone falls at
  `2g` because both you and someone else (you think) added gravity.
- **Don't trust `ext_ground_z` blindly.** Clamp `h = max(z - ground_z, eps)`
  to avoid `1/0` when the drone's exactly on the surface.
- **Spin motors up to hover at `reset()`.** If you start them at
  ω=0, the drone falls for the first ~0.3 s of every episode while
  motors spool up. T0.2 will fail because of the transient drift.
- **Quaternion drift.** If you integrate `q` without renormalizing,
  it drifts off the unit sphere over a long run. The baseline uses
  axis-angle rotation that's exact for constant ω over a substep.
- **Inertia tensor convention.** `spec.inertia` is `(3, 3)` already
  (we wrap diagonal Ixx/Iyy/Izz into a matrix at load time). Use
  `np.linalg.inv(spec.inertia) @ τ`, not `τ / Ixx`.
- **Motor placement signs.** `spec.motors[i].position` is body-frame.
  The thrust force is `T_i * spec.motors[i].thrust_axis`. The thrust
  moment is `r × F`, the drag moment on the body is `-spin_i · Q_i ·
  thrust_axis`. Get either sign wrong and you'll spin uncontrollably.

## Anti-bluff

The sim_scorer flags `claimed=true, passed=false` features in its
breakdown. Don't declare a feature you haven't implemented — it costs
you nothing to be honest (the baseline does this), and the diagnostic
message is public.

---

## Geometry cheat sheet

You don't need a CAD drawing, but a few numbers help.

### VTOL spec (10 kg) — the only airframe

| Thing                              | Size            |
| ---------------------------------- | --------------- |
| Drone mass                         | 10 kg           |
| Drone collision box (fuselage)     | 0.50 × 0.30 × 0.10 m |
| Wingspan (visual + drag only)      | 1.7 m           |
| Inertia (Ixx, Iyy, Izz)            | (2.54, 3.47, 5.74) kg·m² |
| Rotor diameter                     | 0.356 m (14")   |
| Per-motor max thrust               | ~47 N (T/W ≈ 1.9) |
| Yaw authority                      | **~0.85 rad/s² max angular accel** |

### Boat / platform / camera

| Thing                                 | Size                  |
| ------------------------------------- | --------------------- |
| Boat hull                             | 6.0 m × 1.5 m × 0.4 m |
| Landing platform (raised, on hull)    | 1.0 m × 1.0 m × 0.10 m |
| ArUco marker bit pattern              | 0.8 m × 0.8 m         |
| Camera FOV                            | 90° vertical, 480×640 |
| Camera body offset                    | (0, 0, -0.115) m      |
| Control rate                          | 50 Hz (DT = 0.02 s)   |
| Physics substep rate                  | 250 Hz (PHYSICS_DT = 0.004 s) |

The drone **fits comfortably** on the platform in both axes (collision
box << platform footprint). Fuselage alignment is the binding
constraint, not lateral position.

---

## Common gotchas

- **The marker is 0.8 m × 0.8 m.** That's the *physical* edge length
  you must pass to `cv2.solvePnP` as object_points. The 1.0 m plate
  size includes a 0.1 m white quiet zone around the marker — don't
  accidentally use that.
- **`obs['camera']` is RGB.** ArUco wants grayscale (`cv2.cvtColor(...,
  COLOR_RGB2GRAY)`). The baseline does this; if you switch detectors,
  don't forget.
- **The camera is mounted *under* the drone**, not at its centre of
  mass — `boat_landing/camera.py:CAMERA_BODY_OFFSET_Z` (= -0.115 m
  along body z). solvePnP returns `tvec` relative to the *camera*; if
  you forget to add this offset when transforming to world, your
  estimate is off by ~10 cm in altitude.
- **Coordinate frames.** OpenCV's camera frame is X-right, Y-down,
  Z-into-scene. PyBullet's view matrix uses OpenGL conventions. The
  baseline's `R_cam_to_world` derivation is at the top of
  `agents/agent_baseline.py:estimate()` — copy it or rederive, but
  don't guess.
- **Action shape is per-motor throttle.** `act()` returns shape
  `(drone_spec.num_motors,)` in `[0, 1]`, NOT `(4,)` in `[-1, 1]`.
  The legacy 4-DoF (thrust/roll/pitch/yaw_rate) interface still
  exists via `DefaultAttitudeController` if you don't want to write
  a controller — use that and emit motor throttles.
- **Fuselage-aligned landing.** A platform contact with yaw error
  > tolerance is classified as CRASHED (not LANDED). The baseline
  loses points on this; you won't notice unless you check the
  termination outcome.
- **`info` is off-limits.** It contains ground truth. Reading it from
  your agent is disqualifying. The scorer uses it; you don't.
- **Scenarios randomize wind and gust phases per seed.** Don't
  hardcode the drone's hover tilt — estimate it from steady-state.
- **The visualizer is *not* the env.** `viewer.render()` is for
  demos. Don't make decisions based on what you see in the chase-cam.
- **Phase target altitudes vs trigger thresholds.** If you change the
  state machine, remember every phase's target altitude must be
  *strictly below* the trigger threshold for the next phase, otherwise
  the drone settles at equilibrium and never transitions. The baseline
  has a comment about this in `PHASE_ALTITUDE`.
- **PID derivative kick.** If you swap in a different PID class,
  prefer "D on measurement" (uses drone velocity) over "D on error"
  (uses d(setpoint - pos)/dt). The baseline learned this the hard
  way: with D-on-error the controller bang-bangs ±1 every time the
  estimate snaps to a fresh detection, and the simulation diverges
  to NaN within a few seconds.
- **Stale camera frames look identical.** When `camera.fps < 50`, the
  env returns the previous frame in between renders. Your perception
  pipeline can choose to skip the work; your estimator should NOT
  treat the duplicate as a fresh measurement (otherwise the
  measurement noise looks deflated to your Kalman filter).

---

## Debugging

- `python scripts/run_baseline.py --scenario easy --visualize --gui` is
  the fastest "is anything alive" check. The drone-cam panel shows
  what your agent's perceive() sees; if the marker isn't there,
  perception can't help.
- The three `scripts/debug_*.py` files (debug_baseline, debug_hover,
  debug_marker_render) are deliberately scrappy — copy and modify
  them freely.
- `print(info)` in a local debug script confirms the ground truth.
  Just don't commit those prints into your submission.
- `pytest tests/` should be green throughout development. The
  `test_sim_validation.py` meta-tests verify the validation suite
  itself works — useful when iterating on your sim because every
  Tier 1 test you might pass/fail is a single pytest invocation away.

---

## When to stop

Once your MEDIUM agent score crosses ~45 and HARD crosses ~20, AND
your simulator score is at ≥20/30, you're in prize territory.
Further improvements have steeply diminishing returns — spend any
remaining time on robustness (different seeds, different start
positions) rather than chasing a few more points on the public
scenarios. The hidden eval scenarios reward generalisation, not
parameter-hunting.
