# Reference simulator — physics notes

This document describes, **qualitatively**, what the organizer-owned
reference simulator models. It is the simulator your agent will be
scored against during evaluation.

We deliberately do **not** publish the exact equations, coefficients,
or implementation. The binary you receive (`.so` / `.pyd`) is the
authoritative version; everything below is enough to design an agent
that handles its dynamics, but not enough to reproduce it for the
sim-track scoring (that's by design — see [Why no source?](#why-no-source)
below).

---

## Features modelled

### 1. Motor RPM dynamics

The motor responds to throttle commands with a first-order lag plus
state coupling: the effective `omega_max` depends on the instantaneous
battery voltage. Throttle → instant thrust is **not** a valid model.

What you see in flight:
- Commanded thrust changes take a few tens of milliseconds to
  materialise.
- At full throttle, motors plateau slightly below `omega_max` because
  of voltage sag (see point 2).

### 2. Battery voltage sag and state of charge

Two effects compound:
- **Instantaneous sag**: under load, terminal voltage drops by
  `R_int · I(t)`. Heavy commands produce visibly lower available
  thrust during the load.
- **State of charge**: as energy is drawn out, the nominal voltage
  monotonically decreases over the episode.

What you see in flight:
- A landing approach at the end of an episode has less headroom than
  at the start.
- Sudden full-throttle bursts (panic recovery) are less effective
  than the same burst executed smoothly with anticipation.

### 3. Aerodynamic drag

Body-axis linear + quadratic drag, with axis-dependent coefficients.
The VTOL has notably strong lateral drag (long wing chord).

What you see in flight:
- Horizontal velocity does not coast forever — there is real damping.
- A drone moving at 5 m/s feels noticeably "heavier" in lateral
  control than at hover.

### 4. Cross-axis inertial coupling (full Newton-Euler)

The body angular dynamics include the `ω × (I · ω)` precession term.
On asymmetric inertia (VTOL: `Ix ≠ Iy ≠ Izz`), commanded torques on
one axis induce drift on the others.

What you see in flight:
- VTOL yaw and roll are coupled — a pure yaw command produces a small
  roll perturbation, and vice versa.
- Holding hover with the VTOL requires active stabilisation of all
  three axes simultaneously.

### 5. Internal sub-stepping

The simulator integrates internally at ≤ 5 ms per sub-step, regardless
of the `dt` you (the env or a participant test harness) pass to
`step()`. This makes the reference robust to coarse outer dt.

What this means for you:
- You can run the reference at any cadence you like — it will not
  blow up for `dt = 50 ms`.
- The reference does NOT change behaviour as a function of `dt`,
  modulo numerical noise.

### 6. Ground effect

A Cheng-Frantz-style amplification of per-motor thrust as the drone
approaches the surface passed via `ext_ground_z`. Effective only
within ~4 propeller radii of the surface; capped so weird inputs do
not produce runaway thrust.

What you see in flight:
- The last metre of descent over the platform produces noticeably
  more thrust per RPM than free flight — descent slows unexpectedly
  if you do not anticipate.
- An agent that closes the loop on velocity (not just position)
  handles this naturally. An agent that uses thrust feed-forward
  computed at altitude undershoots and bounces.

### 7. Sensor noise on attitude readback

The body angular velocity returned by `get_state()` includes a small
bias term that drifts as a bounded random walk plus a low-amplitude
vibration coupled to motor RPM. The rigid-body dynamics themselves
are NOT perturbed — the noise lives only in the readback. This models
the gyro you would fly with on a real drone.

What this means for you:
- A controller that integrates `angular_velocity` over long horizons
  will accumulate bias.
- Estimators that fuse gyro with a complementary attitude reference
  (here: from your visual pipeline) win.

### 8. Rotor gyroscopic torque

Each spinning rotor stores angular momentum; large body rotations
generate a small gyroscopic torque on the airframe. Subtle but
visible during aggressive VTOL yaw-while-rolling manoeuvres.

---

## Features NOT modelled

The reference is **realistic**, not maximalist. Things we deliberately
left out:

- Blade flapping in fast forward flight (the VTOL is always in
  multirotor mode, never cruise).
- Hover↔cruise transition for the VTOL airframe.
- Tilt-servo dynamics.
- Lift from the VTOL's wings (treated as drag-only mass).
- Detailed BLDC electromechanical modelling (current-torque-current
  loop); the first-order lag is the macroscopic surrogate.
- Wing structural deflection.

If you implement any of these in your own `drone_sim.py`, they earn
Tier-1 points where applicable (see `docs/SIM_SCORING.md`) but they
do **not** alter the agent evaluation, because the agent is always
run against the reference, not against your sim.

---

## Numerical conventions

- **World frame**: ENU, +z up.
- **Body frame**: nose along +x, port wing along +y, top of airframe
  along +z.
- **Quaternion**: `(x, y, z, w)`, PyBullet convention.
- **All units SI**: metres, seconds, radians, Newtons, kilograms.

The simulator preserves these conventions across `step()` calls, with
no hidden frame transformations.

---

## Why no source?

The reference is shipped as a compiled binary because:

1. The simulator's behaviour is **the** target the agent is scored
   against. Hand-tuning to the exact numeric coefficients of the
   reference would be overfitting in the strictest sense — the
   agent's job is to be robust to drone behaviour that *looks like*
   the reference, not identical to it.

2. Submitting the reference's source as a participant's
   `drone_sim.py` would short-circuit the sim track (Tier-1 features
   would all pass trivially). The binary distribution + the
   similarity check in `evaluation/sim_scorer.py` together close
   that loophole.

3. The deployment target (your real VTOL) is itself a black box from
   the agent's perspective. Treating the simulator as one as well is
   honest preparation for real flight.

**You do not need to reverse-engineer the binary.** Everything an
agent author needs to design and tune is in this document, the spec
YAML, and the public CHALLENGE / SCORING docs. If you find yourself
trying to read the `.so`, you are spending time on the wrong problem.
