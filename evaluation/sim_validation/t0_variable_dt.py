"""T0.5 — Variable timestep robustness (HIDDEN gate test).

The participant's simulator is called with `dt` values drawn from
`DT_RANGE` over a single episode, in a non-constant pattern. The same
motor-command sequence is then re-played against the SAME participant
sim but with each step broken into `base_dt`-sized integration chunks.
A sim that integrates robustly under coarse / variable `dt` produces
very similar final positions in both runs; a sim that hardcodes an
internal substep count, or relies on a fixed input `dt`, accumulates
visible error and fails.

This test is part of the Tier-0 GATE: failing it zeroes the simulator
score, regardless of how many Tier-1 features the participant ships.

Visibility policy
-----------------
T0.5 is announced publicly with its CATEGORY ("variable dt robustness")
and its `dt` RANGE ([0.001, 0.050] s). The exact `dt` pattern, the
duration, and the pass threshold are NOT published — to discourage
sims that pass auto-tests by literal pattern-matching while still
allowing teams to prepare in the right direction.

The constants below are the public defaults (used when the test runs
locally on a participant's machine). At evaluation time the organizer
overrides them via environment variables:

    T05_SEED          (default 1337)
    T05_DURATION_S    (default 1.50)
    T05_BASE_DT       (default 0.001)
    T05_THRESHOLD_M   (default 0.05)
    T05_DT_MIN        (default 0.001)
    T05_DT_MAX        (default 0.050)

Implementation strategies that pass:
    * Internal sub-stepping that splits any incoming `dt` into chunks
      of ≤ ~5 ms before integrating.
    * RK4 or other higher-order integrators that tolerate coarse `dt`.
    * Adaptive timestep with error control.

Implementation strategies that FAIL:
    * Plain forward Euler with no sub-stepping.
    * Hardcoded `internal_substeps = N` assuming a fixed input `dt`
      (motor lag with `τ ≈ 50 ms` becomes stiff at `dt = 50 ms` for
      a single Euler step).
    * Treating each `step()` call as one integration tick irrespective
      of `dt`.
"""

from __future__ import annotations

import os

import numpy as np

from evaluation.sim_validation._runner import TestResult, safe_load_spec


NAME = "T0.5 Variable-dt robustness"


def _config() -> dict:
    """Read the test parameters from the environment, falling back to
    the documented public defaults. Eval scoring overrides these."""
    return {
        "seed": int(os.environ.get("T05_SEED", "1337")),
        "duration_s": float(os.environ.get("T05_DURATION_S", "1.5")),
        "base_dt": float(os.environ.get("T05_BASE_DT", "0.001")),
        "threshold_m": float(os.environ.get("T05_THRESHOLD_M", "0.05")),
        "dt_min": float(os.environ.get("T05_DT_MIN", "0.001")),
        "dt_max": float(os.environ.get("T05_DT_MAX", "0.050")),
    }


def _build_dt_pattern(rng: np.random.Generator, cfg: dict) -> np.ndarray:
    """Sequence of `dt` values summing to ~`duration_s`. Each draw is in
    [dt_min, dt_max]. The last entry is clamped so the total matches the
    target duration exactly."""
    dts = []
    elapsed = 0.0
    while elapsed < cfg["duration_s"] - 1e-9:
        dt = float(rng.uniform(cfg["dt_min"], cfg["dt_max"]))
        if elapsed + dt > cfg["duration_s"]:
            dt = cfg["duration_s"] - elapsed
        dt = max(dt, cfg["dt_min"])  # avoid degenerate tiny final step
        dts.append(dt)
        elapsed += dt
    return np.array(dts, dtype=np.float64)


def _build_cmd_sequence(
    rng: np.random.Generator, n_steps: int, n_motors: int
) -> np.ndarray:
    """Smooth-ish throttle sequence in [0.3, 0.9] per motor. Smooth
    enough that the integrator is dominated by motor / rigid-body
    dynamics, not by command discontinuities."""
    cmds = rng.uniform(0.3, 0.9, size=(n_steps, n_motors))
    # 1-D moving average per motor (kernel 3) to remove the worst spikes.
    if n_steps >= 3:
        smoothed = cmds.copy()
        smoothed[1:-1] = (cmds[:-2] + cmds[1:-1] + cmds[2:]) / 3.0
        cmds = smoothed
    return cmds


def _simulate(sim_factory, drone_spec_path, cmds, dts, *, substep_dt):
    """Run the sim through `cmds[i]` for duration `dts[i]`. If
    `substep_dt > 0`, each (cmd, dt) is broken into ceil(dt/substep_dt)
    sub-calls so the sim sees only small `dt` values. Otherwise (the
    'variable' run) the full `dts[i]` is handed to sim.step verbatim.
    """
    sim = sim_factory(drone_spec_path)
    sim.reset(np.array([0.0, 0.0, 100.0]), np.zeros(3))
    for cmd, dt in zip(cmds, dts):
        if substep_dt > 0 and dt > substep_dt * 1.5:
            n_sub = max(1, int(np.ceil(dt / substep_dt)))
            sub_dt = dt / n_sub
            for _ in range(n_sub):
                sim.step(cmd, np.zeros(3), sub_dt)
        else:
            sim.step(cmd, np.zeros(3), dt)
    return sim.get_state()


def run(sim_factory, drone_spec_path: str, **_) -> TestResult:
    cfg = _config()
    spec = safe_load_spec(drone_spec_path)

    rng_dt = np.random.default_rng(cfg["seed"])
    rng_cmd = np.random.default_rng(cfg["seed"] + 1)

    dts = _build_dt_pattern(rng_dt, cfg)
    cmds = _build_cmd_sequence(rng_cmd, n_steps=len(dts), n_motors=spec.num_motors)

    # Reference: same inputs, but every step is sub-stepped to base_dt
    # internally by the runner. A correctly-integrating sim should also
    # converge to this reference under the variable pattern.
    state_ref = _simulate(
        sim_factory, drone_spec_path, cmds, dts, substep_dt=cfg["base_dt"]
    )
    state_var = _simulate(
        sim_factory, drone_spec_path, cmds, dts, substep_dt=0.0
    )

    pos_diff = float(np.linalg.norm(state_ref.position - state_var.position))
    vel_diff = float(np.linalg.norm(state_ref.velocity - state_var.velocity))
    metrics = {
        "pos_diff_m": pos_diff,
        "vel_diff_m_per_s": vel_diff,
        # Stash the public defaults so participants can see what they ran
        # against; the eval run uses overrides and these will differ.
        "duration_s": cfg["duration_s"],
        "n_outer_steps": len(dts),
        "dt_min_s": cfg["dt_min"],
        "dt_max_s": cfg["dt_max"],
        # Threshold is intentionally left out of metrics at eval time so
        # the breakdown JSON does not leak the private value. For local
        # runs it is still emitted (no secret to protect).
        "threshold_m": cfg["threshold_m"],
    }

    if pos_diff > cfg["threshold_m"]:
        return TestResult(
            NAME,
            False,
            f"variable-dt vs reference-dt diverged by {pos_diff*100:.2f} cm; "
            f"sim is not robust to coarse / variable dt — try internal "
            f"sub-stepping or a higher-order integrator",
            metrics,
        )
    return TestResult(
        NAME,
        True,
        f"variable-dt vs reference-dt agree within {pos_diff*1000:.2f} mm "
        f"(threshold {cfg['threshold_m']*100:.0f} mm)",
        metrics,
    )
