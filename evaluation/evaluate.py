"""CLI to score an agent on a scenario.

Usage:
    python evaluation/evaluate.py --agent agents/agent_baseline.py \
        --drone-sim agents/drone_sim_baseline.py \
        --drone drones/quadcopter.yaml \
        --scenario scenarios/easy.yaml --headless

Prints a single JSON object (the result) to stdout. With --headless the
PyBullet GUI is disabled (recommended for batch evaluation).

Both --agent and --drone-sim are dynamically loaded:
    * The agent module must expose `make_agent(drone_spec)` (preferred,
      drone_spec is optional) or an `Agent` class.
    * The drone-sim module must expose `make_drone_sim(spec_path)` or a
      `DroneSim` class taking a spec path.

Defaults: agents/agent_baseline.py + agents/drone_sim_baseline.py +
drones/quadcopter.yaml.

Optionally, the agent may expose:
    * `get_last_estimate() -> dict|None` so the scorer can compute the
      estimation-bonus RMSE.
    * `act_setpoint(obs) -> (thrust_norm, roll, pitch, yaw_rate)` instead
      of `act()` — emits FC-compatible setpoints, scoring +5 HW-readiness
      points. The runner converts them to motor throttles via the stock
      `DefaultAttitudeController` BEFORE handing them to env.step.

Latency profiling: the runner measures wall-clock time inside the agent
call only (perception + estimation + control); the attitude controller
and env are excluded. The episode-wide p95 is recorded and the scorer
awards +5 if it is ≤ 20 ms. Run inside the eval Docker container for
calibrated numbers (see docs/DOCKER.md).
"""

import argparse
import importlib.util
import inspect
import json
import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np

# Make the repo root importable when invoked as a script
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from boat_landing.controllers import DefaultAttitudeController  # noqa: E402
from boat_landing.env import BoatLandingEnv  # noqa: E402
from evaluation.scorer import (  # noqa: E402
    compute_estimation_rmse,
    compute_score,
)


SCENARIO_DIR = REPO_ROOT / "scenarios"
DRONES_DIR = REPO_ROOT / "drones"
DEFAULT_AGENT = REPO_ROOT / "agents" / "agent_baseline.py"
DEFAULT_DRONE_SIM = REPO_ROOT / "agents" / "drone_sim_baseline.py"
DEFAULT_DRONE_SPEC = DRONES_DIR / "quadcopter.yaml"


def resolve_scenario(name_or_path: str) -> Path:
    """Allow either a bare scenario name (e.g. 'easy') or a full path."""
    p = Path(name_or_path)
    if p.is_file():
        return p
    candidate = SCENARIO_DIR / f"{name_or_path}.yaml"
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(
        f"Scenario {name_or_path!r} not found (looked at {p} and {candidate})."
    )


def resolve_drone_spec(name_or_path: str) -> Path:
    """Allow either a bare drone spec name (e.g. 'quadcopter') or a full path."""
    p = Path(name_or_path)
    if p.is_file():
        return p
    candidate = DRONES_DIR / f"{name_or_path}.yaml"
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(
        f"Drone spec {name_or_path!r} not found (looked at {p} and {candidate})."
    )


def _load_module(path: Path, prefix: str):
    if not path.is_file():
        raise FileNotFoundError(f"{prefix} file not found: {path}")
    if path.suffix != ".py":
        raise ValueError(f"{prefix} must be a .py file, got {path.suffix}")
    module_name = f"_dynamic_{prefix}_{int(time.time() * 1000)}_{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_drone_sim(drone_sim_path: str, drone_spec_path: str):
    """Dynamically load a participant DroneSimulator. The module must
    expose `make_drone_sim(spec_path)` (preferred) or a `DroneSim` class
    taking the spec path."""
    module = _load_module(Path(drone_sim_path), "drone_sim")
    if hasattr(module, "make_drone_sim") and callable(module.make_drone_sim):
        return module.make_drone_sim(str(drone_spec_path))
    if hasattr(module, "DroneSim"):
        return module.DroneSim(str(drone_spec_path))
    raise AttributeError(
        f"Drone sim module {drone_sim_path} must expose "
        f"`make_drone_sim(spec_path)` or `DroneSim`."
    )


def load_agent(agent_path: str, drone_spec=None):
    """Dynamically load a participant Agent. Tries the spec-aware
    constructor first, falls back to no-arg for legacy agents."""
    module = _load_module(Path(agent_path), "agent")

    def _try_call(callable_, *args):
        try:
            return callable_(*args)
        except TypeError:
            sig = inspect.signature(callable_)
            if not sig.parameters:
                return callable_()
            raise

    if hasattr(module, "make_agent") and callable(module.make_agent):
        return _try_call(module.make_agent, drone_spec)
    if hasattr(module, "Agent"):
        return _try_call(module.Agent, drone_spec)
    raise AttributeError(
        f"Agent module {agent_path} must expose `make_agent()` or `Agent`."
    )


def _detect_agent_mode(agent) -> str:
    """Decide whether to call `agent.act_setpoint` (FC-compatible 4-DoF
    setpoint, +5 pt bonus) or the legacy `agent.act` (motor throttles).

    `act_setpoint` wins if present, even if `act` is also defined — this
    way an agent that exposes both is unambiguously scored as FC-compatible.
    Raises AttributeError if neither is exposed (so the participant gets
    a clear failure at startup rather than a confused traceback mid-episode).
    """
    if hasattr(agent, "act_setpoint") and callable(agent.act_setpoint):
        return "setpoint"
    if hasattr(agent, "act") and callable(agent.act):
        return "motor"
    raise AttributeError(
        f"Agent {type(agent).__name__} exposes neither `act(obs)` nor "
        f"`act_setpoint(obs)`. Implement at least one — see "
        f"agents/agent_template.py."
    )


def _record_recovery(
    cfg: dict,
    drone_pos: np.ndarray,
    boat_pos: np.ndarray,
    estimate: Optional[dict],
    sim_t: float,
    state: dict,
) -> dict:
    """Update the recovery-bonus tracker with this step's measurements.

    `state` is a mutable dict carrying the in-progress recovery state
    across steps. Keys:
        window: (t_start, t_end) or None
        drone_pos_at_start: 3-vector snapshot
        max_drift_m: float
        reacquire_t: float | None     time-of-reacquisition (sim sec)
        passed: bool                  final verdict
    Returns the updated state for the caller.
    """
    if not cfg or not cfg.get("enabled"):
        return state
    window = state.get("window")
    if window is None:
        return state

    t_start, t_end = window
    drift_tol = float(cfg.get("drift_tolerance_m", 2.0))
    reacq_tol = float(cfg.get("reacquire_tolerance_m", 0.30))
    reacq_window = float(cfg.get("reacquire_window_s", 1.0))

    # Snap the drone position at the moment we enter the window.
    if state.get("drone_pos_at_start") is None and sim_t >= t_start:
        state["drone_pos_at_start"] = np.asarray(drone_pos, dtype=np.float64).copy()

    # Inside the blackout: track horizontal drift from snapshot.
    if t_start <= sim_t < t_end and state.get("drone_pos_at_start") is not None:
        drift = float(
            np.linalg.norm(
                np.asarray(drone_pos[:2]) - state["drone_pos_at_start"][:2]
            )
        )
        state["max_drift_m"] = max(state.get("max_drift_m", 0.0), drift)

    # After the blackout: probe the estimate every step until it agrees
    # with GT within tolerance, or the reacquire window expires.
    if sim_t >= t_end and state.get("reacquire_t") is None:
        if estimate is not None and estimate.get("position") is not None:
            est_xy = np.asarray(estimate["position"], dtype=np.float64).reshape(-1)[:2]
            err = float(np.linalg.norm(est_xy - np.asarray(boat_pos[:2])))
            if err <= reacq_tol:
                state["reacquire_t"] = sim_t

    # Compute the verdict once both checks are decidable.
    if sim_t >= t_end + reacq_window and not state.get("finalized"):
        state["finalized"] = True
        drift_ok = state.get("max_drift_m", 0.0) <= drift_tol
        reacq_t = state.get("reacquire_t")
        reacquired = reacq_t is not None and (reacq_t - t_end) <= reacq_window
        state["passed"] = bool(drift_ok and reacquired)
        state["drift_ok"] = bool(drift_ok)
        state["reacquired"] = bool(reacquired)

    return state


def run_episode(
    env: BoatLandingEnv,
    agent,
    seed: Optional[int] = None,
    progress: bool = True,
    wall_cap_s: Optional[float] = None,
) -> dict:
    """Run one episode and return a scored result.

    `wall_cap_s`: if set, abort the episode with outcome `WALL_TIMEOUT`
    when this many wall-clock seconds have elapsed since reset. Used to
    protect the eval pipeline from agents whose `act()` takes too long
    (or runs forever). Recommended: 10 * scenario.duration_max.
    """
    obs, info = env.reset(seed=seed)
    estimation_log = []
    last_obs_battery = obs["battery"]
    last_info = info
    step_idx = 0
    next_progress_t = 1.0  # next sim-time at which to print a progress line
    wall_start = time.perf_counter()
    wall_timed_out = False

    mode = _detect_agent_mode(agent)
    attitude_ctrl: Optional[DefaultAttitudeController] = None
    if mode == "setpoint":
        # Stock attitude controller stands in for the on-board flight
        # controller. The agent ships (thrust_norm, roll, pitch, yaw_rate);
        # this turns those four numbers into per-motor throttles using the
        # active drone spec. Same code that PX4/Ardupilot OFFBOARD would run
        # on real hardware, modulo gains.
        attitude_ctrl = DefaultAttitudeController(env.drone_sim.spec)

    # Latency profiling: wall-clock of the agent call ONLY. The attitude
    # controller and env step are excluded — the FC and the world are not
    # the agent's responsibility. p95 is computed at the end.
    latencies_s: List[float] = []

    # Recovery scenario bookkeeping.
    recovery_cfg = env.scenario.get("recovery_check") or {}
    cam_cfg = env.scenario.get("camera") or {}
    occlusion_window = cam_cfg.get("occlusion_window")
    recovery_state: dict = {
        "window": (
            (float(occlusion_window[0]), float(occlusion_window[1]))
            if occlusion_window is not None
            else None
        ),
        "drone_pos_at_start": None,
        "max_drift_m": 0.0,
        "reacquire_t": None,
        "passed": False,
        "drift_ok": False,
        "reacquired": False,
        "finalized": False,
    }

    while True:
        # Wall-clock cap: protect the eval from runaway / infinite-loop
        # agents. Treated as a soft TIMEOUT (score 0, not -20), since the
        # drone hasn't actually crashed.
        if wall_cap_s is not None and (time.perf_counter() - wall_start) > wall_cap_s:
            wall_timed_out = True
            print(
                f"  wall-clock cap of {wall_cap_s:.1f}s exceeded; aborting episode.",
                file=sys.stderr,
            )
            break

        # Latency-profiled region: ONLY the agent call.
        t0 = time.perf_counter()
        if mode == "setpoint":
            setpoint = agent.act_setpoint(obs)
        else:
            action = agent.act(obs)
        latencies_s.append(time.perf_counter() - t0)

        # Convert setpoint to motor throttles via the stock FC stand-in.
        if mode == "setpoint":
            sp = np.asarray(setpoint, dtype=np.float64).reshape(-1)
            if sp.shape != (4,):
                raise ValueError(
                    f"act_setpoint must return shape (4,) — "
                    f"(thrust_norm, roll, pitch, yaw_rate); got {sp.shape}"
                )
            thrust_norm, roll, pitch, yaw_rate = sp
            action = attitude_ctrl(  # type: ignore[misc]
                obs["state"], float(roll), float(pitch), float(yaw_rate),
                float(thrust_norm),
            )

        # Optional estimation log for the scorer's RMSE bonus.
        est = None
        if hasattr(agent, "get_last_estimate"):
            try:
                est = agent.get_last_estimate()
            except Exception:
                est = None
            if est is not None and est.get("position") is not None:
                estimation_log.append(
                    (info["boat_position"].copy(), np.asarray(est["position"]))
                )

        # Recovery tracking: needs current GT + drone position + estimate.
        recovery_state = _record_recovery(
            recovery_cfg,
            drone_pos=obs["state"]["position"],
            boat_pos=info["boat_position"],
            estimate=est,
            sim_t=obs["time"],
            state=recovery_state,
        )

        obs, _, terminated, truncated, info = env.step(action)
        last_obs_battery = obs["battery"]
        last_info = info
        step_idx += 1

        if progress and obs["time"] >= next_progress_t:
            phase = getattr(agent, "phase", "?")
            pos = obs["state"]["position"]
            print(
                f"  sim {obs['time']:5.2f}s  pos=({pos[0]:6.2f},{pos[1]:6.2f},"
                f"{pos[2]:5.2f})  phase={phase:8s}  bat={obs['battery']*100:4.1f}%",
                file=sys.stderr,
                flush=True,
            )
            next_progress_t += 1.0

        if terminated or truncated:
            break

    outcome = "WALL_TIMEOUT" if wall_timed_out else (last_info.get("outcome") or "TIMEOUT")
    final_drone_pos = env._traj[-1]["drone_pos"] if env._traj else np.zeros(3)
    final_boat_pos = env._traj[-1]["boat_pos"] if env._traj else np.zeros(3)
    landing_position_error = float(
        np.linalg.norm(final_drone_pos[:2] - final_boat_pos[:2])
    )

    rmse = compute_estimation_rmse(estimation_log) if estimation_log else None

    latency_p95_ms = (
        float(np.percentile(latencies_s, 95) * 1000.0) if latencies_s else None
    )
    latency_mean_ms = (
        float(np.mean(latencies_s) * 1000.0) if latencies_s else None
    )

    # Finalize the recovery verdict using whatever we observed, even if
    # the episode terminated before the official reacquire-window cutoff.
    # An episode that ends early without re-acquiring fails the bonus
    # by default — it had its chance.
    if recovery_cfg.get("enabled") and not recovery_state.get("finalized"):
        drift_tol = float(recovery_cfg.get("drift_tolerance_m", 2.0))
        reacq_window = float(recovery_cfg.get("reacquire_window_s", 1.0))
        window = recovery_state.get("window")
        drift_ok = recovery_state.get("max_drift_m", 0.0) <= drift_tol
        reacq_t = recovery_state.get("reacquire_t")
        if window is None or reacq_t is None:
            reacquired = False
        else:
            reacquired = (reacq_t - window[1]) <= reacq_window
        recovery_state["drift_ok"] = bool(drift_ok)
        recovery_state["reacquired"] = bool(reacquired)
        recovery_state["passed"] = bool(drift_ok and reacquired)
        recovery_state["finalized"] = True

    # Build hw_readiness only for the bonuses that apply to this scenario.
    hw_readiness: dict = {
        "fc_compatible": mode == "setpoint",
        "latency_p95_ms": latency_p95_ms,
    }
    if recovery_cfg.get("enabled"):
        hw_readiness["recovery_passed"] = bool(recovery_state.get("passed", False))

    score, breakdown = compute_score(
        outcome=outcome,
        landing_position_error=(
            landing_position_error if outcome == "LANDED" else None
        ),
        time_to_land=env.t,
        duration_max=float(env.scenario["duration_max"]),
        battery_remaining=last_obs_battery,
        max_descent_velocity=last_info.get("max_descent_velocity", 0.0),
        estimation_rmse=rmse,
        hw_readiness=hw_readiness,
    )

    result: dict = {
        "score": score,
        "outcome": outcome,
        "scenario_id": env.scenario.get("scenario_id", "unknown"),
        "time_to_terminate_s": env.t,
        "battery_remaining": last_obs_battery,
        "landing_position_error_m": landing_position_error,
        "max_descent_velocity_mps": last_info.get("max_descent_velocity", 0.0),
        "estimation_rmse_m": rmse,
        "agent_mode": mode,
        "latency_p95_ms": latency_p95_ms,
        "latency_mean_ms": latency_mean_ms,
        "breakdown": breakdown,
    }
    if recovery_cfg.get("enabled"):
        result["recovery"] = {
            "passed": bool(recovery_state.get("passed", False)),
            "drift_ok": bool(recovery_state.get("drift_ok", False)),
            "reacquired": bool(recovery_state.get("reacquired", False)),
            "max_drift_m": float(recovery_state.get("max_drift_m", 0.0)),
            "reacquire_t_after_window_s": (
                None
                if recovery_state.get("reacquire_t") is None
                else float(
                    recovery_state["reacquire_t"] - recovery_state["window"][1]
                )
            ),
        }
    return result


def _error_result(env: BoatLandingEnv, outcome: str, exc: BaseException) -> dict:
    """Build a scoring result for a run that failed before LANDING.

    Used by `run_episode_safe` when the agent or env raised an
    exception. Score is forced to 0 via the scorer's soft-fail branch,
    so participants who crash mid-episode get the same treatment as
    a TIMEOUT (not the -20 of a CRASHED drone).

    The traceback is NOT included — only `type(exc).__name__` and the
    short message — to avoid leaking team file paths or sensitive
    internals into the public scoreboard.
    """
    duration_max = float(env.scenario.get("duration_max", 60.0))
    score, breakdown = compute_score(
        outcome=outcome,
        landing_position_error=None,
        time_to_land=float(env.t),
        duration_max=duration_max,
        battery_remaining=0.0,
        max_descent_velocity=0.0,
        estimation_rmse=None,
        hw_readiness=None,
    )
    return {
        "score": score,
        "outcome": outcome,
        "scenario_id": env.scenario.get("scenario_id", "unknown"),
        "time_to_terminate_s": float(env.t),
        "battery_remaining": None,
        "landing_position_error_m": None,
        "max_descent_velocity_mps": None,
        "estimation_rmse_m": None,
        "agent_mode": None,
        "latency_p95_ms": None,
        "latency_mean_ms": None,
        "error_type": type(exc).__name__,
        "error_message": str(exc)[:240],
        "breakdown": breakdown,
    }


def run_episode_safe(
    env: BoatLandingEnv,
    agent,
    seed: Optional[int] = None,
    progress: bool = True,
    wall_cap_s: Optional[float] = None,
) -> dict:
    """Exception-safe wrapper around `run_episode`.

    Catches:
        - MemoryError                  → outcome = OUT_OF_MEMORY
        - any other Exception          → outcome = ERROR
        - KeyboardInterrupt            → re-raised (organiser ctrl-c respected)

    The result is always a valid JSON-serialisable dict, so batch eval
    pipelines never die mid-run. The traceback is dropped on purpose —
    only the exception type + message are recorded.
    """
    try:
        return run_episode(env, agent, seed=seed, progress=progress, wall_cap_s=wall_cap_s)
    except KeyboardInterrupt:
        raise
    except MemoryError as exc:
        return _error_result(env, "OUT_OF_MEMORY", exc)
    except Exception as exc:  # noqa: BLE001 — intentional broad catch
        return _error_result(env, "ERROR", exc)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--agent",
        default=str(DEFAULT_AGENT),
        help="Path to a .py file exposing make_agent()/Agent. "
        f"Default: {DEFAULT_AGENT.relative_to(REPO_ROOT)}",
    )
    parser.add_argument(
        "--drone-sim",
        default=str(DEFAULT_DRONE_SIM),
        help="Path to a .py file exposing make_drone_sim(spec_path)/DroneSim. "
        f"Default: {DEFAULT_DRONE_SIM.relative_to(REPO_ROOT)}. Ignored if "
        "--use-reference-sim is also passed.",
    )
    parser.add_argument(
        "--use-reference-sim",
        action="store_true",
        help="Load the organizer reference simulator from "
        "boat_landing.reference_sim instead of --drone-sim. This is what "
        "the official agent scoring uses; pass this when you want to test "
        "your agent against the same physics it will face at evaluation.",
    )
    parser.add_argument(
        "--drone",
        default=str(DEFAULT_DRONE_SPEC),
        help="Drone spec name (e.g. 'quadcopter', 'vtol') or path to YAML. "
        f"Default: {DEFAULT_DRONE_SPEC.relative_to(REPO_ROOT)}",
    )
    parser.add_argument(
        "--scenario",
        required=True,
        help="Scenario name (e.g. 'easy') or path to a YAML file.",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--headless", action="store_true", help="Disable GUI")
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Open the PyBullet GUI window. Implies the OpenGL (GPU) "
        "renderer for the drone camera, which is 3-5× faster than the "
        "default TINY (CPU) renderer. Cannot be combined with --headless.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-second progress lines on stderr.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to write the result JSON (in addition to stdout).",
    )
    parser.add_argument(
        "--wall-cap-multiplier",
        type=float,
        default=10.0,
        help="Wall-clock cap = multiplier * scenario.duration_max. If the "
        "episode runs longer than this in wall time, abort with outcome "
        "WALL_TIMEOUT (score 0). Default 10x. Set 0 to disable.",
    )
    parser.add_argument(
        "--save-traj",
        type=str,
        default=None,
        help="Optional path to write the trajectory JSON log (e.g. 'logs/traj.json').",
    )
    args = parser.parse_args()

    scenario_path = resolve_scenario(args.scenario)
    drone_spec_path = resolve_drone_spec(args.drone)
    if args.use_reference_sim:
        # Import lazily so dev environments that haven't built the
        # reference sim wheel can still use evaluate.py for testing.
        from boat_landing.reference_sim import make_drone_sim as _make_ref_sim
        drone_sim = _make_ref_sim(str(drone_spec_path))
    else:
        drone_sim = load_drone_sim(args.drone_sim, str(drone_spec_path))
    agent = load_agent(args.agent, drone_spec=drone_sim.spec)
    if args.headless and args.gui:
        print(
            "warning: --gui and --headless are exclusive; --headless wins.",
            file=sys.stderr,
        )
    use_gui = args.gui and not args.headless
    env = BoatLandingEnv(str(scenario_path), drone_sim=drone_sim, gui=use_gui)
    # Translate the user's wall-cap multiplier into seconds. 0 disables.
    # `duration_max` is required by every well-formed scenario; missing
    # it would normally cause `env.reset()` to fail later anyway, but we
    # default to 60s here so a typo in the scenario YAML still produces
    # a scored JSON instead of a bare KeyError.
    wall_cap_s: Optional[float] = None
    if args.wall_cap_multiplier and args.wall_cap_multiplier > 0:
        scenario_duration = float(env.scenario.get("duration_max", 60.0))
        wall_cap_s = float(args.wall_cap_multiplier) * scenario_duration
    try:
        result = run_episode_safe(
            env, agent,
            seed=args.seed,
            progress=not args.quiet,
            wall_cap_s=wall_cap_s,
        )
        if args.save_traj:
            traj_path = Path(args.save_traj)
            traj_path.parent.mkdir(parents=True, exist_ok=True)
            traj = env.get_trajectory()
            with open(traj_path, "w", encoding="utf-8") as f:
                json.dump(traj, f, indent=2, default=_json_default)
            print(f"Trajectory saved to {traj_path}", file=sys.stderr)
    finally:
        env.close()

    result["agent"] = str(Path(args.agent).resolve())
    result["drone_sim"] = (
        "boat_landing.reference_sim" if args.use_reference_sim
        else str(Path(args.drone_sim).resolve())
    )
    result["drone_spec"] = str(drone_spec_path)
    result["scenario_path"] = str(scenario_path)
    payload = json.dumps(result, indent=2, default=_json_default)
    print(payload)
    if args.output:
        Path(args.output).write_text(payload, encoding="utf-8")
    return 0


def _json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    raise TypeError(f"Cannot serialize {type(obj).__name__}")


if __name__ == "__main__":
    sys.exit(main())
