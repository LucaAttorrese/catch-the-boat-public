"""One-command launcher for the baseline agent.

Usage:
    python scripts/run_baseline.py --scenario easy --visualize

If --visualize is set, opens a pygame window showing a chase view of the
scene, the drone's downward camera, and a HUD. Otherwise runs headless
and prints the outcome at the end.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np  # noqa: F401 — used by agent's get_last_estimate consumers

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agents.agent_baseline import BaselineAgent  # noqa: E402
from agents.drone_sim_baseline import BaselineDroneSimulator  # noqa: E402
from boat_landing.env import BoatLandingEnv  # noqa: E402
from evaluation.evaluate import (  # noqa: E402
    DEFAULT_DRONE_SIM,
    DEFAULT_DRONE_SPEC,
    resolve_drone_spec,
    resolve_scenario,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scenario", default="easy", help="Scenario name or path")
    ap.add_argument(
        "--drone",
        default=str(DEFAULT_DRONE_SPEC),
        help="Drone spec name (e.g. 'quadcopter', 'vtol') or path to YAML.",
    )
    ap.add_argument("--visualize", action="store_true", help="Open the pygame viewer")
    ap.add_argument(
        "--gui",
        action="store_true",
        help="Open the PyBullet GUI window. Enables the OpenGL (GPU) renderer "
        "for the drone camera, which is 3-5× faster than the default TINY "
        "(CPU) renderer.",
    )
    ap.add_argument(
        "--realtime",
        action="store_true",
        help="Sleep so the simulation runs at wall-clock speed (only with --visualize)",
    )
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Hard cap on env steps (default: until the scenario ends).",
    )
    args = ap.parse_args()

    scenario_path = resolve_scenario(args.scenario)
    drone_spec_path = resolve_drone_spec(args.drone)
    drone_sim = BaselineDroneSimulator(str(drone_spec_path))
    env = BoatLandingEnv(str(scenario_path), drone_sim=drone_sim, gui=args.gui)
    agent = BaselineAgent(drone_spec=drone_sim.spec)

    viewer = None
    if args.visualize:
        from visualizer.viewer import Viewer  # local import keeps headless path lean

        viewer = Viewer(env)

    obs, info = env.reset(seed=args.seed)
    last_info = info
    step = 0
    try:
        while True:
            action = agent.act(obs)
            obs, _, terminated, truncated, info = env.step(action)
            last_info = info

            if viewer is not None:
                if not viewer.poll_events():
                    print("Viewer closed; aborting episode.")
                    break
                target = (
                    agent._last_marker_world
                    if getattr(agent, "_last_marker_world", None) is not None
                    else None
                )
                pos = obs["state"]["position"]
                hud = [
                    f"scenario   : {env.scenario.get('scenario_id', '?')}",
                    f"phase      : {agent.phase}",
                    f"t          : {obs['time']:6.2f} s",
                    f"battery    : {obs['battery']*100:5.1f} %",
                    f"alt        : {pos[2]:6.2f} m",
                    f"speed      : {float(np.linalg.norm(obs['state']['velocity'])):6.2f} m/s",
                ]
                if target is not None:
                    horiz = float(np.linalg.norm(pos[:2] - target[:2]))
                    hud.append(f"horiz dist : {horiz:6.2f} m")
                viewer.render(obs["camera"], hud)

                if args.realtime:
                    time.sleep(BoatLandingEnv.DT)

            step += 1
            if terminated or truncated:
                break
            if args.max_steps is not None and step >= args.max_steps:
                break

        outcome = last_info.get("outcome") or "RUNNING"
        print(
            f"Episode end | outcome={outcome} | t={env.t:.2f}s | "
            f"battery={obs['battery']*100:.1f}% | steps={step}"
        )
        return 0 if outcome == "LANDED" else 1
    finally:
        env.close()
        if viewer is not None:
            viewer.close()


if __name__ == "__main__":
    sys.exit(main())
