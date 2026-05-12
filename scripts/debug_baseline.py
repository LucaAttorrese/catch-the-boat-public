"""Debug: print per-step state + ArUco detection result on EASY.

Drops a PNG of the very first camera frame for visual inspection of the
marker rendering, and prints every 25 steps (0.5s sim time) the drone
position, the marker world-estimate (or 'NO DETECT'), and the agent's
phase. Use this when the baseline fails to land on EASY — most failures
are either marker not detected (texture orientation) or controller not
converging (gains).
"""

import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agents.agent_baseline import BaselineAgent
from boat_landing.env import BoatLandingEnv


def main():
    env = BoatLandingEnv(str(REPO_ROOT / "scenarios" / "easy.yaml"), gui=False)
    agent = BaselineAgent()
    obs, info = env.reset(seed=42)

    # Save first frame for inspection
    out = REPO_ROOT / "first_frame.png"
    cv2.imwrite(str(out), cv2.cvtColor(obs["camera"], cv2.COLOR_RGB2BGR))
    print(f"first frame -> {out}")

    # Test detection on first frame
    perception = agent.perceive(obs["camera"])
    print(f"first-frame detection: {perception.get('detected')}")
    if perception.get("detected"):
        print(f"  tvec = {perception['tvec']}")

    print(f"\n  step | t    | drone_xyz                | est_xyz                  | phase     | det")
    print("  " + "-" * 100)

    last_print = -1
    for step in range(3000):  # 60s
        action = agent.act(obs)
        obs, _, terminated, truncated, info = env.step(action)
        if step % 25 == 0:
            pos = obs["state"]["position"]
            est = agent._last_estimate
            est_pos = est["position"] if est and est.get("position") is not None else None
            est_str = (
                f"({est_pos[0]:6.2f},{est_pos[1]:6.2f},{est_pos[2]:5.2f})"
                if est_pos is not None
                else "(none)                "
            )
            fresh = "F" if est and est.get("fresh") else " "
            prior = "P" if est and est.get("from_prior") else " "
            print(
                f"  {step:4d} | {obs['time']:4.2f} | "
                f"({pos[0]:6.2f},{pos[1]:6.2f},{pos[2]:5.2f}) | "
                f"{est_str} | {agent.phase:9s} | {fresh}{prior}"
            )
        if terminated or truncated:
            print(f"\n  -> ended at t={obs['time']:.2f}s, outcome={info.get('outcome')}")
            break
    env.close()


if __name__ == "__main__":
    main()
