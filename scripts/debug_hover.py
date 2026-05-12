"""Per-step trace of the baseline showing position, velocity, attitude
and the action it commands. Use to spot the moment the controller
diverges."""

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agents.agent_baseline import BaselineAgent
from boat_landing.env import BoatLandingEnv

env = BoatLandingEnv(str(REPO_ROOT / "scenarios" / "easy.yaml"), gui=False)
agent = BaselineAgent()
obs, _ = env.reset(seed=42)

print(" step | pos                         | vel                     | rpy                     | action")
import time as _time

t_start = _time.time()
for i in range(3000):  # full 60s scenario
    action = agent.act(obs)
    obs, _, t, tr, info = env.step(action)
    if i % 50 == 0:  # every 1 sim-second
        p_ = obs["state"]["position"]
        v_ = obs["state"]["velocity"]
        wall = _time.time() - t_start
        print(
            f"sim {obs['time']:5.2f}s wall {wall:5.1f}s "
            f"pos=({p_[0]:6.2f},{p_[1]:6.2f},{p_[2]:5.2f}) "
            f"vel=({v_[0]:5.2f},{v_[1]:5.2f},{v_[2]:5.2f}) "
            f"phase={agent.phase} bat={obs['battery']*100:4.1f}%",
            flush=True,
        )
    if np.any(np.isnan(obs["state"]["position"])):
        print(f"  -> NaN at step {i}")
        break
    if t or tr:
        print(
            f"  -> terminated step={i} t={obs['time']:.2f}s outcome={info.get('outcome')}",
            flush=True,
        )
        break
env.close()
