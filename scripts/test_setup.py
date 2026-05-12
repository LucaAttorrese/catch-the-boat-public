"""Setup verification.

Run this once after `pip install -r requirements.txt`. It checks that
every dependency the simulator and baseline rely on is importable and
working, and that a 1-second rollout completes without errors.

Exits 0 with "All systems go!" if everything passes, or 1 with a list of
failures otherwise.
"""

import platform
import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


CHECKS = []


def check(name):
    def deco(fn):
        CHECKS.append((name, fn))
        return fn
    return deco


@check("Python >= 3.10")
def _check_python():
    if sys.version_info < (3, 10):
        return False, (
            f"Python {sys.version_info.major}.{sys.version_info.minor} too old; "
            f"need >= 3.10."
        )
    return True, f"Python {platform.python_version()}"


@check("numpy / scipy")
def _check_numpy():
    import numpy as np
    import scipy

    return True, f"numpy {np.__version__}, scipy {scipy.__version__}"


@check("PyBullet (DIRECT connect)")
def _check_pybullet():
    import pybullet as p

    cid = p.connect(p.DIRECT)
    p.disconnect(cid)
    return True, f"pybullet {p.getAPIVersion()} OK"


@check("OpenCV with ArUco")
def _check_aruco():
    import cv2

    if not hasattr(cv2, "aruco"):
        return (
            False,
            "cv2.aruco missing — install opencv-contrib-python (not opencv-python).",
        )
    try:
        cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
    except AttributeError:
        cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_100)
    return True, f"OpenCV {cv2.__version__} with ArUco DICT_5X5_100"


@check("PyYAML / pygame")
def _check_misc():
    import yaml
    import pygame

    return True, f"yaml {yaml.__version__}, pygame {pygame.version.ver}"


@check("Repository imports")
def _check_local():
    from agents.agent_baseline import BaselineAgent  # noqa: F401
    from boat_landing.env import BoatLandingEnv  # noqa: F401
    from evaluation.scorer import compute_score  # noqa: F401

    return True, "boat_landing, agents, evaluation all importable"


@check("ArUco detector instantiates")
def _check_detector():
    from agents.agent_baseline import BaselineAgent

    agent = BaselineAgent()
    if agent._detector is None and not hasattr(__import__("cv2").aruco, "detectMarkers"):
        return False, "Neither ArucoDetector nor detectMarkers is available."
    return True, "ArUco detection path OK"


@check("1-second baseline rollout on EASY")
def _check_run():
    from agents.agent_baseline import BaselineAgent
    from boat_landing.env import BoatLandingEnv

    scenario = REPO_ROOT / "scenarios" / "easy.yaml"
    env = BoatLandingEnv(str(scenario), gui=False)
    agent = BaselineAgent()
    try:
        obs, _ = env.reset(seed=42)
        steps = 0
        for _ in range(int(1.0 / BoatLandingEnv.DT)):  # 1 sim-second
            action = agent.act(obs)
            obs, _, terminated, truncated, _ = env.step(action)
            steps += 1
            if terminated or truncated:
                break
        return True, f"{steps} steps OK; sim-time {env.t:.2f}s"
    finally:
        env.close()


def main() -> int:
    print("Catch the Boat — setup checker")
    print("=" * 60)
    failed = 0
    for name, fn in CHECKS:
        try:
            ok, msg = fn()
        except Exception as exc:  # pragma: no cover — diagnostic only
            ok = False
            msg = f"{type(exc).__name__}: {exc}"
            traceback.print_exc()
        flag = "OK  " if ok else "FAIL"
        print(f"  [{flag}] {name}: {msg}")
        if not ok:
            failed += 1
    print("=" * 60)
    if failed == 0:
        print("All systems go!")
        return 0
    print(f"{failed} check(s) failed. Fix the messages above before continuing.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
