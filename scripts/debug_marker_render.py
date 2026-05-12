"""Diagnostic: how does PyBullet render the marker at varying altitudes?

For each altitude in [1, 2, 4, 8] m directly above the boat, save the
rendered camera frame and run the ArUco detector. Reports detection
status, detected corner pixels, and writes PNGs to repo root.
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import pybullet as p

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from boat_landing.env import BoatLandingEnv

env = BoatLandingEnv(str(REPO_ROOT / "scenarios" / "easy.yaml"), gui=False)
env.reset(seed=42)

dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
try:
    detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
    detect = lambda img: detector.detectMarkers(img)
except AttributeError:
    detect = lambda img: cv2.aruco.detectMarkers(img, dictionary)

for alt in (1.0, 2.0, 4.0, 8.0):
    # Move drone directly above the boat marker, attitude level
    p.resetBasePositionAndOrientation(
        env.drone_id,
        [0.0, 0.0, alt],
        p.getQuaternionFromEuler([0, 0, 0]),
        physicsClientId=env.client,
    )
    p.resetBaseVelocity(env.drone_id, [0, 0, 0], [0, 0, 0], physicsClientId=env.client)
    obs = env._get_observation()
    img = obs["camera"]
    out = REPO_ROOT / f"render_alt_{int(alt)}m.png"
    cv2.imwrite(str(out), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    corners, ids, _ = detect(gray)
    if ids is not None and len(ids) > 0:
        c = corners[0].reshape(4, 2)
        side_px = float(np.linalg.norm(c[0] - c[1]))
        print(
            f"alt={alt:4.1f}m  DETECTED  ids={ids.flatten().tolist()}  "
            f"marker_side_px={side_px:.1f}  png={out.name}"
        )
    else:
        # Estimate apparent marker size from scene knowledge
        focal_y = 480 / (2 * np.tan(np.deg2rad(90) / 2))
        marker_dist = alt - 0.31  # marker plate top ~0.31m
        expected_px = 0.8 * focal_y / max(marker_dist, 0.1)
        print(
            f"alt={alt:4.1f}m  no detection.  expected ~{expected_px:.0f} px  png={out.name}"
        )

env.close()
