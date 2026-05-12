"""Generate the ArUco marker texture used by the simulator.

Run once after install (or after deleting boat_landing/assets/aruco/):

    python scripts/generate_aruco.py

Output: boat_landing/assets/aruco/marker_<id>.png

`BoatLandingEnv` will also auto-generate the marker on reset() if it is
missing — running this script directly is provided so participants can
pre-generate or print the marker.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = REPO_ROOT / "boat_landing" / "assets" / "aruco"


def make_marker(marker_id: int, pixel_size: int, quiet_zone_px: int = 80) -> np.ndarray:
    """Return a uint8 grayscale image of a DICT_5X5_100 marker with a
    white quiet zone padded around it for reliable detection."""
    try:
        # OpenCV 4.7+ API
        adict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
        inner = cv2.aruco.generateImageMarker(adict, marker_id, pixel_size)
    except AttributeError:
        # Older OpenCV API
        adict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_100)
        inner = cv2.aruco.drawMarker(adict, marker_id, pixel_size)

    pad = quiet_zone_px
    padded = np.full(
        (inner.shape[0] + 2 * pad, inner.shape[1] + 2 * pad), 255, dtype=np.uint8
    )
    padded[pad : pad + inner.shape[0], pad : pad + inner.shape[1]] = inner
    return padded


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--id", type=int, default=0, help="Marker ID (0..99 for 5x5_100)")
    parser.add_argument("--size", type=int, default=600, help="Pixel size of inner marker")
    parser.add_argument(
        "--out", type=str, default=str(DEFAULT_OUT), help="Output directory"
    )
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    img = make_marker(args.id, args.size)
    out_path = out_dir / f"marker_{args.id}.png"
    cv2.imwrite(str(out_path), img)
    print(f"Wrote {out_path} ({img.shape[1]}x{img.shape[0]} px)")


if __name__ == "__main__":
    main()
