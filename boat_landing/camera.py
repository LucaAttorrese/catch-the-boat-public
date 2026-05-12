"""Camera utilities for BoatLandingEnv.

The environment renders a downward-facing RGB camera attached to the drone
with PyBullet's TINY (CPU) renderer. This module exposes:

    - Image dimensions and FOV constants.
    - Helpers to build PyBullet view/projection matrices from drone pose.
    - Image degradation primitives: noise, motion blur, fog, occlusion.

Frame rate degradation (fps drop) is handled in `env.py` because it
requires caching the previous frame across env.step() calls — the
camera module stays stateless on purpose.

Agents NEVER import from this module — they consume the (480, 640, 3)
uint8 frame from `obs['camera']`. ArUco detection lives in the agent code.
"""

from typing import List

import cv2
import numpy as np
import pybullet as p


CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FOV_DEG = 90.0
CAMERA_NEAR = 0.05
CAMERA_FAR = 100.0

# Camera mount offset along body -z (downward) from the drone's COM,
# expressed in metres. Just below the fuselage/legs (which extend to
# body_z = -0.10) so the camera's optical axis isn't occluded by them.
CAMERA_BODY_OFFSET_Z = -0.115


def get_intrinsics() -> np.ndarray:
    """Return the 3x3 pinhole intrinsic matrix for this camera.

    Derived from a vertical FOV of CAMERA_FOV_DEG. fx == fy (square pixels).
    """
    fov_rad = float(np.deg2rad(CAMERA_FOV_DEG))
    fy = CAMERA_HEIGHT / (2.0 * np.tan(fov_rad / 2.0))
    fx = fy
    cx = CAMERA_WIDTH / 2.0
    cy = CAMERA_HEIGHT / 2.0
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


def get_view_matrix(drone_pos: np.ndarray, drone_quat: np.ndarray) -> List[float]:
    """Build PyBullet's flattened view matrix for a downward camera mounted
    on the drone body.

    The camera optical axis points along -body_z (so when the drone is
    hovering level, the camera looks straight down). The image "up" is
    aligned with body +x (drone forward), so the rendered frame yaws with
    the drone.
    """
    rot = np.array(p.getMatrixFromQuaternion(drone_quat)).reshape(3, 3)
    body_x = rot[:, 0]
    body_z = rot[:, 2]
    # Camera mounted below the drone's underside.
    eye = np.asarray(drone_pos, dtype=np.float64) + body_z * CAMERA_BODY_OFFSET_Z
    target = eye - body_z  # 1 m below the camera in body frame
    up = body_x
    return p.computeViewMatrix(
        cameraEyePosition=eye.tolist(),
        cameraTargetPosition=target.tolist(),
        cameraUpVector=up.tolist(),
    )


def get_projection_matrix() -> List[float]:
    return p.computeProjectionMatrixFOV(
        fov=CAMERA_FOV_DEG,
        aspect=CAMERA_WIDTH / CAMERA_HEIGHT,
        nearVal=CAMERA_NEAR,
        farVal=CAMERA_FAR,
    )


def apply_noise(
    image: np.ndarray, noise_level: float, rng: np.random.Generator
) -> np.ndarray:
    """Add zero-mean Gaussian noise. `noise_level` is sigma in [0, 1] units
    (1.0 == sigma of full pixel range)."""
    if noise_level <= 0:
        return image
    sigma = float(noise_level) * 255.0
    noise = rng.normal(0.0, sigma, image.shape).astype(np.float32)
    noisy = image.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def apply_occlusion(image: np.ndarray, occluded: bool) -> np.ndarray:
    """If `occluded`, fill the frame with a flat water-blue color so the
    marker is hidden. Otherwise return the image unchanged."""
    if not occluded:
        return image
    out = image.copy()
    out[:] = (50, 80, 120)
    return out


# ---------------------------------------------------------------------------
# Fog / haze
# ---------------------------------------------------------------------------

# Atmospheric light color used in the fog blend. Slightly desaturated and
# bluish — the typical look of marine haze.
FOG_ATMOSPHERE_RGB = np.array([200, 210, 220], dtype=np.float32)


def apply_fog(
    image: np.ndarray,
    fog_density: float,
    altitude_m: float,
) -> np.ndarray:
    """Blend the image toward a uniform atmospheric color via Beer-Lambert.

    The visibility transmittance is `t = exp(-fog_density * altitude_m)`,
    where `fog_density` is the extinction coefficient (1/m). The output
    pixel is `t * image + (1 - t) * FOG_ATMOSPHERE_RGB`, which is the
    standard single-scattering haze model.

    `fog_density = 0` is a no-op. Typical values:
        0.05 - light haze (drone at 5 m sees ~78% of original signal)
        0.20 - dense fog  (drone at 5 m sees ~37%)
        0.50 - whiteout   (drone at 5 m sees ~8%)
    """
    if fog_density <= 0.0:
        return image
    altitude = max(float(altitude_m), 0.0)
    transmittance = float(np.exp(-fog_density * altitude))
    if transmittance >= 0.999:
        return image
    out = image.astype(np.float32) * transmittance + FOG_ATMOSPHERE_RGB * (
        1.0 - transmittance
    )
    return np.clip(out, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Motion blur (box kernel)
# ---------------------------------------------------------------------------

def apply_motion_blur(image: np.ndarray, kernel_size: int) -> np.ndarray:
    """Apply a horizontal box blur of width `kernel_size` pixels.

    A horizontal blur emulates the dominant motion blur on a downward
    camera under boat-relative drift. `kernel_size` should be an odd
    integer >= 3; values < 3 are treated as no-op.
    """
    k = int(kernel_size)
    if k < 3:
        return image
    if k % 2 == 0:
        k += 1  # cv2.blur tolerates even but odd kernels are conventional
    return cv2.blur(image, (k, 1))
