"""Scripted boat dynamics.

The boat is NOT physics-driven. It is a kinematic platform whose pose is
recomputed every environment step from a deterministic (or pseudo-random,
with a fixed seed) trajectory function. The environment then resets the
PyBullet body's position/orientation each step.

Supported trajectory types (set via scenario YAML):

    static       -- stationary boat
    linear       -- constant speed in a fixed heading
    curve        -- constant-speed arc of a given radius/direction
    random_walk  -- heading drifts under bounded Gaussian noise

On top of that, an oscillation block can apply roll/pitch sine waves to
simulate sea state. Yaw is always the heading direction, so the boat
"points where it's going".
"""

from typing import Dict, Tuple

import numpy as np


class Boat:
    """Kinematic boat platform.

    Pose convention: `position` is the geometric center of the boat hull
    (a 4 m x 2 m x 0.3 m box). The deck top sits at `position.z + 0.15`
    when the boat is level. `heading` is the world-frame yaw (rad).
    """

    def __init__(self, config: Dict, rng: np.random.Generator):
        self.config = dict(config)
        self.rng = rng
        self.trajectory_type = str(config.get("trajectory_type", "static"))

        start = config.get("start_position", [0.0, 0.0, 0.15])
        self.position = np.array(start, dtype=np.float64)
        self.heading = float(config.get("heading_initial", 0.0))
        self.speed = float(config.get("speed", 0.0))

        self._osc = config.get("oscillation", {}) or {}
        self.roll = 0.0
        self.pitch = 0.0
        self._t = 0.0

        # Random-walk state — heading drifts via Ornstein-Uhlenbeck-like noise
        self._heading_drift = 0.0

        # For curve, derive angular velocity once so it's stable
        self._curve_radius = float(config.get("curve_radius", 20.0))
        self._curve_direction = int(config.get("curve_direction", 1))

    def step(self, dt: float) -> None:
        """Advance the boat by one environment step (`dt` seconds)."""
        self._t += dt

        if self.trajectory_type == "static":
            pass

        elif self.trajectory_type == "linear":
            self._advance_along_heading(dt)

        elif self.trajectory_type == "curve":
            # Constant-speed circular arc: dheading/dt = sign * speed / radius
            if self._curve_radius > 1e-6 and self.speed > 1e-6:
                self.heading += self._curve_direction * self.speed / self._curve_radius * dt
            self._advance_along_heading(dt)

        elif self.trajectory_type == "random_walk":
            # Slow heading drift, bounded magnitude
            self._heading_drift += self.rng.normal(0.0, 0.05) * dt
            self._heading_drift = float(np.clip(self._heading_drift, -0.5, 0.5))
            self.heading += self._heading_drift * dt
            self._advance_along_heading(dt)

        else:
            raise ValueError(f"Unknown trajectory_type: {self.trajectory_type!r}")

        # Oscillation (roll/pitch sine waves to simulate sea state)
        freq = float(self._osc.get("frequency", 0.0))
        if freq > 0.0:
            roll_amp = float(self._osc.get("roll_amplitude", 0.0))
            pitch_amp = float(self._osc.get("pitch_amplitude", 0.0))
            phase = 2.0 * np.pi * freq * self._t
            # Quadrature so the roll and pitch peaks are not simultaneous
            self.roll = roll_amp * np.sin(phase)
            self.pitch = pitch_amp * np.sin(phase + np.pi / 2.0)
        else:
            self.roll = 0.0
            self.pitch = 0.0

    def _advance_along_heading(self, dt: float) -> None:
        if self.speed <= 0:
            return
        self.position[0] += self.speed * np.cos(self.heading) * dt
        self.position[1] += self.speed * np.sin(self.heading) * dt

    def get_pose(self) -> Tuple[np.ndarray, list]:
        """Return (position, [roll, pitch, yaw])."""
        return self.position, [self.roll, self.pitch, self.heading]

    def get_velocity(self) -> np.ndarray:
        """Return world-frame linear velocity (m/s).

        Z-component is always 0 for the kinematic motion model. Oscillation
        produces angular motion only, not linear.
        """
        if self.trajectory_type == "static" or self.speed <= 0:
            return np.zeros(3, dtype=np.float64)
        return np.array(
            [
                self.speed * np.cos(self.heading),
                self.speed * np.sin(self.heading),
                0.0,
            ],
            dtype=np.float64,
        )
