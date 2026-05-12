"""Wind force model.

Applies an external world-frame force to the drone. Decomposed into:
    mean_force: a constant DC term (vector)
    gust:       a sinusoidal AC term scaled by gust_amplitude with random
                phase per axis (so x/y gusts aren't coupled)

This is intentionally simple. Teams can model wind better in their agent
if they want to feed-forward compensate for it.
"""

from typing import Dict

import numpy as np


class Wind:
    def __init__(self, config: Dict, rng: np.random.Generator):
        self.enabled = bool(config.get("enabled", False))
        self.mean = np.array(
            config.get("mean_force", [0.0, 0.0, 0.0]), dtype=np.float64
        )
        self.gust_amp = float(config.get("gust_amplitude", 0.0))
        self.gust_freq = float(config.get("gust_frequency", 0.0))

        # Random phase per axis so the gust pattern differs each scenario seed
        self._phase = rng.uniform(0.0, 2.0 * np.pi, size=3)
        self._t = 0.0

    def get_force(self, dt: float) -> np.ndarray:
        """Advance the wind clock and return the world-frame force (N)."""
        self._t += dt
        if not self.enabled:
            return np.zeros(3, dtype=np.float64)
        if self.gust_amp <= 0 or self.gust_freq <= 0:
            return self.mean.copy()
        omega = 2.0 * np.pi * self.gust_freq * self._t
        gust = self.gust_amp * np.array(
            [
                np.sin(omega + self._phase[0]),
                np.cos(omega + self._phase[1]),
                # Vertical gust is intentionally smaller — sea-level wind is
                # mostly horizontal.
                0.25 * np.sin(omega + self._phase[2]),
            ],
            dtype=np.float64,
        )
        return self.mean + gust
