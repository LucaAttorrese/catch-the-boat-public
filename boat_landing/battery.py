"""Battery drain model.

Charge runs from 1.0 (full) to 0.0 (depleted). Two contributions:

    base drain:        constant rate (`battery_drain_rate`, units: 1/sec)
    aggressive drain:  proportional to body angular-velocity magnitude,
                       scaled by `battery_drain_aggressive`

The intent is to penalize agents that thrash the attitude controller —
realistic-ish, since high angular rates correlate with high thrust demands.
"""

from typing import Dict

import numpy as np


class Battery:
    # Reference angular velocity (rad/s) at which the aggressive term hits
    # its full coefficient. 5 rad/s is "spinning hard"; below that the
    # contribution scales linearly down to 0.
    AGGRESSIVE_OMEGA_REF = 5.0

    def __init__(self, scenario_or_cfg: Dict):
        # The constructor accepts either the full scenario dict or just the
        # battery sub-block. Callers in env.py pass the whole scenario.
        cfg = scenario_or_cfg
        self.charge = float(cfg.get("battery_initial", 1.0))
        self.drain_rate = float(cfg.get("battery_drain_rate", 0.008))
        self.aggressive_rate = float(cfg.get("battery_drain_aggressive", 0.02))

    def step(self, dt: float, angular_velocity_world) -> None:
        omega_norm = float(np.linalg.norm(angular_velocity_world))
        scale = min(1.0, omega_norm / self.AGGRESSIVE_OMEGA_REF)
        rate = self.drain_rate + self.aggressive_rate * scale
        self.charge = max(0.0, self.charge - rate * dt)

    @property
    def depleted(self) -> bool:
        return self.charge <= 0.0
