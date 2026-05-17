"""Template agent — patched per Option 1: full-negative thrust in LAND.

Inherits all baseline behavior and overrides ONLY the LAND-phase thrust
command. The baseline z-PID outputs thrust_norm ~= -0.10, which gives
F_z ~= 0.95 * mg. Over the platform (ground effect zone) that gets
amplified above mg and the drone hovers at z ~= 0.65 m.

This override bypasses the z-PID in LAND and emits thrust_norm = -1.0
(50% hover thrust, the DefaultAttitudeController's lower bound). The
ground-effect cap of 2x in reference_sim/_core.py:227 means the worst
case is F_z ~= mg at h ~= 0.056 m, so the drone descends until contact.

Horizontal PID and yaw stay as in the baseline so the drone keeps
tracking the platform during the final drop.

Run with:
    python evaluation/evaluate.py --agent agents/agent_template.py \
        --use-reference-sim --scenario easy
"""

from typing import Dict, Optional

import numpy as np

from agents.agent_baseline import BaselineAgent, PHASE_LAND, rpy_to_matrix
from boat_landing.drone_interface import DroneSpec


class TemplateAgent(BaselineAgent):
    def control(
        self, drone_state: Dict, boat_estimate: Dict, phase: str
    ) -> np.ndarray:
        if phase != PHASE_LAND:
            return super().control(drone_state, boat_estimate, phase)

        pos = np.asarray(drone_state["position"], dtype=np.float64)
        target = boat_estimate.get("position")
        if target is None:
            target = np.array(
                [self.SEARCH_PRIOR_XY[0], self.SEARCH_PRIOR_XY[1], 0.3],
                dtype=np.float64,
            )

        R = rpy_to_matrix(drone_state["attitude"])
        vel_world = np.asarray(drone_state["velocity"], dtype=np.float64)
        vel_body = R.T @ vel_world
        err_world = np.array(
            [target[0] - pos[0], target[1] - pos[1], 0.0],
            dtype=np.float64,
        )
        err_body = R.T @ err_world

        pitch_cmd = self.pid_x(
            err_body[0], self.DT, measurement_velocity=vel_body[0]
        )
        roll_cmd = -self.pid_y(
            err_body[1], self.DT, measurement_velocity=vel_body[1]
        )

        return self.attitude_ctrl(
            drone_state,
            roll_des=roll_cmd,
            pitch_des=pitch_cmd,
            yaw_rate_des=0.0,
            thrust_norm=-1.0,
        )


Agent = TemplateAgent


def make_agent(drone_spec: Optional[DroneSpec] = None) -> TemplateAgent:
    return TemplateAgent(drone_spec)
