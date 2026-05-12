"""Scoring for the Catch the Boat challenge.

Score formula (frozen — see docs/SCORING.md for the rationale):

    if outcome == "CRASHED":           score = -20
    elif outcome in {"TIMEOUT",
                     "OUT_OF_BATTERY"}: score = 0
    elif outcome == "LANDED":
        base               = 100
        precision_factor   = max(0,    1 - landing_position_error / 1.5)
        time_factor        = max(0.5,  1 - time_to_land / duration_max)
        battery_factor     = max(0.3,  battery_remaining)
        soft_landing_bonus = 10  if  max_descent_velocity < 1.0  else 0
        estimation_bonus   = compute_estimation_bonus(...)        # 0..20

        score = (base * precision_factor * time_factor * battery_factor
                 + soft_landing_bonus + estimation_bonus)

HW-readiness add-ons (only awarded when outcome == LANDED):
    fc_compat_bonus    = 5  if agent.act_setpoint was used                  (deploy-friendly output)
    latency_bonus      = 5  if act() p95 latency <= LATENCY_BUDGET_MS       (HW-realistic CPU budget)
    recovery_bonus     = 5  if scenario.recovery and marker re-acquired     (only on recovery-tagged scenarios)
"""

from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np


CRASH_PENALTY = -20.0
LATENCY_BUDGET_MS = 20.0          # p95 cap for the latency bonus
FC_COMPAT_BONUS = 5.0
LATENCY_BONUS = 5.0
RECOVERY_BONUS = 5.0


def compute_estimation_bonus(rmse: Optional[float]) -> float:
    """Map a 2D-position RMSE (in metres) to a 0..20 bonus.

    rmse == 0 m -> 20.  rmse >= 2 m or None -> 0.  Linear in between.
    """
    if rmse is None or not np.isfinite(rmse):
        return 0.0
    return float(max(0.0, 20.0 * (1.0 - min(rmse, 2.0) / 2.0)))


def compute_estimation_rmse(
    log: Iterable[Tuple[np.ndarray, np.ndarray]]
) -> Optional[float]:
    """Compute the 2D RMSE between true and estimated boat positions.

    `log` is an iterable of `(true_position, estimated_position)` pairs,
    each a 3D vector. Only the (x, y) components are used.
    """
    sq_errors: List[float] = []
    for true, est in log:
        true = np.asarray(true, dtype=np.float64).reshape(-1)[:2]
        est = np.asarray(est, dtype=np.float64).reshape(-1)[:2]
        diff = true - est
        sq_errors.append(float(np.dot(diff, diff)))
    if not sq_errors:
        return None
    return float(np.sqrt(np.mean(sq_errors)))


def compute_score(
    outcome: str,
    landing_position_error: Optional[float],
    time_to_land: float,
    duration_max: float,
    battery_remaining: float,
    max_descent_velocity: float,
    estimation_rmse: Optional[float] = None,
    hw_readiness: Optional[Dict[str, Any]] = None,
) -> Tuple[float, Dict]:
    """Return (score, breakdown_dict).

    `landing_position_error` is the horizontal (xy) distance between the
    drone and the boat's center at the moment of touchdown, in metres.
    Required for outcome == "LANDED"; ignored otherwise.

    `hw_readiness` is an optional dict with deployment-readiness signals
    measured by `evaluate.py`. Recognised keys:
        - "fc_compatible"   bool   — agent emitted (thrust, roll, pitch,
                                     yaw_rate) setpoints via `act_setpoint`
                                     instead of raw motor throttles.
        - "latency_p95_ms"  float  — p95 of agent.act / act_setpoint over
                                     the episode, measured inside the eval
                                     container.
        - "recovery_passed" bool   — marker re-acquired within tolerance
                                     after the scripted occlusion window
                                     (only meaningful on recovery scenarios).
    Each bonus is awarded ONLY if outcome == "LANDED".
    """
    breakdown: Dict = {
        "outcome": outcome,
        "components": {},
    }

    if outcome == "CRASHED":
        breakdown["components"] = {"crash_penalty": CRASH_PENALTY}
        return CRASH_PENALTY, breakdown

    if outcome in ("TIMEOUT", "OUT_OF_BATTERY", "ABORTED"):
        return 0.0, breakdown

    if outcome != "LANDED":
        # Unknown / non-terminal — neutral.
        return 0.0, breakdown

    err = float(landing_position_error if landing_position_error is not None else 0.0)
    base = 100.0
    precision_factor = max(0.0, 1.0 - err / 1.5)
    time_factor = max(0.5, 1.0 - time_to_land / max(duration_max, 1e-6))
    battery_factor = max(0.3, float(battery_remaining))
    soft_bonus = 10.0 if max_descent_velocity < 1.0 else 0.0
    est_bonus = compute_estimation_bonus(estimation_rmse)

    landing_score = base * precision_factor * time_factor * battery_factor
    total = landing_score + soft_bonus + est_bonus

    components = {
        "base": base,
        "landing_position_error_m": err,
        "precision_factor": precision_factor,
        "time_to_land_s": float(time_to_land),
        "duration_max_s": float(duration_max),
        "time_factor": time_factor,
        "battery_remaining": float(battery_remaining),
        "battery_factor": battery_factor,
        "max_descent_velocity_mps": float(max_descent_velocity),
        "soft_landing_bonus": soft_bonus,
        "estimation_rmse_m": (
            None if estimation_rmse is None else float(estimation_rmse)
        ),
        "estimation_bonus": est_bonus,
        "landing_score": landing_score,
    }

    # HW-readiness bonuses: only awarded when the agent actually landed.
    # Each bonus is independent — a deploy-ready agent can claim all three.
    if hw_readiness:
        fc_compatible = bool(hw_readiness.get("fc_compatible", False))
        components["fc_compatible"] = fc_compatible
        if fc_compatible:
            total += FC_COMPAT_BONUS
            components["fc_compat_bonus"] = FC_COMPAT_BONUS

        latency_p95 = hw_readiness.get("latency_p95_ms")
        if latency_p95 is not None:
            components["latency_p95_ms"] = float(latency_p95)
            if float(latency_p95) <= LATENCY_BUDGET_MS:
                total += LATENCY_BONUS
                components["latency_bonus"] = LATENCY_BONUS

        recovery_passed = hw_readiness.get("recovery_passed")
        if recovery_passed is not None:
            components["recovery_passed"] = bool(recovery_passed)
            if recovery_passed:
                total += RECOVERY_BONUS
                components["recovery_bonus"] = RECOVERY_BONUS

    breakdown["components"] = components
    return float(total), breakdown
