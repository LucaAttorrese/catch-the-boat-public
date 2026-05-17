"""Simulator-quality validation suite.

Each submodule (`t0_*`, `t1_*`) exposes a `run(sim_factory, drone_spec_path,
**kwargs)` function returning a `TestResult`. The Tier-0 modules are gates
(failing any → simulator score = 0). The Tier-1 modules are auto-tested
features worth 5 points each, declared by the participant in
`submission.yaml`.

The orchestrator that ties them together lives in
`evaluation/sim_scorer.py`. The runner / dataclass plumbing is in
`evaluation/sim_validation/_runner.py`.
"""

from evaluation.sim_validation._runner import (  # noqa: F401
    TestResult,
    load_sim_factory,
    make_hover_action,
)

# Mapping from submission.yaml feature key -> test module.
# Used by sim_scorer.py to look up the right test for each declared
# tier-1 feature. New tier-1 tests must be registered here AND in the
# submission.yaml schema.
TIER0_TESTS = [
    "evaluation.sim_validation.t0_protocol",
    "evaluation.sim_validation.t0_hover_steady",
    "evaluation.sim_validation.t0_determinism",
    "evaluation.sim_validation.t0_variable_dt",
]

TIER1_TESTS = {
    "motor_lag":       "evaluation.sim_validation.t1_motor_lag",
    "battery_sag":     "evaluation.sim_validation.t1_battery_sag",
    "aero_drag":       "evaluation.sim_validation.t1_aero_drag",
    "cross_coupling":  "evaluation.sim_validation.t1_cross_coupling",
    "substepping":     "evaluation.sim_validation.t1_substepping",
    "ground_effect":   "evaluation.sim_validation.t1_ground_effect",
}

POINTS_PER_TIER1_FEATURE = 5
