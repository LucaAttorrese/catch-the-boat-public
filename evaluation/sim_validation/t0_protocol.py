"""T0.1 — DroneSimulator protocol contract.

Verifies the participant simulator implements the contract documented in
`boat_landing/drone_interface.py`:
    * exposes `.spec` (DroneSpec instance with num_motors >= 1)
    * `reset(position, attitude)` accepts (3,) arrays
    * `step(motor_cmds, ext_force_world, dt)` accepts the documented shapes
    * `get_state()` returns a `DroneState` with all required fields and
      shapes

This is a structural test, not a physics test. Failure means the sim
cannot even be plugged into the env — all other tests would crash.
"""

from __future__ import annotations

import numpy as np

from boat_landing.drone_interface import DroneState
from evaluation.sim_validation._runner import TestResult


NAME = "T0.1 Protocol contract"


def run(sim_factory, drone_spec_path: str, **_) -> TestResult:
    metrics = {}
    try:
        sim = sim_factory(drone_spec_path)
    except Exception as exc:
        return TestResult(NAME, False, f"sim_factory raised: {exc!r}")

    for attr in ("spec", "reset", "step", "get_state"):
        if not hasattr(sim, attr):
            return TestResult(NAME, False, f"missing attribute/method: {attr}")

    spec = sim.spec
    if not hasattr(spec, "num_motors") or spec.num_motors < 1:
        return TestResult(NAME, False, "spec.num_motors invalid or missing")
    metrics["num_motors"] = float(spec.num_motors)

    n = spec.num_motors
    try:
        sim.reset(np.zeros(3), np.zeros(3))
        sim.step(np.zeros(n), np.zeros(3), 0.01)
    except Exception as exc:
        return TestResult(NAME, False, f"reset()+step() raised: {exc!r}")

    state = sim.get_state()
    if not isinstance(state, DroneState):
        return TestResult(
            NAME,
            False,
            f"get_state() returned {type(state).__name__}, expected DroneState",
        )

    expected_shapes = {
        "position": (3,),
        "velocity": (3,),
        "quaternion": (4,),
        "angular_velocity_body": (3,),
    }
    for fname, shape in expected_shapes.items():
        v = getattr(state, fname, None)
        if not isinstance(v, np.ndarray):
            return TestResult(
                NAME, False, f"state.{fname} type={type(v).__name__}, expected np.ndarray"
            )
        if v.shape != shape:
            return TestResult(
                NAME, False, f"state.{fname} shape={v.shape}, expected {shape}"
            )

    # Quaternion must be (approximately) unit norm.
    qnorm = float(np.linalg.norm(state.quaternion))
    metrics["quat_norm"] = qnorm
    if not (0.99 < qnorm < 1.01):
        return TestResult(
            NAME, False, f"quaternion not unit norm: |q|={qnorm:.4f}", metrics
        )

    return TestResult(NAME, True, "protocol contract OK", metrics)
