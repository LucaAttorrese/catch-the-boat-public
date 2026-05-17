"""Common plumbing for the sim_validation suite.

Provides:
    * `TestResult` — uniform return type for every test.
    * `load_sim_factory` — load a participant's drone-sim module dynamically
      and return a callable `(spec_path) -> DroneSimulator`.
    * `make_hover_action` — compute the per-motor throttle that holds
      hover for a given drone spec under symmetric motor placement.

Tests are pure functions (`run(sim_factory, drone_spec_path, **kwargs)
-> TestResult`). They do not import pytest, so the scorer can call them
directly without a pytest dependency at evaluation time.
"""

from __future__ import annotations

import importlib.util
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Optional

import numpy as np

from boat_landing.drone_interface import DroneSimulator, DroneSpec, load_drone_spec


REPO_ROOT = Path(__file__).resolve().parent.parent.parent


@dataclass
class TestResult:
    """Uniform return type for every validation test.

    name:    short label like "T0.2 Hover steady-state".
    passed:  True iff the test's pass criteria are met.
    message: human-readable explanation (always populated).
    metrics: raw measurements that produced the pass/fail decision —
             included in the JSON breakdown so participants can see
             *why* their sim failed and tune accordingly.
    """

    name: str
    passed: bool
    message: str
    metrics: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "passed": self.passed,
            "message": self.message,
            "metrics": dict(self.metrics),
        }


def load_sim_factory(drone_sim_path: str) -> Callable[[str], DroneSimulator]:
    """Dynamically load a participant's drone-sim module and return the
    callable that constructs a sim from a spec path.

    The module must expose either `make_drone_sim(spec_path)` (preferred)
    or a `DroneSim` class taking the spec path. Raises `AttributeError`
    if neither is present. Module paths are made unique per call so
    repeated imports during a scoring run do not collide.
    """
    path = Path(drone_sim_path)
    if not path.is_file():
        raise FileNotFoundError(f"Drone sim file not found: {path}")
    if path.suffix != ".py":
        raise ValueError(f"Drone sim must be a .py file, got {path.suffix}")
    module_name = f"_dynamic_drone_sim_{int(time.time() * 1000)}_{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    if hasattr(module, "make_drone_sim") and callable(module.make_drone_sim):
        return module.make_drone_sim  # callable(spec_path) -> sim
    if hasattr(module, "DroneSim"):
        return module.DroneSim  # class(spec_path) -> sim
    raise AttributeError(
        f"Drone sim module {drone_sim_path} must expose "
        f"`make_drone_sim(spec_path)` or `DroneSim`."
    )


def make_hover_action(spec: DroneSpec) -> np.ndarray:
    """Per-motor throttle that produces hover thrust under symmetric
    motor placement (all motors thrust along body +z, equal moment arms).

    The shipped vtol.yaml is symmetric in this sense so this is the
    analytically correct hover. For asymmetric sims the proper hover
    comes from the inverse mixer in DefaultAttitudeController — use
    that one instead in tests where attitude must be stabilized.
    """
    T_per_motor = spec.hover_thrust_per_motor
    omega_hover = float(np.sqrt(max(T_per_motor, 0.0) / max(spec.propeller.thrust_coefficient, 1e-12)))
    return np.full(spec.num_motors, omega_hover / spec.motor.omega_max, dtype=np.float64)


def quat_to_rpy(q: np.ndarray) -> np.ndarray:
    """PyBullet (x, y, z, w) quaternion to RPY (Z-Y-X intrinsic)."""
    x, y, z, w = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    sinp = 2.0 * (w * y - z * x)
    pitch = np.arcsin(np.clip(sinp, -1.0, 1.0))
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return np.array([roll, pitch, yaw], dtype=np.float64)


def safe_load_spec(drone_spec_path: str) -> DroneSpec:
    """Convenience wrapper used by every test."""
    return load_drone_spec(drone_spec_path)
