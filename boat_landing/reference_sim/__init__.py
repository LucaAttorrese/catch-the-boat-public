"""boat_landing.reference_sim — organizer-owned reference drone simulator.

This package is the fixed target every agent is scored against. It
implements the `DroneSimulator` protocol with all of the Tier-1 fidelity
features active (motor lag, battery sag, body drag, cross-axis coupling,
internal sub-stepping, ground effect) plus a few low-amplitude effects
that are not individually scored but make the dynamics more realistic
(IMU-readback bias drift, sensor vibration coupling, RPM cap from
battery voltage).

Distribution policy
-------------------
The numerical implementation of the physics lives in `_core`. At
release time `_core` is built as a Cython extension (`.so`/`.pyd`) and
the wheel that ships to participants contains ONLY the compiled module.
The pure-Python `_core.py` you can see in source control is the master
copy used by organizers; do not distribute it.

The package's public surface is small:

    from boat_landing.reference_sim import ReferenceDroneSim, make_drone_sim

Both call the same constructor; `make_drone_sim(spec_path)` is provided
for symmetry with the participant baseline.

What participants are told
--------------------------
See `physics_notes.md` (shipped) for the qualitative description of
every effect. Numerical constants are NOT published — they are baked
into the binary distribution and tuned against the YAML drone specs.

This module never imports `boat_landing.boat`, `boat_landing.wind`, or
`boat_landing.camera`. The sealing rule applies to the reference as
much as to participant simulators.
"""

from boat_landing.reference_sim._core import ReferenceDroneSim  # noqa: F401


def make_drone_sim(spec_path: str) -> "ReferenceDroneSim":
    """Constructor symmetric with the participant baseline.

    Mirrors `agents/drone_sim_baseline.py:make_drone_sim` so the eval
    runner can switch between baseline and reference with a path-level
    flag (`--drone-sim`).
    """
    return ReferenceDroneSim(spec_path)


# Alias for path-based loaders that look for `DroneSim`.
DroneSim = ReferenceDroneSim


__all__ = ["ReferenceDroneSim", "DroneSim", "make_drone_sim"]
