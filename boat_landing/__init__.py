"""Catch the Boat — simulation environment package.

Public exports:
    BoatLandingEnv: the gym-style environment used by all agents.
    load_scenario:  helper that parses a scenario YAML into a dict.
"""

from boat_landing.env import BoatLandingEnv, load_scenario

__all__ = ["BoatLandingEnv", "load_scenario"]
__version__ = "0.1.0"
