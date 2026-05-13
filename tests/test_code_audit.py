"""Smoke tests for the forbidden-pattern scanner."""

import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.code_audit import scan_file  # noqa: E402


def _scan_source(source: str) -> list:
    with tempfile.NamedTemporaryFile(
        suffix=".py", mode="w", encoding="utf-8", delete=False
    ) as f:
        f.write(source)
        path = Path(f.name)
    try:
        return scan_file(path)
    finally:
        path.unlink(missing_ok=True)


def test_clean_agent_has_zero_findings():
    src = """
import numpy as np
class Agent:
    def act(self, obs):
        return np.zeros(4)
"""
    findings = _scan_source(src)
    assert findings == [], findings


def test_inspect_stack_flagged():
    src = """
import inspect
class Agent:
    def act(self, obs):
        for f in inspect.stack():
            pass
        return None
"""
    findings = _scan_source(src)
    patterns = {f["pattern"] for f in findings}
    assert "forbidden_import:inspect" in patterns, findings
    assert "forbidden_call:inspect.stack" in patterns, findings


def test_gc_get_objects_flagged():
    src = """
import gc
def find_env():
    for obj in gc.get_objects():
        pass
"""
    findings = _scan_source(src)
    patterns = {f["pattern"] for f in findings}
    assert "forbidden_import:gc" in patterns, findings
    assert "forbidden_call:gc.get_objects" in patterns, findings


def test_f_locals_flagged():
    src = """
def walk():
    frame = None
    return frame.f_locals
"""
    findings = _scan_source(src)
    patterns = {f["pattern"] for f in findings}
    assert "forbidden_attr:.f_locals" in patterns, findings


def test_construct_env_flagged():
    src = """
from boat_landing.env import BoatLandingEnv
env = BoatLandingEnv("scenarios/easy.yaml")
"""
    findings = _scan_source(src)
    patterns = {f["pattern"] for f in findings}
    assert "import_env" in patterns, findings
    assert "construct_env" in patterns, findings


def test_numpy_stack_NOT_flagged():
    """np.stack is the only common false-positive class. Make sure
    the scanner does NOT flag it."""
    src = """
import numpy as np
class Sim:
    def __init__(self):
        self.motors = np.stack([np.zeros(3), np.zeros(3)])
"""
    findings = _scan_source(src)
    assert findings == [], findings


def test_subprocess_flagged():
    src = """
import subprocess
subprocess.run(["cat", "/etc/passwd"])
"""
    findings = _scan_source(src)
    patterns = {f["pattern"] for f in findings}
    assert "forbidden_import:subprocess" in patterns, findings


def test_obs_info_lookup_flagged():
    """Reading info from the env's obs dict is forbidden."""
    src = """
def act(self, obs):
    return obs["info"]["boat_position"]
"""
    findings = _scan_source(src)
    patterns = {f["pattern"] for f in findings}
    # Should hit at least one of the suspicious key lookups.
    assert any("suspicious_key_lookup" in p for p in patterns), findings
