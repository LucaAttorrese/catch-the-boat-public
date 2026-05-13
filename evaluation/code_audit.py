"""Forbidden-pattern scanner for team submissions.

ORGANIZER-ONLY. Walks the AST of `agent.py` and `drone_sim.py` and flags
patterns that are explicitly forbidden by the challenge rules:

    - `import inspect` + uses of `inspect.stack`, `inspect.currentframe`,
      `inspect.getframeinfo` (used to traverse the call stack and reach
      the env's internals, bypassing perception).
    - `import gc` + `gc.get_objects`, `gc.get_referrers` (similar reach).
    - Direct reads of `info` (the env's debug dict — forbidden to use).
    - References to internal env attributes like `_traj`, `_get_info`,
      or constructing a new BoatLandingEnv inside the agent.
    - Reading files outside the agent's own directory (heuristic).
    - `subprocess`, `multiprocessing`, `os.exec*`, `socket` — any of
      these would let the agent escape sandboxing.

This scanner is a HEURISTIC. A determined cheater can obfuscate. Its
purpose is to produce a list of hits that a human reviewer will check
before awarding the top scores. Any submission with hits should be
audited by hand before going on the official scoreboard.

Output: JSON to stdout (and optionally to --output path) with the
structure:

    {
      "files_scanned": [...],
      "findings": [
        {"file": ..., "line": ..., "pattern": ..., "snippet": ...},
        ...
      ],
      "clean": true | false
    }

Exit code: 0 always — scanner never fails the eval; it just reports.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path
from typing import Iterable, List, Tuple


# Forbidden imports — module-level (e.g. `import inspect`).
FORBIDDEN_IMPORTS = {
    "inspect",
    "gc",
    "subprocess",
    "multiprocessing",
    "socket",
    "ctypes",
    "pickle",   # arbitrary code execution via unpickle
}

# Forbidden attribute reads / calls. These are short attr names whose
# ONLY plausible use is introspection / cheating (no overlap with
# numpy.stack, list.stack, etc.). The forbidden imports above already
# catch `inspect`/`gc`/`subprocess`; this list is the second line.
FORBIDDEN_ATTRS = {
    "currentframe",   # inspect.currentframe
    "getframeinfo",   # inspect.getframeinfo
    "get_objects",    # gc.get_objects
    "get_referrers",  # gc.get_referrers
    "f_locals",       # frame.f_locals — almost always used for cheating
    "f_globals",
    "f_back",
    "_get_info",      # env's private accessor
    "_traj",          # env's private trajectory
}

# Chains of attribute access that are forbidden when seen verbatim.
# Matched against `_attr_chain(node)` output.
FORBIDDEN_CHAINS = {
    "inspect.stack",
    "inspect.currentframe",
    "inspect.getframeinfo",
    "gc.get_objects",
    "gc.get_referrers",
}

# Forbidden function names. Matched against Call nodes' func.
FORBIDDEN_CALLS = {
    "execfile",
    "compile",
    "exec",
    "eval",
}

# Forbidden module attributes (full module path).
FORBIDDEN_FULL = {
    "os.execv", "os.execvp", "os.execve",
    "os.system",
    "subprocess.run", "subprocess.Popen", "subprocess.call",
}

# Sentinel: importing/constructing the env itself.
ENV_CLASS = "BoatLandingEnv"


def _attr_chain(node: ast.AST) -> str:
    """Return `a.b.c` for an Attribute chain, else empty."""
    parts: List[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    parts.reverse()
    return ".".join(parts) if parts else ""


def scan_file(path: Path) -> List[dict]:
    """Return a list of finding dicts for one Python file."""
    findings: List[dict] = []
    try:
        source = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError) as exc:
        return [{
            "file": str(path),
            "line": 0,
            "pattern": "READ_ERROR",
            "snippet": f"could not read file: {exc}",
        }]

    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return [{
            "file": str(path),
            "line": exc.lineno or 0,
            "pattern": "PARSE_ERROR",
            "snippet": f"syntax error: {exc.msg}",
        }]

    lines = source.splitlines()

    def _line(lineno: int) -> str:
        if 0 < lineno <= len(lines):
            return lines[lineno - 1].strip()
        return ""

    for node in ast.walk(tree):
        # Forbidden imports.
        if isinstance(node, ast.Import):
            for alias in node.names:
                base = alias.name.split(".")[0]
                if base in FORBIDDEN_IMPORTS:
                    findings.append({
                        "file": str(path),
                        "line": node.lineno,
                        "pattern": f"forbidden_import:{alias.name}",
                        "snippet": _line(node.lineno),
                    })
        elif isinstance(node, ast.ImportFrom):
            base = (node.module or "").split(".")[0]
            if base in FORBIDDEN_IMPORTS:
                findings.append({
                    "file": str(path),
                    "line": node.lineno,
                    "pattern": f"forbidden_import:{node.module}",
                    "snippet": _line(node.lineno),
                })
            if base == "boat_landing" and any(
                a.name == "env" or a.name == ENV_CLASS for a in node.names
            ):
                findings.append({
                    "file": str(path),
                    "line": node.lineno,
                    "pattern": "import_env",
                    "snippet": _line(node.lineno),
                })

        # Forbidden attribute accesses anywhere.
        if isinstance(node, ast.Attribute):
            if node.attr in FORBIDDEN_ATTRS:
                findings.append({
                    "file": str(path),
                    "line": node.lineno,
                    "pattern": f"forbidden_attr:.{node.attr}",
                    "snippet": _line(node.lineno),
                })
            chain = _attr_chain(node)
            if chain in FORBIDDEN_FULL or chain in FORBIDDEN_CHAINS:
                findings.append({
                    "file": str(path),
                    "line": node.lineno,
                    "pattern": f"forbidden_call:{chain}",
                    "snippet": _line(node.lineno),
                })

        # Direct constructor of BoatLandingEnv.
        if isinstance(node, ast.Call):
            func = node.func
            name = None
            if isinstance(func, ast.Name):
                name = func.id
            elif isinstance(func, ast.Attribute):
                name = func.attr
            if name == ENV_CLASS:
                findings.append({
                    "file": str(path),
                    "line": node.lineno,
                    "pattern": "construct_env",
                    "snippet": _line(node.lineno),
                })
            if name in FORBIDDEN_CALLS:
                findings.append({
                    "file": str(path),
                    "line": node.lineno,
                    "pattern": f"forbidden_call:{name}",
                    "snippet": _line(node.lineno),
                })

        # `info` is a forbidden parameter name reading in act() etc.
        # (heuristic: any function whose body reads obj["info"] or
        # has a parameter named info that gets indexed).
        if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant):
            if node.slice.value == "info" or node.slice.value == "boat_position":
                # this matches obs["info"] or anywhere we look up these
                # specific keys — flag for review since obs["state"] is
                # the legit pattern.
                findings.append({
                    "file": str(path),
                    "line": node.lineno,
                    "pattern": f"suspicious_key_lookup:{node.slice.value}",
                    "snippet": _line(node.lineno),
                })

    return findings


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--agent", required=True, help="Path to team's agent.py")
    ap.add_argument("--drone-sim", required=True, help="Path to team's drone_sim.py")
    ap.add_argument("--output", default=None, help="Write JSON report to this path")
    args = ap.parse_args()

    files = [Path(args.agent), Path(args.drone_sim)]
    findings: List[dict] = []
    for f in files:
        if not f.is_file():
            findings.append({
                "file": str(f),
                "line": 0,
                "pattern": "MISSING",
                "snippet": "submission file not found",
            })
            continue
        findings.extend(scan_file(f))

    report = {
        "files_scanned": [str(f) for f in files],
        "findings": findings,
        "clean": len(findings) == 0,
        "summary": (
            "No suspicious patterns. Submission may be auto-scored."
            if not findings
            else f"{len(findings)} suspicious pattern(s) detected. "
                 "Manual review recommended before awarding score."
        ),
    }
    payload = json.dumps(report, indent=2)
    print(payload)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(payload, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
