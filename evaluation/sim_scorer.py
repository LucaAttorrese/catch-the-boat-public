"""CLI to score a participant's drone simulator on the validation suite.

Usage:
    python evaluation/sim_scorer.py \
        --drone-sim agents/drone_sim_baseline.py \
        --drone quadcopter \
        --submission evaluation/submission_baseline.yaml

The scorer:
    1. Runs the four Tier 0 gate tests. ALL must pass; otherwise the
       simulator score is 0 and Tier 1 is skipped.
    2. Reads `submission.yaml` to find which Tier 1 features the
       participant claims to implement.
    3. Runs the test for each declared Tier 1 feature. Each pass = 5
       points. A *declared* feature whose test fails earns 0 AND is
       flagged in the breakdown ("declared but not detected").
    4. Tier 2 (rubric) and the validation-dataset bonus are NOT scored
       here — those are handled by judge review and a separate
       reference-trajectory comparison respectively.

Prints a JSON breakdown to stdout. Exit code 0 if Tier 0 passed,
1 otherwise.

Schema of submission.yaml:
    team:                "Team X"
    chosen_drone:        "quadcopter" | "vtol"     # which spec the agent uses
    drone_sim_path:      relative path to drone-sim .py
    agent_path:          relative path to agent .py
    simulator_features:
      tier_1:
        motor_lag:        bool
        battery_sag:      bool
        aero_drag:        bool
        cross_coupling:   bool
        substepping:      bool
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Optional

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.similarity import check_submission  # noqa: E402
from evaluation.sim_validation import (  # noqa: E402
    POINTS_PER_TIER1_FEATURE,
    TIER0_TESTS,
    TIER1_TESTS,
)
from evaluation.sim_validation._runner import (  # noqa: E402
    TestResult,
    load_sim_factory,
)


DRONES_DIR = REPO_ROOT / "drones"


def resolve_drone_spec(name_or_path: str) -> Path:
    p = Path(name_or_path)
    if p.is_file():
        return p
    candidate = DRONES_DIR / f"{name_or_path}.yaml"
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(
        f"Drone spec {name_or_path!r} not found (looked at {p} and {candidate})."
    )


def _run_test_module(module_path: str, sim_factory, drone_spec_path: str) -> TestResult:
    """Import and execute a single test module. Catches exceptions so a
    crashy test never derails the whole scoring run."""
    try:
        module = importlib.import_module(module_path)
    except Exception as exc:
        return TestResult(
            module_path,
            False,
            f"could not import test module: {exc!r}",
        )
    try:
        return module.run(sim_factory, drone_spec_path)
    except Exception as exc:
        name = getattr(module, "NAME", module_path)
        return TestResult(name, False, f"test raised: {exc!r}")


def score_simulator(
    drone_sim_path: str,
    drone_spec_path: str,
    declared_features: dict,
    skip_similarity: bool = False,
) -> dict:
    """Run the full validation suite and return a breakdown dict.

    `declared_features` is a dict of {feature_id: bool}. Only features
    set to True are tested for points. Unknown feature IDs are flagged
    in the breakdown but don't affect the score.

    A similarity check against the public baseline runs first. If the
    submission is a verbatim copy of any reference file, the total
    score is capped at 0 (the gate and feature tests still run for the
    breakdown JSON, so the team sees what they would have scored).

    `skip_similarity` disables that capping. Use it for organiser smoke
    testing (running the scorer against the literal baseline file path
    always triggers the match — which is correct for participants but
    annoying for CI sanity checks).
    """
    sim_factory = load_sim_factory(drone_sim_path)
    if skip_similarity:
        from evaluation.similarity import SimilarityVerdict
        similarity = SimilarityVerdict(
            is_copy=False,
            matched_reference=None,
            note="similarity check skipped (--skip-similarity)",
        )
    else:
        similarity = check_submission(drone_sim_path)
    breakdown = {
        "similarity": similarity.to_dict(),
        "tier_0_gate": [],
        "tier_1": [],
        "tier_0_passed": False,
        "tier_1_score": 0,
        "total_score": 0,
        "max_score_possible": len(TIER1_TESTS) * POINTS_PER_TIER1_FEATURE,
    }

    # Tier 0: every test must pass.
    gate_passed = True
    for module_path in TIER0_TESTS:
        r = _run_test_module(module_path, sim_factory, drone_spec_path)
        breakdown["tier_0_gate"].append(r.to_dict())
        if not r.passed:
            gate_passed = False
    breakdown["tier_0_passed"] = gate_passed

    if not gate_passed:
        breakdown["total_score"] = 0
        breakdown["reason"] = "Tier 0 gate failed; Tier 1 skipped"
        return breakdown

    # Tier 1: only declared features.
    tier1_score = 0
    for feature_id, claimed in declared_features.items():
        if feature_id not in TIER1_TESTS:
            breakdown["tier_1"].append({
                "feature": feature_id,
                "claimed": bool(claimed),
                "passed": False,
                "score": 0,
                "message": "unknown feature ID — ignored",
            })
            continue
        if not claimed:
            continue
        module_path = TIER1_TESTS[feature_id]
        r = _run_test_module(module_path, sim_factory, drone_spec_path)
        feature_score = POINTS_PER_TIER1_FEATURE if r.passed else 0
        tier1_score += feature_score
        breakdown["tier_1"].append({
            "feature": feature_id,
            "claimed": True,
            "passed": r.passed,
            "score": feature_score,
            "name": r.name,
            "message": r.message,
            "metrics": dict(r.metrics),
        })

    breakdown["tier_1_score"] = tier1_score
    # Verbatim copies forfeit the sim-track score outright. Everything
    # above this point is preserved for diagnostic value.
    if similarity.is_copy:
        breakdown["total_score"] = 0
        breakdown["reason"] = (
            "Submission is a verbatim copy of a reference file; "
            "sim-track score capped at 0 (see breakdown.similarity)."
        )
    else:
        breakdown["total_score"] = tier1_score
    return breakdown


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--drone-sim",
        required=True,
        help="Path to a .py file exposing make_drone_sim(spec_path)/DroneSim.",
    )
    parser.add_argument(
        "--drone",
        required=True,
        help="Drone spec name (e.g. 'quadcopter', 'vtol') or path to YAML.",
    )
    parser.add_argument(
        "--submission",
        required=True,
        help="Path to submission.yaml declaring which Tier 1 features to test.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to write the breakdown JSON (in addition to stdout).",
    )
    parser.add_argument(
        "--skip-similarity",
        action="store_true",
        help="Skip the AST similarity check (organiser smoke testing). "
        "Submitting the baseline file path always matches itself — use "
        "this when running on the literal baseline for CI.",
    )
    args = parser.parse_args()

    drone_spec_path = resolve_drone_spec(args.drone)
    submission_path = Path(args.submission)
    if not submission_path.is_file():
        print(f"Submission file not found: {submission_path}", file=sys.stderr)
        return 2
    with open(submission_path, "r", encoding="utf-8") as f:
        submission = yaml.safe_load(f) or {}

    features = (submission.get("simulator_features", {}) or {}).get("tier_1", {}) or {}
    breakdown = score_simulator(
        args.drone_sim,
        str(drone_spec_path),
        features,
        skip_similarity=args.skip_similarity,
    )
    breakdown["meta"] = {
        "team": str(submission.get("team", "unknown")),
        "drone_sim": str(Path(args.drone_sim).resolve()),
        "drone_spec": str(drone_spec_path),
        "submission": str(submission_path.resolve()),
    }
    payload = json.dumps(breakdown, indent=2, default=_json_default)
    print(payload)
    if args.output:
        Path(args.output).write_text(payload, encoding="utf-8")
    return 0 if breakdown.get("tier_0_passed") else 1


def _json_default(obj):
    import numpy as np
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    raise TypeError(f"Cannot serialize {type(obj).__name__}")


if __name__ == "__main__":
    sys.exit(main())
