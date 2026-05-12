"""Drone-sim similarity check.

When the simulator score is computed, the participant's `drone_sim.py`
is compared against a set of "trusted reference" files to detect
verbatim resubmissions. Used by `evaluation/sim_scorer.py`.

What this catches
-----------------
- Submitting `agents/drone_sim_baseline.py` unmodified (the public
  starter kit).
- Submitting the baseline with cosmetic changes only (comments removed,
  docstrings reflowed, whitespace normalised, variable renamed
  consistently).

What this does NOT catch
------------------------
- The participant rewrote the sim from scratch in their own style
  but happened to land on a similar AST shape. By design — that's
  honest engineering, not copying.
- The participant translated the binary reference sim back into Python
  via reverse engineering. The binary distribution and the no-RE
  policy (see CHALLENGE.md) push that path beyond useful effort; we
  rely on policy + organiser review for that, not on this check.

Match policy
------------
A submission whose AST is identical to any reference (after the
normalisation below) is *flagged* and capped at 0 sim-track points.
Below the cap, no further penalty is applied — partial similarity is
treated as a code-review signal, not a hard fail.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Iterable, List, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parent.parent

# Files whose AST is treated as "reference" for the verbatim-copy check.
# Add new references here as they are introduced (e.g. a public reference
# baseline derived from organiser code). The compiled binary reference
# sim has no source on disk so it is not — and cannot — be listed here.
DEFAULT_REFERENCE_FILES: List[Path] = [
    REPO_ROOT / "agents" / "drone_sim_baseline.py",
]


class SimilarityVerdict:
    """Result of comparing a submission against the reference files.

    Attributes:
        is_copy:  True iff the submission matches a reference verbatim
                  after normalisation.
        matched_reference: path of the reference that matched (when
                  is_copy is True), else None.
        note:     human-readable explanation, included in the scoring
                  breakdown.
    """

    __slots__ = ("is_copy", "matched_reference", "note")

    def __init__(
        self,
        is_copy: bool,
        matched_reference: Optional[Path],
        note: str,
    ):
        self.is_copy = bool(is_copy)
        self.matched_reference = matched_reference
        self.note = note

    def to_dict(self) -> dict:
        return {
            "is_copy": self.is_copy,
            "matched_reference": (
                str(self.matched_reference) if self.matched_reference else None
            ),
            "note": self.note,
        }


def _normalised_ast_dump(source: str) -> Optional[str]:
    """Parse and dump source as an AST string with no field annotations.

    Comments, docstrings, and whitespace differences disappear at the
    AST layer, so two files that only differ in those respects produce
    the same dump. Returns None if the source is not valid Python (we
    do not want a parse error here to fake a "non-match" verdict).
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    return ast.dump(tree, annotate_fields=False, include_attributes=False)


def check_submission(
    submission_path: str | Path,
    references: Optional[Iterable[Path]] = None,
) -> SimilarityVerdict:
    """Compare one submission against all known references.

    Returns a `SimilarityVerdict`. Never raises on a missing reference
    file — a missing reference is simply skipped (with a note in the
    scoring breakdown explaining the empty comparison).
    """
    sub_path = Path(submission_path)
    if not sub_path.is_file():
        return SimilarityVerdict(
            is_copy=False,
            matched_reference=None,
            note=f"submission file not found: {sub_path}",
        )

    sub_source = sub_path.read_text(encoding="utf-8")
    sub_dump = _normalised_ast_dump(sub_source)
    if sub_dump is None:
        return SimilarityVerdict(
            is_copy=False,
            matched_reference=None,
            note="submission did not parse as Python; similarity skipped",
        )

    refs = list(references) if references is not None else list(DEFAULT_REFERENCE_FILES)
    checked: List[Tuple[Path, bool]] = []
    for ref_path in refs:
        if not ref_path.is_file():
            checked.append((ref_path, False))
            continue
        ref_dump = _normalised_ast_dump(ref_path.read_text(encoding="utf-8"))
        if ref_dump is None:
            checked.append((ref_path, False))
            continue
        if ref_dump == sub_dump:
            return SimilarityVerdict(
                is_copy=True,
                matched_reference=ref_path,
                note=(
                    f"submission AST matches {ref_path.name} verbatim "
                    f"(after comment/whitespace normalisation); "
                    f"sim-track score capped at 0"
                ),
            )
        checked.append((ref_path, True))

    checked_names = ", ".join(
        f"{p.name}{'' if found else ' (missing)'}" for p, found in checked
    )
    return SimilarityVerdict(
        is_copy=False,
        matched_reference=None,
        note=f"no verbatim match against references [{checked_names}]",
    )
