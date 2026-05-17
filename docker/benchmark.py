"""Calibrate your machine against the official evaluation target.

Run inside the eval container:

    ./docker/run-local.sh python docker/benchmark.py

The script runs a fixed CPU-bound workload that exercises the same code
paths as the agent loop (numpy ops, ArUco-style image filtering, simple
linear algebra) and prints a scaling factor relative to the organizer's
reference machine. Use it to translate your local p95 latency to what
you'll see at evaluation time.

Reference target: organizer eval machine running the same Dockerfile.eval
with --cpus=6 --memory=8g. Calibration constant `REFERENCE_RUNTIME_S` is
the runtime observed on that machine for the workload defined here.

Output JSON to stdout, human summary to stderr. Exit 0 always (this is
diagnostic, never blocking).
"""

from __future__ import annotations

import json
import platform
import sys
import time

import numpy as np

# Wall-clock seconds the workload takes on the organizer reference
# machine (the laptop that will run the official scoring at the event),
# inside this same Dockerfile.eval container with --cpus=6 --memory=8g
# and BLAS threads pinned to 6 (OPENBLAS/OMP/MKL_NUM_THREADS=6, set by
# docker/run-local.ps1 / .sh). Pinning is required: without it BLAS
# oversubscribes the cgroup CPU quota and run-to-run variance is ~3x.
# Measured 2026-05-17 on the event laptop (Windows 11 + WSL2 + Docker
# Desktop), mean of 7 runs, std dev 0.06 s (CV 3.1%). Re-measure on
# your hardware via:
#     ./docker/run-local.sh python docker/benchmark.py
# and read the "scaling_factor" in the JSON to compare.
REFERENCE_RUNTIME_S = 1.93


def _workload_numpy(n: int = 50) -> float:
    """Numpy-heavy: matmul + svd. Stresses BLAS like a Kalman update would."""
    rng = np.random.default_rng(0)
    total = 0.0
    t0 = time.perf_counter()
    for _ in range(n):
        A = rng.standard_normal((256, 256))
        B = rng.standard_normal((256, 256))
        C = A @ B
        u, s, vh = np.linalg.svd(C, full_matrices=False)
        total += float(s.sum())
    return time.perf_counter() - t0


def _workload_image(n: int = 200) -> float:
    """Image-ops heavy: 2d convolution + thresholding. Mimics ArUco preproc."""
    rng = np.random.default_rng(1)
    img = rng.integers(0, 255, size=(480, 640), dtype=np.uint8)
    kernel = np.ones((5, 5), dtype=np.float32) / 25.0
    t0 = time.perf_counter()
    for _ in range(n):
        # Naive separable blur, deterministic and reasonably costly.
        f = img.astype(np.float32)
        blurred = np.zeros_like(f)
        # Two 1d passes; not as fast as cv2.filter2D but more reproducible.
        for r in range(2, f.shape[0] - 2):
            blurred[r] = (
                f[r - 2] * 0.1 + f[r - 1] * 0.2 + f[r] * 0.4
                + f[r + 1] * 0.2 + f[r + 2] * 0.1
            )
        thr = (blurred > 128).astype(np.uint8) * 255
        img = thr.astype(np.uint8)
    return time.perf_counter() - t0


def _workload_control(n: int = 5000) -> float:
    """Tight loop: small matmuls + trig. Mimics the inner control loop."""
    rng = np.random.default_rng(2)
    M = rng.standard_normal((4, 8))
    M_pinv = np.linalg.pinv(M)
    state = rng.standard_normal(4)
    # Disturbance term shaped to match `state` (broadcast-compatible).
    omega = rng.standard_normal(4)
    t0 = time.perf_counter()
    for i in range(n):
        # Simulate a per-step control update.
        target = np.array([np.sin(i * 0.01), np.cos(i * 0.01), 0.0, 1.0])
        wrench = target - state
        cmd = M_pinv @ wrench
        cmd = np.clip(cmd, 0.0, 1.0)
        # Feedback into state to avoid the optimizer eliding work.
        state = state + 0.01 * (wrench + 0.1 * np.sin(omega))
    return time.perf_counter() - t0


def main() -> int:
    print("Catch the Boat — machine calibration", file=sys.stderr)
    print("=" * 60, file=sys.stderr)

    # Warm up to avoid first-iteration import / JIT artefacts.
    _ = _workload_numpy(n=5)

    t_np = _workload_numpy()
    t_img = _workload_image()
    t_ctl = _workload_control()
    total = t_np + t_img + t_ctl
    factor = total / REFERENCE_RUNTIME_S

    if factor < 0.85:
        verdict = "FASTER than the eval machine"
        guidance = (
            f"Your local p95 latency will be ~{factor*100:.0f}% of what you'll "
            f"see at evaluation. Aim for a margin (target p95 < {20 / factor:.1f} ms "
            f"locally to stay under 20 ms in eval)."
        )
    elif factor < 1.15:
        verdict = "MATCHED to the eval machine (±15%)"
        guidance = "Your local timings are a good proxy for evaluation timings."
    else:
        verdict = "SLOWER than the eval machine"
        guidance = (
            f"Your local p95 will be ~{factor*100:.0f}% of evaluation. "
            f"If you measure p95 = {20 * factor:.1f} ms locally, you'll still "
            f"pass the 20 ms threshold."
        )

    result = {
        "machine": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "cpu_count": _cpu_count_in_container(),
        },
        "workload_seconds": {
            "numpy_linalg": round(t_np, 3),
            "image_ops": round(t_img, 3),
            "control_loop": round(t_ctl, 3),
            "total": round(total, 3),
        },
        "reference_seconds": REFERENCE_RUNTIME_S,
        "scaling_factor": round(factor, 3),
        "verdict": verdict,
        "guidance": guidance,
    }

    print(f"  numpy_linalg    {t_np:6.2f} s", file=sys.stderr)
    print(f"  image_ops       {t_img:6.2f} s", file=sys.stderr)
    print(f"  control_loop    {t_ctl:6.2f} s", file=sys.stderr)
    print(f"  total           {total:6.2f} s   (reference: {REFERENCE_RUNTIME_S:.2f} s)", file=sys.stderr)
    print(f"  factor          {factor:6.2f}", file=sys.stderr)
    print(f"  verdict         {verdict}", file=sys.stderr)
    print(f"  guidance        {guidance}", file=sys.stderr)
    print("=" * 60, file=sys.stderr)

    print(json.dumps(result, indent=2))
    return 0


def _cpu_count_in_container() -> int:
    """Best-effort CPU count visible to this process."""
    try:
        import os
        n = os.cpu_count()
        return int(n) if n else 0
    except Exception:
        return 0


if __name__ == "__main__":
    sys.exit(main())
