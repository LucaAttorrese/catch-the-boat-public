# Docker — evaluation environment

All scoring happens inside a Docker container with fixed CPU and memory
limits, on a single organizer machine. The container is the **contract**:
if your code meets the latency budget *inside it*, it meets it at
evaluation time. If it doesn't, it won't.

This page explains how to install Docker, build the local image, and
run the same container the organizers will run.

---

## Installing Docker

Install **before** the event. Docker Desktop is a 600 MB-2 GB download
and the `:full` image pull is ~500 MB — both are slow on event-day Wi-Fi.

### Windows — install with ADMINISTRATOR rights, system-wide

> ⚠️ **Do NOT install Docker Desktop in "user only" mode.** A per-user
> install fails on volume mounts (your `${PWD}:/workspace` mount may
> silently land in a sandboxed location), the WSL 2 backend
> integration doesn't always register, and the `docker` CLI sometimes
> ends up not on PATH for other shells. **Always install with admin
> rights, in the system-wide default location** (`C:\Program Files\Docker`).

Steps:

1. Download Docker Desktop for Windows from
   <https://www.docker.com/products/docker-desktop/>.
2. **Right-click the installer → "Run as administrator"**. If the UAC
   prompt doesn't show up, you're not running with elevation — abort
   and restart the install. The installer's default options are
   correct; do not change the install path away from
   `C:\Program Files\Docker`.
3. When asked: select **"Use the WSL 2 based engine"** (default).
4. Reboot when prompted.
5. After reboot, open Docker Desktop once so it can finish initializing
   WSL 2 (it'll ask to download / update the WSL 2 kernel — accept).
6. Open a **new** PowerShell (any user, not necessarily admin) and run:

   ```powershell
   docker --version
   docker run --rm hello-world
   ```

   Both must succeed. If the second one prints "Hello from Docker!",
   you're done.

Requirements:
- Windows 10 22H2 / 11 (64-bit)
- WSL 2 enabled (`wsl --install` if you haven't already)
- Virtualization (VT-x / AMD-V) enabled in BIOS — most laptops have it
  on by default; if `docker run` complains about virtualization, reboot
  into BIOS and enable it
- 8 GB+ RAM (16 GB recommended; the eval container is given 8 GB so
  with the host OS overhead you want 16 to be comfortable)
- ~10 GB free disk for images + container layers

### macOS

1. Download Docker Desktop for Mac (pick the Apple Silicon build on
   M1/M2/M3/M4 hardware, Intel build on older Macs).
2. Drag to `/Applications`, open, accept the "Docker needs privileged
   access" prompt (this is a one-time admin password — it's how Docker
   sets up the privileged helper).
3. Verify:

   ```bash
   docker --version
   docker run --rm hello-world
   ```

### Linux

Use the official repository to install Docker Engine + Compose (NOT
Docker Desktop unless you specifically want the GUI):

- Ubuntu / Debian: <https://docs.docker.com/engine/install/ubuntu/>
- Fedora: <https://docs.docker.com/engine/install/fedora/>
- Arch: `sudo pacman -S docker docker-buildx docker-compose`

Then add yourself to the `docker` group so you can run `docker` without
`sudo`:

```bash
sudo usermod -aG docker $USER
newgrp docker          # picks up the new group in this shell
docker run --rm hello-world
```

If `hello-world` works without `sudo`, you're set. Log out and back in
to make the group change permanent for future shells.

---

## Two images

| Image | Source | Contains | Used for |
| --- | --- | --- | --- |
| `challenge-eval:base` | `docker/Dockerfile.eval` (this repo) | Python 3.10, all `requirements.txt`, baseline drone sim | Local dev. Build it yourself with the command below. |
| `ghcr.io/skyeusoftware/catch-the-boat:2026-hackathon` | Published by organizers on GHCR | All of `:base`, plus the compiled **reference simulator** binary | Final evaluation. **Publicly pullable, no login required.** Pull and re-tag as `challenge-eval:full` to use it locally (see TL;DR below). |

The reference simulator is a compiled Cython binary; the source is not
included in `:full`. For the qualitative description of what the
reference models, see `boat_landing/reference_sim/physics_notes.md`.

---

## TL;DR

```bash
# build the base image (your dev environment)
docker build -f docker/Dockerfile.eval -t challenge-eval:base .

# run scenarios with the baseline drone simulator inside the container
./docker/run-local.sh python evaluation/evaluate.py --scenario hard --headless --seed 42

# calibrate your machine against the organizer reference timing
./docker/run-local.sh python docker/benchmark.py

# --- evaluate against the reference simulator (publicly pullable) ---
docker pull ghcr.io/skyeusoftware/catch-the-boat:2026-hackathon
docker tag  ghcr.io/skyeusoftware/catch-the-boat:2026-hackathon challenge-eval:full
CHALLENGE_IMAGE=challenge-eval:full ./docker/run-local.sh \
    python evaluation/evaluate.py --use-reference-sim --scenario hard --headless

# Note: --use-reference-sim outside the :full image exits with code 2
# and a placeholder JSON ({"outcome":"ERROR", "score":0, ...}) on stdout
# — so batch pipelines that read one JSON per run don't choke.
```

Windows users: replace `./docker/run-local.sh` with `.\docker\run-local.ps1`.

---

## Why a container?

Two reasons:

1. **Reproducibility.** Same Python, same OpenCV, same NumPy, same
   PyBullet wheel. No "works on my Mac, breaks in eval".
2. **Resource fairness.** The container is started with
   `--cpus=6 --memory=8g`. A submission that exploits 16 cores on a
   dev laptop will see those 6 cores at scoring time. Better to know
   that during dev.

Picking the limits at **6 vCPU / 8 GB RAM** approximates the deployment
target (NVIDIA Jetson Orin Nano 8 GB, MAXN mode) on the CPU side. The
final eval is run on **one fixed organizer machine** — the same one for
every team — so any residual host-vs-host variation is absorbed.

---

## What the container ships

| Item | Value |
| --- | --- |
| Base image | `python:3.10-slim-bookworm` |
| Python | 3.10.x |
| Dependencies | exactly `requirements.txt` |
| OS libs | `libgl1`, `libglib2.0-0`, `libgomp1`, `libsm6`, `libxext6`, `libxrender1`, `build-essential` |
| Headless flags | `SDL_VIDEODRIVER=dummy`, `MPLBACKEND=Agg`, `PYBULLET_EGL=1` |
| Mount point for the repo | `/workspace` |

You can install **anything you want** on top inside your own image —
just add a `Dockerfile` next to your submission that does
`FROM challenge-eval` and `RUN pip install ...`. The base image is the
floor, not the ceiling.

---

## Standard workflows

### Run a single scenario

```bash
./docker/run-local.sh python evaluation/evaluate.py \
    --agent agents/agent_baseline.py \
    --drone-sim agents/drone_sim_baseline.py \
    --scenario easy --headless --seed 42
```

### Run the simulator-quality scorer

```bash
./docker/run-local.sh python evaluation/sim_scorer.py \
    --drone-sim   agents/drone_sim_baseline.py \
    --drone       vtol \
    --submission  evaluation/submission_baseline.yaml
```

### Open an interactive shell inside the container

```bash
./docker/run-local.sh bash
```

You're now at `/workspace` in the container. Edit code on your host
(your IDE), re-run inside the container, repeat.

### Run pytest

```bash
./docker/run-local.sh pytest tests/
```

---

## Calibrating your machine

The 20 ms p95 latency budget (see `docs/AGENT_SCORING.md`, "HW‑readiness")
is measured **on the organizer eval machine**, inside the container.
If your laptop is much faster (or slower) than that machine, your
local timings won't match what you'll see at scoring time.

To bridge the gap:

```bash
./docker/run-local.sh python docker/benchmark.py
```

Output (excerpt):

```
  numpy_linalg     1.20 s
  image_ops        2.10 s
  control_loop     0.80 s
  total            4.10 s   (reference: 4.50 s)
  factor           0.91
  verdict          MATCHED to the eval machine (±15%)
  guidance         Your local timings are a good proxy for evaluation timings.
```

Three outcomes:

| `factor` | Meaning | What it means for your 20 ms p95 budget |
| --- | --- | --- |
| `< 0.85` | Your host is **faster** than the eval machine | Local p95 = 12 ms → eval p95 ≈ 13 ms but could be 16+ ms; leave margin. |
| `0.85 – 1.15` | **Matched** | Local p95 ≈ eval p95. |
| `> 1.15` | Your host is **slower** | Local p95 = 28 ms → eval p95 ≈ 22 ms; might pass, don't over-optimize. |

The benchmark is **diagnostic only** and never blocks. Run it whenever
you change machines or want to confirm what timing target to aim for.

---

## What scoring looks like

For every submission, organizers run roughly:

```bash
docker run --rm \
    --cpus=6 --memory=8g \
    -v <submission_path>:/workspace \
    -w /workspace \
    challenge-eval \
    python evaluation/evaluate.py --scenario <name> --headless --seed <fixed>
```

Repeated for each private scenario, with a fixed seed. The latency
profiler in `evaluate.py` records p95 of `agent.act()` over the full
run and writes it to the result JSON. The scorer maps p95 ≤ 20 ms to
the HW‑readiness latency bonus.

---

## Known gotchas

### `pybullet` falls back to TINY renderer

Inside the container there's no display. The env automatically uses
PyBullet's TINY (CPU) renderer for camera frames. That's the **same
renderer the scorer uses**, so your local sim and the eval sim are
pixel-identical. Don't try to enable `--gui` inside the container — it
won't work and isn't needed.

### File permissions on Linux

`docker/run-local.sh` passes `--user $(id -u):$(id -g)`, so files
created inside the container are owned by you on the host. If you see
root-owned files, you ran a `docker run` by hand and forgot the flag —
clean them up with `sudo chown -R $(id -u):$(id -g) .`.

### Windows: filesystem performance

Mounting your repo into a Linux container on Windows goes through a
9P/virtiofs layer. If pytest feels slow, that's why. Workarounds:
move the repo onto the WSL2 filesystem (`\\wsl$\Ubuntu\home\...`), or
just accept the cost — it's tolerable for the challenge workloads.

### Network access during evaluation

The eval run **does not** have internet (the container is started with
`--network none` in scoring). If your agent downloads weights from the
internet, package them with your submission. The local
`run-local.sh` defaults leave network on for convenience during dev.
