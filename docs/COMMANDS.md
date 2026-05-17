# Commands

Quick reference for every command you need to install, run, debug, and
evaluate the agent. Paths are relative to the `catch-the-boat/` repo
root.

> **Note.** Every command below assumes your Python 3.10 environment is
> active. How you activate it depends on the OS and install path you
> chose:
>
> - **Linux** (venv): `source .venv/bin/activate`
> - **macOS Intel** (venv): `source .venv/bin/activate`
> - **macOS Apple Silicon** (Miniforge, recommended): `conda activate catch-the-boat`
> - **Windows** (Miniconda, recommended): `conda activate catch-the-boat`
>
> **PyBullet wheel availability** (compiler-free install): Linux pip
> wheels; macOS Intel pip wheels; macOS ARM **only via conda-forge**;
> Windows **only via conda-forge**. See sections 1.1–1.3 for the
> OS-by-OS details.
>
> **Docker is a separate prerequisite** for running against the
> organizer reference simulator. Install it in parallel — see
> [`DOCKER.md#installing-docker`](DOCKER.md#installing-docker). On
> Windows install with administrator rights (system-wide), not
> "user only".

---

## 1. Initial setup

### 1.1 Install (Linux)

Standard path: venv + pip. PyBullet has pre-built wheels for
Linux x86_64 and works out of the box.

**Step 1. Make sure Python 3.10 is installed.** On Ubuntu/Debian:

```bash
sudo apt update
sudo apt install -y python3.10 python3.10-venv python3.10-dev \
                    build-essential libgl1 libglib2.0-0 \
                    libsm6 libxext6 libxrender1 libgomp1
```

On Fedora / RHEL:

```bash
sudo dnf install -y python3.10 python3.10-devel gcc \
                    mesa-libGL glib2 libSM libXext libXrender libgomp
```

On Arch:

```bash
sudo pacman -S python python-pip mesa libsm libxext libxrender
# Arch ships Python 3.12+; for an exact 3.10, use pyenv or conda.
```

**Step 2. Create a venv + install requirements:**

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

**Step 3. Verify:**

```bash
python scripts/test_setup.py
```

**Linux gotchas:**

- **`ImportError: libGL.so.1: cannot open shared object file`** when
  OpenCV imports: you're missing `libgl1`. Install it as in Step 1.
- **Pygame black window or "Couldn't connect to display"** under
  Wayland. Quickest workaround: `export SDL_VIDEODRIVER=x11` before
  running `--visualize`. Permanent fix: add it to `.bashrc`/`.zshrc`.
- **Headless server (no display)**: use `--headless`. All scorers run
  fine without DISPLAY. `--visualize` needs X11 or Wayland.
- **WSL2 without WSLg**: `--visualize` won't work, `--headless` will.
  For the GUI install WSLg (default on Win11) or use an X server (VcXsrv).

### 1.1b Install (macOS)

Works on both Intel and Apple Silicon, but the recommended path
changes with architecture. On **Apple Silicon (M1/M2/M3/M4)** PyBullet
from PyPI has no ARM64 wheels and tries to compile from source: the
conda path is **recommended**.

**Step 1. Install Python + tooling.**

Path A — **Homebrew (Intel or ARM)**:

```bash
brew install python@3.10 cmake pkg-config
```

Path B — **Miniforge (ARM, recommended for Apple Silicon)**:

```bash
brew install miniforge
conda init zsh   # or bash, depending on your shell
# close and reopen the terminal
```

**Step 2. Create the environment.**

With Homebrew (Intel):

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

With Miniforge (Apple Silicon, recommended):

```bash
conda create -n catch-the-boat python=3.10 -y
conda activate catch-the-boat
conda install -c conda-forge pybullet -y
pip install -r requirements.txt
```

**Step 3. Verify:**

```bash
python scripts/test_setup.py
```

**macOS gotchas:**

- **`pip install pybullet` hangs for minutes on Apple Silicon** then
  fails: you're compiling from source without CMake/clang properly
  configured. Switch to the Miniforge path
  (`conda install -c conda-forge pybullet`).
- **`ImportError: cannot find OpenGL framework`** when running the
  visualizer: grant Screen Recording permissions to your terminal in
  Privacy & Security → Screen Recording.
- **OpenCV ArUco missing** (`AttributeError: module 'cv2' has no
  attribute 'aruco'`): you installed `opencv-python`, you need
  `opencv-contrib-python`. Uninstall the wrong one and reinstall the
  right one from `requirements.txt`.
- **Multiple Python installations** clashing (system Python +
  Homebrew + conda). Check with `which python` that it points where
  you expect.
- **Lagging display in `--gui` scenarios**: PyBullet's hardware
  renderer on macOS is less performant than on Linux/Windows. For long
  scenarios use `--headless` (3-5× slower than `--gui` but
  reproducible).

### 1.2 Install (Windows — MSVC Build Tools)

Native path that uses `pip` on a standard venv. Requires ~7 GB of
compiler and ~15 minutes for pybullet.

```powershell
winget install --id Microsoft.VisualStudio.2022.BuildTools --override "--passive --wait --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
py -3.10 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 1.3 Install (Windows — Miniconda, recommended)

Faster and compiler-free: `pybullet` comes from `conda-forge` as a
pre-built wheel. If you've already tried other paths and got stuck on
the PyBullet build, this is where you want to end up.

**Step 1. Install Miniconda** (skip if already installed):

```powershell
winget install --id Anaconda.Miniconda3
```

**Step 2. Enable execution policy + init conda for PowerShell**
(one-time per user):

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned -Force
& "$env:USERPROFILE\miniconda3\Scripts\conda.exe" init powershell
```

**Step 3. Close and reopen PowerShell.** The prompt must show `(base)`
at the start: conda is loaded.

**Step 4. Create the env, install pybullet from conda-forge, then the
rest via pip:**

```powershell
conda create -n catch-the-boat python=3.10 -y
conda activate catch-the-boat
conda install -c conda-forge pybullet -y
cd C:\path\to\catch-the-boat   # change to your actual path
pip install -r requirements.txt
```

**Step 5. Verify `python` points to the conda env** (not somewhere else):

```powershell
where.exe python
```

The **first** line must be
`C:\Users\<you>\miniconda3\envs\catch-the-boat\python.exe`. If you see
a different one first (e.g. `.venv\Scripts\python.exe` or a global
Python), deactivate that venv (`deactivate`) or delete it (see
[§1.5](#15-windows-gotchas)).

### 1.4 Verify the setup

```bash
python scripts/test_setup.py
```

Runs every check (Python, numpy, PyBullet, OpenCV+ArUco, PyYAML,
pygame, local imports, a 1-second rollout). Exits with code 0 if
everything is OK.

### 1.5 Windows gotchas

All of these are errors actually hit in practice — read them before
spending hours debugging.

- **`ModuleNotFoundError: No module named 'numpy'/'cv2'/...`** even
  after running `pip install`. Means `python` is pointing to a
  **different** environment than the one you installed into. Check
  with `where.exe python`: the first line must be the active env.
- **Prompt showing two stacked envs like `(base) (catch-the-boat)`**
  or `(catch-the-boat) (catch-the-boat)`. You activated both a venv
  and the conda env. The PATHs overlap and `python` can end up in the
  wrong venv. Fix: `deactivate` until the prompt is clean, then
  activate conda only.
- **`.venv` created by `uv` but empty** (no `pip.exe`, no packages).
  `uv venv` creates an environment without pip inside. If you have one
  and don't use it, delete it to avoid accidentally reactivating it:
  ```powershell
  Remove-Item -Recurse -Force .\.venv
  ```
- **`conda not recognized`** after installing Miniconda. You must have
  run `conda init powershell` AND **reopened** PowerShell. The session
  in which you ran the init will never see conda — you need a new
  terminal.
- **`script execution is disabled`**. PowerShell blocks the activation
  scripts. Permanent fix:
  `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned -Force`
- **PyBullet fails with `Microsoft Visual C++ 14.0 or greater is
  required`** during `pip install`. You're using the venv+pip path on
  Windows and PyBullet is trying to compile from source. Switch to the
  Miniconda path
  ([§1.3](#13-install-windows--miniconda-recommended)).

### Generate the ArUco marker (optional)

```bash
python scripts/generate_aruco.py
python scripts/generate_aruco.py --id 0 --size 600 --out boat_landing/assets/aruco
```

The env regenerates the marker automatically on `reset()` if missing;
this script is for pre-generating or printing it.

---

## 1.6 Daily startup (after a reboot)

The setup above is one-time. Once it works, every new working session
is just:

**On Windows (Miniconda):**

```powershell
conda activate catch-the-boat
cd C:\path\to\catch-the-boat
python scripts/run_baseline.py --scenario easy --visualize --gui
```

> ⚠️ Do NOT run `Activate.ps1` of a venv — only `conda activate`.
> If you see more than one `(...)` in front of the prompt, you have
> stacked envs (see [§1.5](#15-windows-gotchas)).

**On macOS Apple Silicon (Miniforge):**

```bash
conda activate catch-the-boat
cd /path/to/catch-the-boat
python scripts/run_baseline.py --scenario easy --visualize --gui
```

**On Linux / macOS Intel (venv):**

```bash
source .venv/bin/activate
cd /path/to/catch-the-boat
python scripts/run_baseline.py --scenario easy --visualize --gui
```

---

## 2. Running the baseline

### Headless (CPU, no visualization)

```bash
python scripts/run_baseline.py --scenario easy
```

### With the pygame visualizer

```bash
python scripts/run_baseline.py --scenario easy --visualize
```

Opens a pygame window with the chase view, the drone camera, and a HUD.

### With PyBullet's GPU renderer (3-5× faster)

```bash
python scripts/run_baseline.py --scenario easy --visualize --gui
```

### At wall-clock speed (good for recording demos)

```bash
python scripts/run_baseline.py --scenario easy --visualize --realtime
```

### With a fixed seed and step cap

```bash
python scripts/run_baseline.py --scenario medium --seed 42 --max-steps 3000
```

### Choosing the drone spec

The only shipped airframe is `drones/vtol.yaml`. `--drone` accepts
either the short name (`vtol`) or a full YAML path (e.g.
`drones/my_drone.yaml`). The spec is loaded by both the sim and the
baseline agent.

```bash
python scripts/run_baseline.py --scenario easy --drone vtol --visualize
```

### All public scenarios

```bash
python scripts/run_baseline.py --scenario easy
python scripts/run_baseline.py --scenario medium
python scripts/run_baseline.py --scenario hard
```

### Full argument list for `run_baseline.py`

| Flag           | Default                  | Description                                              |
| -------------- | ------------------------ | -------------------------------------------------------- |
| `--scenario`   | `easy`                   | Name (`easy`/`medium`/`hard`) or path to YAML            |
| `--drone`      | `drones/vtol.yaml`       | Name (`vtol`) or path to a drone YAML                    |
| `--visualize`  | off                      | Open the pygame viewer                                   |
| `--gui`        | off                      | Open the PyBullet window (OpenGL/GPU renderer)           |
| `--realtime`   | off                      | Sleep to run at wall-clock speed (with `--visualize` only) |
| `--seed`       | random                   | Env RNG seed                                              |
| `--max-steps`  | none                     | Max step cap                                              |

---

## 3. Evaluation

There are **two independent scorers**, one for each half of the
automated score (the HW track rubric is judged by hand by the
organizers):

| CLI | Purpose | Score |
| --- | --- | --- |
| `evaluation/evaluate.py`    | Runs the agent on one scenario and computes the **landing** score + HW-readiness bonuses (FC-compat, latency, recovery) | 0–115 |
| `evaluation/sim_scorer.py`  | Runs the validation suite (Tier-0 gate + Tier-1 features) against your **simulator** | 0–30 |

### 3.1 Agent score (`evaluate.py`)

Dynamically loads agent + drone_sim + drone_spec, runs the episode,
prints a JSON with score and breakdown.

**Defaults:** `agents/agent_baseline.py` + `agents/drone_sim_baseline.py`
+ `drones/vtol.yaml`. All three flags are optional — the only
required one is `--scenario`.

#### Standard evaluation (headless, recommended for CI/A-B testing)

```bash
python evaluation/evaluate.py --scenario easy --headless --seed 42
```

#### Evaluating your agent + drone sim

```bash
python evaluation/evaluate.py \
    --agent      teams/myteam/agent.py     \
    --drone-sim  teams/myteam/drone_sim.py \
    --drone      vtol                      \
    --scenario   medium --headless --seed 42
```

The agent must expose `make_agent(drone_spec)` or an `Agent` class.
The drone sim must expose `make_drone_sim(spec_path)` or a `DroneSim`
class. See [`API.md`](API.md) for the full contract.

#### Evaluation with the PyBullet GUI

```bash
python evaluation/evaluate.py --scenario easy --gui --seed 42
```

#### Saving the result to a file

```bash
python evaluation/evaluate.py --scenario hard --headless --seed 42 \
                              --output results/hard_seed42.json
```

#### Saving the telemetry (for attitude / safety analysis)

```bash
python evaluation/evaluate.py --scenario easy --headless \
                              --save-traj logs/baseline_flight.json
```

The saved JSON contains position and attitude (roll, pitch, yaw) at
every timestep.

#### Telemetry analysis (judging)

After saving a trajectory, you can run the analysis script to generate
plots and check the safety limits (e.g. max tilt):

```bash
python scripts/analyze_telemetry.py logs/baseline_flight.json --save-plot logs/baseline_plot.png
```

The script reports:
- **Max tilt:** warns if it exceeds 45° (potentially dangerous /
  unrealistic manoeuvre).
- **Descent velocity:** to assess touchdown softness.
- **Plots:** writes a PNG with the attitude and altitude trace.

#### Quiet mode (no progress on stderr)

```bash
python evaluation/evaluate.py --scenario medium --headless --quiet
```

#### Full argument list for `evaluate.py`

| Flag          | Default                            | Description                                          |
| ------------- | ---------------------------------- | ---------------------------------------------------- |
| `--agent`     | `agents/agent_baseline.py`         | Path to the agent `.py` file                         |
| `--drone-sim` | `agents/drone_sim_baseline.py`     | Path to the drone simulator `.py` file               |
| `--drone`     | `drones/vtol.yaml`                 | Name (`vtol`) or path to a drone YAML                |
| `--scenario`  | (required)                         | Name (`easy`/`medium`/`hard`) or path to YAML        |
| `--seed`      | from scenario YAML                 | Env RNG seed                                         |
| `--headless`  | off                                | Disable the GUI (recommended for batch)              |
| `--gui`       | off                                | Open the PyBullet GUI (mutually exclusive with headless) |
| `--quiet`     | off                                | Suppress per-step progress on stderr                 |
| `--output`    | none                               | Optional path to write the JSON result               |
| `--save-traj` | none                               | Optional path to write the telemetry log             |

### 3.2 Simulator score (`sim_scorer.py`)

Runs the `evaluation/sim_validation/` suite against your simulator: 4
gate tests (Tier 0, mandatory) + 6 auto-tested feature tests (Tier 1,
5 pts each). Reads `submission.yaml` to know which Tier 1 features you
claim to implement. Prints a JSON breakdown with pass/fail and
metrics for each test.

See [`SIM_SCORING.md`](SIM_SCORING.md) for the full rubric.

#### Baseline score (expected: 15/30)

```bash
python evaluation/sim_scorer.py \
    --drone-sim   agents/drone_sim_baseline.py \
    --drone       vtol \
    --submission  evaluation/submission_baseline.yaml
```

#### Scoring your submission

```bash
python evaluation/sim_scorer.py \
    --drone-sim   teams/myteam/drone_sim.py     \
    --drone       vtol                          \
    --submission  teams/myteam/submission.yaml  \
    --output      results/sim_score.json
```

#### Full argument list for `sim_scorer.py`

| Flag          | Required | Description                                                       |
| ------------- | -------- | ----------------------------------------------------------------- |
| `--drone-sim` | yes      | Path to the drone simulator `.py` file                            |
| `--drone`     | no       | Name (`vtol`) or path to a drone YAML; defaults to `vtol`         |
| `--submission`| yes      | Path to the `submission.yaml` declaring the Tier 1 features       |
| `--output`    | no       | Optional path to write the JSON breakdown                         |

#### Full submission: three files

A submission is made of three files:

```
teams/myteam/
├── drone_sim.py      # implements DroneSimulator (Protocol)
├── agent.py          # implements Agent
└── submission.yaml   # declares Tier 1 features + paths
```

The manifest template is at
[`evaluation/submission.yaml.template`](../evaluation/submission.yaml.template).
Detailed schema + anti-bluff rules in
[`SIM_SCORING.md`](SIM_SCORING.md#submission-manifest).

---

## 4. Test suite

```bash
# Full suite (env + baseline agent + scoring + sim_validation)
pytest

# A single file
pytest tests/test_baseline.py

# A single test
pytest tests/test_baseline.py::test_act_returns_motor_throttles_in_unit_interval

# Only the simulator-validation meta-tests
pytest tests/test_sim_validation.py -v

# Only the tests for a single Tier 1 feature
pytest tests/test_sim_validation.py -k motor_lag -v
pytest tests/test_sim_validation.py -k cross_coupling -v

# Verbose with stdout enabled
pytest -v -s

# With coverage (if installed)
pytest --cov=boat_landing --cov=agents --cov=evaluation
```

`tests/test_sim_validation.py` verifies that the baseline sim meets
the expected outcomes: all 4 Tier 0 gates PASS, T1.A motor lag and
T1.D cross-coupling PASS, the other 4 Tier 1 features FAIL (the
baseline doesn't implement them). When developing your sim, run these
tests to confirm the features you want to claim in `submission.yaml`
actually pass.

---

## 5. Debug scripts

Use these when the baseline doesn't land or your custom agent diverges.

### Per-step trace of state + ArUco detection

```bash
python scripts/debug_baseline.py
```

Saves `first_frame.png` in the repo root and prints drone position,
marker estimate, and controller phase every 0.5 s of simulated time.

### Render the marker at various altitudes

```bash
python scripts/debug_marker_render.py
```

Saves `render_alt_1m.png`, `render_alt_2m.png`, etc. and reports
whether the detector finds the marker at each altitude. Useful to
verify the texture is correctly oriented.

### Hover / control trace

```bash
python scripts/debug_hover.py
```

Prints position, velocity, RPY, and baseline action every simulated
second. Use it to pin down when the controller diverges.

---

## 6. Evaluation batches (examples)

### Evaluate the baseline on every scenario (PowerShell)

```powershell
foreach ($s in 'easy','medium','hard') {
    python evaluation/evaluate.py --scenario $s --headless --seed 42 `
                                  --output "results/$s.json"
}
```

### Evaluate the baseline on every scenario (bash)

```bash
for s in easy medium hard; do
    python evaluation/evaluate.py --scenario "$s" --headless --seed 42 \
                                  --output "results/$s.json"
done
```

### Seed sweep on one scenario

```bash
for seed in 0 1 2 3 4; do
    python evaluation/evaluate.py --scenario medium --headless --seed "$seed" \
                                  --quiet --output "results/medium_$seed.json"
done
```

---

## 7. Typical workflows

### Agent development (edit → test → run)

```bash
pytest tests/test_baseline.py -q
python scripts/run_baseline.py --scenario easy --visualize --gui
python evaluation/evaluate.py \
    --agent     teams/myteam/agent.py     \
    --drone-sim teams/myteam/drone_sim.py \
    --scenario  easy --headless --seed 42
```

### Simulator development (iterating on one Tier 1 feature)

```bash
# Pick a feature to implement (e.g. ground_effect)
# 1. Edit drone_sim.py
# 2. Run the single test in isolation (fast)
pytest tests/test_sim_validation.py -k ground_effect -v

# 3. When it passes, declare it in submission.yaml and run the full scorer
python evaluation/sim_scorer.py \
    --drone-sim   teams/myteam/drone_sim.py    \
    --drone       vtol                         \
    --submission  teams/myteam/submission.yaml
```

### Live demo

```bash
python scripts/run_baseline.py --scenario medium --visualize --gui --realtime
```

### Submission check (what the judges do)

```bash
# 1) Agent score on the three public scenarios
for s in easy medium hard; do
    python evaluation/evaluate.py \
        --agent      teams/myteam/agent.py     \
        --drone-sim  teams/myteam/drone_sim.py \
        --drone      vtol                      \
        --scenario   "$s" --headless --seed 1
done

# 2) Simulator score (Tier 0 gate + declared Tier 1)
python evaluation/sim_scorer.py \
    --drone-sim   teams/myteam/drone_sim.py     \
    --drone       vtol                          \
    --submission  teams/myteam/submission.yaml

# Total = sum of the 4 scores (3 agent scenarios + 1 sim).
```
