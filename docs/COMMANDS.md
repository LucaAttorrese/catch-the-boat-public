# Comandi per lanciare il codice

Riferimento rapido di tutti i comandi disponibili per installare, eseguire,
debuggare e valutare l'agente. I percorsi sono relativi alla cartella
`catch-the-boat/`.

> **Nota.** Tutti i comandi presuppongono che l'environment Python sia
> attivo e che si usi Python 3.10. La forma di "attivare l'environment"
> dipende dall'OS e dal path di install scelto:
>
> - **Linux** (venv): `source .venv/bin/activate`
> - **macOS Intel** (venv): `source .venv/bin/activate`
> - **macOS Apple Silicon** (Miniforge, raccomandato): `conda activate catch-the-boat`
> - **Windows** (Miniconda, raccomandato): `conda activate catch-the-boat`
>
> **Path con wheel pre-compilati di PyBullet** (no compilatore necessario):
> Linux pip wheels; macOS Intel pip wheels; macOS ARM **solo via conda-forge**;
> Windows **solo via conda-forge**. Vedi le sezioni 1.1–1.3 per i dettagli OS-by-OS.

---

## 1. Setup iniziale

### 1.1 Installazione (Linux)

Path standard: venv + pip. PyBullet ha wheel pre-compilati su Linux x86_64
e funziona out of the box.

**Step 1. Assicurati di avere Python 3.10.** Su Ubuntu/Debian:

```bash
sudo apt update
sudo apt install -y python3.10 python3.10-venv python3.10-dev \
                    build-essential libgl1 libglib2.0-0 \
                    libsm6 libxext6 libxrender1 libgomp1
```

Su Fedora / RHEL:

```bash
sudo dnf install -y python3.10 python3.10-devel gcc \
                    mesa-libGL glib2 libSM libXext libXrender libgomp
```

Su Arch:

```bash
sudo pacman -S python python-pip mesa libsm libxext libxrender
# Arch ships Python 3.12+; se serve 3.10 esatto, usa pyenv o conda.
```

**Step 2. Crea venv + installa requirements:**

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

**Step 3. Verifica:**

```bash
python scripts/test_setup.py
```

**Gotcha Linux:**

- **`ImportError: libGL.so.1: cannot open shared object file`** quando
  OpenCV importa: ti manca `libgl1`. Installalo come da Step 1.
- **Pygame finestra nera o "Couldn't connect to display"** sotto
  Wayland. Workaround più rapido: `export SDL_VIDEODRIVER=x11` prima
  di lanciare `--visualize`. Permanente: aggiungilo a `.bashrc`/`.zshrc`.
- **Headless server (no display)**: usa `--headless`. Tutti gli
  scorer girano bene senza DISPLAY. `--visualize` richiede X11 o Wayland.
- **WSL2 senza WSLg**: `--visualize` non funziona, `--headless` sì.
  Per la GUI installa WSLg (default su Win11) o usa un X server (VcXsrv).

### 1.1b Installazione (macOS)

Funziona sia su Intel che Apple Silicon, ma il path consigliato cambia
in base all'architettura. Su **Apple Silicon (M1/M2/M3/M4)** PyBullet
da PyPI non ha wheel ARM64 e tenta di compilarsi: il path conda è
**raccomandato**.

**Step 1. Installa Python + tooling.**

Path A — **Homebrew (Intel o ARM)**:

```bash
brew install python@3.10 cmake pkg-config
```

Path B — **Miniforge (ARM, raccomandato per Apple Silicon)**:

```bash
brew install miniforge
conda init zsh   # o bash, in base alla tua shell
# chiudi e riapri il terminale
```

**Step 2. Crea l'environment.**

Con Homebrew (Intel):

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

Con Miniforge (Apple Silicon, raccomandato):

```bash
conda create -n catch-the-boat python=3.10 -y
conda activate catch-the-boat
conda install -c conda-forge pybullet -y
pip install -r requirements.txt
```

**Step 3. Verifica:**

```bash
python scripts/test_setup.py
```

**Gotcha macOS:**

- **`pip install pybullet` blocca per minuti su Apple Silicon** poi
  fallisce: stai compilando da source senza CMake/clang configurati
  bene. Passa al path Miniforge (`conda install -c conda-forge pybullet`).
- **`ImportError: cannot find OpenGL framework`** lanciando lo
  visualizer: aggiungi le permissions Screen Recording al tuo
  terminale in Privacy & Security → Screen Recording.
- **OpenCV ArUco mancante** (`AttributeError: module 'cv2' has no
  attribute 'aruco'`): hai installato `opencv-python`, ti serve
  `opencv-contrib-python`. Disinstalla quello sbagliato e reinstalla
  il giusto dal `requirements.txt`.
- **Multiple Python installations** che si scontrano (system Python +
  Homebrew + conda). Verifica con `which python` che punti dove credi.
- **Display ritardato su scenari `--gui`**: PyBullet's hardware
  renderer su macOS è meno performante che su Linux/Windows. Per
  scenari lunghi usa `--headless` (3-5× più lento del `--gui` ma
  riproducibile).

### 1.2 Installazione (Windows — MSVC Build Tools)

Path nativo che usa `pip` su un venv standard. Richiede ~7 GB di compilatore
e ~15 minuti per pybullet.

```powershell
winget install --id Microsoft.VisualStudio.2022.BuildTools --override "--passive --wait --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
py -3.10 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 1.3 Installazione (Windows — Miniconda, raccomandato)

Più veloce e niente compilatore: `pybullet` arriva da `conda-forge` come
wheel pre-compilato. Se hai già provato altri path e ti sei bloccato sul
build di PyBullet, è qui che vuoi finire.

**Step 1. Installa Miniconda** (skip se già installato):

```powershell
winget install --id Anaconda.Miniconda3
```

**Step 2. Abilita execution policy + inizializza conda per PowerShell**
(una tantum per il tuo utente):

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned -Force
& "$env:USERPROFILE\miniconda3\Scripts\conda.exe" init powershell
```

**Step 3. Chiudi e riapri PowerShell.** Il prompt deve mostrare `(base)`
all'inizio: vuol dire che conda è caricato.

**Step 4. Crea l'env, installa pybullet da conda-forge, poi il resto via pip:**

```powershell
conda create -n catch-the-boat python=3.10 -y
conda activate catch-the-boat
conda install -c conda-forge pybullet -y
cd C:\Users\nicco\Documents\GitHub\Hackathlon_Milan\catch-the-boat #cambia con il tuo percorso
pip install -r requirements.txt
```

**Step 5. Verifica che `python` punti all'env conda** (non a un altro):

```powershell
where.exe python
```

La **prima** riga deve essere
`C:\Users\nicco\miniconda3\envs\catch-the-boat\python.exe`. Se ne vedi
prima un'altra (es. `.venv\Scripts\python.exe` o un Python globale),
disattiva quel venv (`deactivate`) o cancellalo (vedi
[§1.5](#15-gotcha-windows)).

### 1.4 Verifica setup

```bash
python scripts/test_setup.py
```

Esegue tutti i check (Python, numpy, PyBullet, OpenCV+ArUco, PyYAML,
pygame, import locali, rollout di 1 secondo). Esce con codice 0 se tutto
ok.

### 1.5 Gotcha Windows

Tutti errori che ho effettivamente incontrato — leggi prima di passare ore
a debuggarli.

- **`ModuleNotFoundError: No module named 'numpy'/'cv2'/...`** anche dopo
  aver eseguito `pip install`. Significa che `python` sta puntando a un
  environment **diverso** da quello in cui hai installato. Controlla con
  `where.exe python`: la prima riga deve essere l'env attivo.
- **Prompt che mostra due env stacked tipo `(base) (catch-the-boat)`** o
  `(catch-the-boat) (catch-the-boat)`. Hai attivato sia un venv che
  l'env conda. Le PATH si sovrappongono e `python` può finire sul venv
  sbagliato. Soluzione: `deactivate` finché il prompt è pulito, poi
  attiva solo conda.
- **Venv `.venv` creato da `uv` ma vuoto** (no `pip.exe`, no pacchetti).
  `uv venv` crea environment senza pip dentro. Se ce l'hai e non lo usi,
  cancellalo per evitare di riattivarlo per sbaglio:
  ```powershell
  Remove-Item -Recurse -Force .\.venv
  ```
- **`conda non riconosciuto`** dopo aver installato Miniconda. Devi aver
  fatto `conda init powershell` E **riaperto** PowerShell. La sessione
  in cui hai lanciato l'init non vedrà mai conda — serve un terminale
  nuovo.
- **`L'esecuzione di script è disabilitata`**. PowerShell blocca gli
  script di attivazione. Fix permanente:
  `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned -Force`
- **PyBullet fallisce con `Microsoft Visual C++ 14.0 or greater is
  required`** durante `pip install`. Stai usando il path venv+pip su
  Windows, e PyBullet sta tentando di compilarsi da sorgente. Passa al
  path Miniconda ([§1.3](#13-installazione-windows--miniconda-raccomandato)).

### Generare il marker ArUco (opzionale)

```bash
python scripts/generate_aruco.py
python scripts/generate_aruco.py --id 0 --size 600 --out boat_landing/assets/aruco
```

L'env rigenera il marker in automatico al `reset()` se manca; questo
script serve per pre-generarlo o stamparlo.

---

## 1.6 Avvio quotidiano (dopo aver chiuso/riavviato il PC)

Il setup è una tantum. Una volta che funziona, ogni nuova sessione di
lavoro è solo:

**Su Windows (Miniconda):**

```powershell
conda activate catch-the-boat
cd C:\Users\nicco\Documents\GitHub\Hackathlon_Milan\catch-the-boat
python scripts/run_baseline.py --scenario easy --visualize --gui
```

> ⚠️ NON lanciare `Activate.ps1` di un venv — solo `conda activate`.
> Se vedi più di un `(...)` davanti al prompt, hai env sovrapposti
> (vedi [§1.5](#15-gotcha-windows)).

**Su macOS Apple Silicon (Miniforge):**

```bash
conda activate catch-the-boat
cd /path/to/catch-the-boat
python scripts/run_baseline.py --scenario easy --visualize --gui
```

**Su Linux / macOS Intel (venv):**

```bash
source .venv/bin/activate
cd /path/to/catch-the-boat
python scripts/run_baseline.py --scenario easy --visualize --gui
```

---

## 2. Esecuzione del baseline

### Headless (CPU, senza visualizzazione)

```bash
python scripts/run_baseline.py --scenario easy
```

### Con il visualizer pygame

```bash
python scripts/run_baseline.py --scenario easy --visualize
```

Apre una finestra pygame con chase view, camera del drone e HUD.

### Con il renderer GPU di PyBullet (3-5x più veloce)

```bash
python scripts/run_baseline.py --scenario easy --visualize --gui
```

### A velocità wall-clock (per registrare demo)

```bash
python scripts/run_baseline.py --scenario easy --visualize --realtime
```

### Con seed e cap di step

```bash
python scripts/run_baseline.py --scenario medium --seed 42 --max-steps 3000
```

### Cambiare drone (quad ↔ VTOL)

Default = `quadcopter`. Per girare il VTOL pesante (yaw debole) basta:

```bash
python scripts/run_baseline.py --scenario easy --drone vtol --visualize
```

`--drone` accetta sia il nome breve (`quadcopter` / `vtol`) sia un path
YAML completo (es. `drones/mio_drone.yaml`). Lo spec viene caricato dal
sim e dall'agent baseline contemporaneamente.

### Tutti gli scenari pubblici

```bash
python scripts/run_baseline.py --scenario easy
python scripts/run_baseline.py --scenario medium
python scripts/run_baseline.py --scenario hard
```

### Argomenti completi di `run_baseline.py`

| Flag           | Default        | Descrizione                                              |
| -------------- | -------------- | -------------------------------------------------------- |
| `--scenario`   | `easy`         | Nome (`easy`/`medium`/`hard`) o path YAML                |
| `--drone`      | `quadcopter`   | Nome (`quadcopter`/`vtol`) o path al drone YAML          |
| `--visualize`  | off            | Apre il viewer pygame                                    |
| `--gui`        | off            | Apre la finestra PyBullet (renderer OpenGL/GPU)          |
| `--realtime`   | off            | Sleep per girare a velocità reale (solo con `--visualize`) |
| `--seed`       | random         | Seed RNG dell'env                                         |
| `--max-steps`  | nessuno        | Tetto massimo di step                                     |

---

## 3. Valutazione

Ci sono **due scorer indipendenti**, uno per ogni metà del punteggio
totale (la rubrica HW track è giudicata a mano dagli organizzatori):

| CLI | Scopo | Punteggio |
| --- | --- | --- |
| `evaluation/evaluate.py`    | Esegue l'agente su uno scenario e calcola il punteggio di **landing** + i bonus HW-readiness (FC-compat, latenza, recovery) | 0–145 |
| `evaluation/sim_scorer.py`  | Esegue la suite di validazione (Tier-0 gate + Tier-1 features) sul tuo **simulatore** | 0–30 |

### 3.1 Score dell'agente (`evaluate.py`)

Carica dinamicamente agent + drone_sim + drone_spec, esegue l'episodio e
stampa un JSON con score e breakdown.

**Default:** `agents/agent_baseline.py` + `agents/drone_sim_baseline.py`
+ `drones/quadcopter.yaml`. Tutti e tre i flag sono opzionali — devi
specificare obbligatoriamente solo `--scenario`.

#### Valutazione standard (headless, raccomandata per CI/A-B test)

```bash
python evaluation/evaluate.py --scenario easy --headless --seed 42
```

#### Valutare il tuo agent + drone sim

```bash
python evaluation/evaluate.py \
    --agent      teams/myteam/agent.py     \
    --drone-sim  teams/myteam/drone_sim.py \
    --drone      vtol                      \
    --scenario   medium --headless --seed 42
```

L'agent deve esporre `make_agent(drone_spec)` o una classe `Agent`.
Il drone sim deve esporre `make_drone_sim(spec_path)` o una classe
`DroneSim`. Vedi [`API.md`](API.md) per il contratto completo.

#### Valutazione con GUI PyBullet

```bash
python evaluation/evaluate.py --scenario easy --gui --seed 42
```

#### Salvare il risultato su file

```bash
python evaluation/evaluate.py --scenario hard --headless --seed 42 \
                              --output results/hard_seed42.json
```

#### Salvare la telemetria (per analisi assetto/sicurezza)

```bash
python evaluation/evaluate.py --scenario easy --headless \
                              --save-traj logs/volo_baseline.json
```

Il file JSON salvato contiene posizione e assetto (roll, pitch, yaw)
per ogni istante.

#### Analisi della Telemetria (Judging)

Dopo aver salvato una traiettoria, puoi usare lo script di analisi per
generare grafici e verificare il rispetto dei limiti di sicurezza (es.
inclinazione massima):

```bash
python scripts/analyze_telemetry.py logs/volo_baseline.json --save-plot logs/plot_baseline.png
```

Lo script calcola:
- **Inclinazione massima (Tilt):** se supera i 45° segnala un warning
  (manovra potenzialmente pericolosa/irrealistica).
- **Velocità di discesa:** per valutare la dolcezza del tocco.
- **Grafici:** genera un file PNG con l'andamento di assetto e altitudine.

#### Modalità silenziosa (senza progress su stderr)

```bash
python evaluation/evaluate.py --scenario medium --headless --quiet
```

#### Argomenti completi di `evaluate.py`

| Flag          | Default                            | Descrizione                                          |
| ------------- | ---------------------------------- | ---------------------------------------------------- |
| `--agent`     | `agents/agent_baseline.py`         | Path al file `.py` dell'agente                       |
| `--drone-sim` | `agents/drone_sim_baseline.py`     | Path al file `.py` del drone simulator               |
| `--drone`     | `drones/quadcopter.yaml`           | Nome (`quadcopter`/`vtol`) o path al drone YAML      |
| `--scenario`  | (richiesto)                        | Nome (`easy`/`medium`/`hard`) o path YAML            |
| `--seed`      | dal YAML scenario                  | Seed RNG dell'env                                    |
| `--headless`  | off                                | Disabilita la GUI (consigliato per batch)            |
| `--gui`       | off                                | Apre PyBullet GUI (mutuamente esclusivo con headless)|
| `--quiet`     | off                                | Sopprime le righe di progress su stderr              |
| `--output`    | nessuno                            | Path opzionale per salvare il JSON                   |
| `--save-traj` | nessuno                            | Path opzionale per salvare il log di telemetria      |

### 3.2 Score del simulatore (`sim_scorer.py`)

Esegue la suite `evaluation/sim_validation/` sul tuo simulatore: 4
test gate (Tier 0, mandatory) + 6 test feature auto-testate (Tier 1, 5
pt ciascuno). Legge `submission.yaml` per sapere quali Tier 1
dichiari di implementare. Stampa un JSON breakdown con pass/fail e
metriche per ogni test.

Vedi [`SIM_SCORING.md`](SIM_SCORING.md) per la rubrica completa.

#### Score del baseline (atteso: 15/30)

```bash
python evaluation/sim_scorer.py \
    --drone-sim   agents/drone_sim_baseline.py \
    --drone       quadcopter \
    --submission  evaluation/submission_baseline.yaml
```

#### Score della tua submission

```bash
python evaluation/sim_scorer.py \
    --drone-sim   teams/myteam/drone_sim.py     \
    --drone       quadcopter                    \
    --submission  teams/myteam/submission.yaml  \
    --output      results/sim_score.json
```

Tutti e tre i flag sono **richiesti**.

#### Argomenti completi di `sim_scorer.py`

| Flag          | Required | Descrizione                                                       |
| ------------- | -------- | ----------------------------------------------------------------- |
| `--drone-sim` | sì       | Path al file `.py` del drone simulator                            |
| `--drone`     | sì       | Nome (`quadcopter`/`vtol`) o path al drone YAML                   |
| `--submission`| sì       | Path al `submission.yaml` con la dichiarazione delle feature      |
| `--output`    | no       | Path opzionale per salvare il JSON breakdown                      |

#### Submission completa: tre file

Una submission è composta da tre file:

```
teams/myteam/
├── drone_sim.py      # implementa DroneSimulator (Protocol)
├── agent.py          # implementa Agent
└── submission.yaml   # dichiara chosen_drone + Tier 1 features + paths
```

Il template del manifest è in
[`evaluation/submission.yaml.template`](../evaluation/submission.yaml.template).
Schema dettagliato + regole anti-bluff in
[`SIM_SCORING.md`](SIM_SCORING.md#submission-manifest).

---

## 4. Test suite

```bash
# Tutta la suite (env + agent baseline + scoring + sim_validation)
pytest

# Un singolo file
pytest tests/test_baseline.py

# Un singolo test
pytest tests/test_baseline.py::test_act_returns_motor_throttles_in_unit_interval

# Solo i meta-test della validation suite del simulatore
pytest tests/test_sim_validation.py -v

# Solo i test che riguardano una singola feature Tier 1
pytest tests/test_sim_validation.py -k motor_lag -v
pytest tests/test_sim_validation.py -k cross_coupling -v

# Verbose con stdout abilitato
pytest -v -s

# Con coverage (se installato)
pytest --cov=boat_landing --cov=agents --cov=evaluation
```

`tests/test_sim_validation.py` verifica che il baseline sim soddisfi
gli outcome attesi: tutti i 4 gate Tier 0 in PASS, T1.A motor lag e
T1.D cross-coupling in PASS, le altre 4 feature Tier 1 in FAIL
(perché il baseline non le implementa). Quando svilupp il tuo sim,
lancia questi test per controllare che le feature che vuoi dichiarare
in `submission.yaml` passino davvero.

---

## 5. Script di debug

Da usare quando il baseline non atterra o l'agente custom diverge.

### Trace per-step di stato + detection ArUco

```bash
python scripts/debug_baseline.py
```

Salva `first_frame.png` nella root e stampa ogni 0.5s sim posizione del
drone, stima del marker e fase del controller.

### Render del marker a varie altitudini

```bash
python scripts/debug_marker_render.py
```

Salva `render_alt_1m.png`, `render_alt_2m.png`, ecc. e segnala se il
detector trova il marker a ogni altitudine. Utile per verificare che la
texture sia orientata correttamente.

### Trace di hover/control

```bash
python scripts/debug_hover.py
```

Stampa ogni secondo simulato posizione, velocità, RPY e azione del
baseline. Usalo per individuare il momento in cui il controller diverge.

---

## 6. Batch di valutazione (esempi)

### Valutare il baseline su tutti gli scenari (PowerShell)

```powershell
foreach ($s in 'easy','medium','hard') {
    python evaluation/evaluate.py --scenario $s --headless --seed 42 `
                                  --output "results/$s.json"
}
```

### Valutare il baseline su tutti gli scenari (bash)

```bash
for s in easy medium hard; do
    python evaluation/evaluate.py --scenario "$s" --headless --seed 42 \
                                  --output "results/$s.json"
done
```

### Comparare quad vs VTOL sullo stesso scenario

```bash
for d in quadcopter vtol; do
    python evaluation/evaluate.py --drone "$d" --scenario medium \
                                  --headless --seed 42 \
                                  --output "results/medium_$d.json"
done
```

### Sweep di seed su uno scenario

```bash
for seed in 0 1 2 3 4; do
    python evaluation/evaluate.py --scenario medium --headless --seed "$seed" \
                                  --quiet --output "results/medium_$seed.json"
done
```

---

## 7. Workflow tipici

### Sviluppo agente (modifica → test → run)

```bash
pytest tests/test_baseline.py -q
python scripts/run_baseline.py --scenario easy --visualize --gui
python evaluation/evaluate.py \
    --agent     teams/myteam/agent.py     \
    --drone-sim teams/myteam/drone_sim.py \
    --scenario  easy --headless --seed 42
```

### Sviluppo simulatore (iterare su una feature Tier 1)

```bash
# Scegli una feature da implementare (es. ground_effect)
# 1. Edita drone_sim.py
# 2. Lancia il singolo test in isolamento (veloce)
pytest tests/test_sim_validation.py -k ground_effect -v

# 3. Quando passa, dichiarala in submission.yaml e lancia lo scorer completo
python evaluation/sim_scorer.py \
    --drone-sim   teams/myteam/drone_sim.py    \
    --drone       quadcopter                   \
    --submission  teams/myteam/submission.yaml
```

### Demo dal vivo

```bash
python scripts/run_baseline.py --scenario medium --visualize --gui --realtime
```

### Submission check (quello che fanno i giudici)

```bash
# 1) Score dell'agente sui tre scenari pubblici
for s in easy medium hard; do
    python evaluation/evaluate.py \
        --agent      teams/myteam/agent.py     \
        --drone-sim  teams/myteam/drone_sim.py \
        --drone      $(yq '.chosen_drone' teams/myteam/submission.yaml) \
        --scenario   "$s" --headless --seed 1
done

# 2) Score del simulatore (Tier 0 gate + Tier 1 dichiarate)
python evaluation/sim_scorer.py \
    --drone-sim   teams/myteam/drone_sim.py     \
    --drone       quadcopter                    \
    --submission  teams/myteam/submission.yaml

# Punteggio totale = somma dei 4 score (3 scenari agent + 1 sim).
```
