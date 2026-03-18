# `scripts/` — Experiment Runner Reference

This directory contains everything needed to run the full parameter sweeps
and write results to HDF5 files in `data/`. 

---

## Directory layout

```
scripts/
├── _runner.py              Shared parallel runner (not called directly)
├── generate_logistic.py    Logistic map sweep
├── generate_henon.py       Hénon map sweep
├── generate_rossler.py     Rössler system sweep
├── run_all.sh              Orchestrates all three scripts sequentially
└── README.md               (this file)
```

---

## Quick start

```bash
# 1. Activate your environment
conda activate env_name

# 2. Smoke test — completes in seconds, verifies the whole pipeline
./scripts/run_all.sh --quick

# 3. Production run — all three maps, all L values
./scripts/run_all.sh --workers 16
```

All output goes to `data/` by default.  Override with `--outdir /path/to/dir`.

---

## Running scripts individually

Each generate script is fully self-contained and can be run independently.
This is useful when you want to re-run just one map, or experiment with
different parameters without touching the others.

```bash
# Logistic map only
python scripts/generate_logistic.py \
    --n-param 500 \
    --L 100000 1000000 \
    --D 3 5 7 \
    --bins 2 3 4 5 \
    --noise 0.0 0.01 0.05 0.1 \
    --workers 8

# Hénon map only
python scripts/generate_henon.py \
    --n-param 500 \
    --L 100000 1000000 \
    --workers 8

# Rössler system only
python scripts/generate_rossler.py \
    --n-param 500 \
    --L 10000 100000 \
    --workers 8
```

Every script has a `--help` flag that prints all available options with defaults.

---

## Parameters reference

### Shared across all three scripts

| Flag | Default | Description |
|---|---|---|
| `--n-param` | 500 | Number of parameter values in the sweep |
| `--L` | (map-specific) | Series length(s); accepts multiple values |
| `--D` | `3 5 7` | PE embedding dimensions |
| `--bins` | `2 3 4 5` | ETC bin counts |
| `--noise` | `0.0 0.01 0.05 0.1` | Measurement noise σ values |
| `--workers` | all cores | Parallel worker processes |
| `--outdir` | `data/` | Output directory for HDF5 files |
| `--seed` | `42` | RNG seed for initial conditions |

### Logistic-specific

| Flag | Default | Description |
|---|---|---|
| `--a-min` | `3.5` | Left edge of `a` sweep |
| `--a-max` | `4.0` | Right edge of `a` sweep |
| `--transient` | `1000` | Burn-in iterates discarded before sampling |

### Hénon-specific

| Flag | Default | Description |
|---|---|---|
| `--a-min` | `0.8` | Left edge of `a` sweep |
| `--a-max` | `1.4` | Right edge of `a` sweep |
| `--b-fixed` | `0.3` | Fixed value of `b` (area contraction) |
| `--transient` | `1000` | Burn-in iterates |

### Rössler-specific

| Flag | Default | Description |
|---|---|---|
| `--c-min` | `2.0` | Left edge of `c` sweep |
| `--c-max` | `8.0` | Right edge of `c` sweep |
| `--a-fixed` | `0.2` | Fixed `a` parameter |
| `--b-fixed` | `0.2` | Fixed `b` parameter |
| `--dt` | `0.1` | Sampling interval (time units) |
| `--transient-time` | `500.0` | Burn-in time (time units, not iterations) |

---

## Parallelism design

All three generate scripts use the same parallel runner in `_runner.py`.

### Unit of parallelism

The parameter sweep axis is parallelised — each worker handles one parameter
value independently.  Noise levels are looped sequentially in the main
process; for each noise level, N tasks (one per parameter value) are
dispatched to the pool.

This means:
- `--workers 16` with `--n-param 500` → 16 workers process 500 tasks per batch.
- Peak live memory = `n_workers × L × 8 bytes` (one `float64` series per worker).

### Memory guidance

| L | Series size | `n_workers=8` peak RAM | `n_workers=16` peak RAM |
|---|---|---|---|
| 10⁵ | 0.8 MB | ~6 MB | ~13 MB |
| 10⁶ | 8 MB | ~64 MB | ~128 MB |
| 10⁷ | 80 MB | ~640 MB | ~1.3 GB |
| 10⁸ | 800 MB | ~6.4 GB | ~12.8 GB |

---

## Output files

Each script writes one HDF5 file per `L` value:

```
data/
  logistic_L100000.h5
  logistic_L1000000.h5
  henon_L100000.h5
  henon_L1000000.h5
  rossler_L10000.h5
  rossler_L100000.h5
```

Log files are written alongside the data:

```
data/
  logistic_20250318_143022.log
  henon_20250318_153501.log
  rossler_20250318_161244.log
```

See `src/README.md` for the internal HDF5 layout.

---

## `run_all.sh` 

`run_all.sh` runs the three generate scripts sequentially (not in parallel
with each other).  Running all three concurrently would flood all cores
with 3× the workers, causing context-switch overhead and no throughput gain.

```bash
./scripts/run_all.sh [options]

Options:
  --workers INT     Pass --workers to every generate script
  --outdir  PATH    Output directory (default: data/)
  --quick           Smoke-test mode: small N and L, completes in seconds
  -h, --help        Show help
```

The production parameter values are defined inside the script in the
`LOGISTIC_ARGS`, `HENON_ARGS`, and `ROSSLER_ARGS` blocks.  Edit those
directly to change sweep ranges or L values without touching the Python code.

`run_all.sh` uses `set -euo pipefail` — if any generate script fails, the
whole run aborts immediately and prints the exit code.

---

## Rössler wall-time note

Rössler is significantly slower than the discrete maps at the same `L`
because each sample requires ODE integration.  At `L = 10⁶`, `dt = 0.1`,
one parameter value requires integrating `10⁵` time units of ODE — roughly
15–30 s on a modern core.  A sweep of 500 parameter values on 16 cores
therefore takes approximately:

```
500 values / 16 workers × 20 s/value ≈ 625 s  (~10 min) per L
```

---

## Reproducing results from scratch

```bash
git clone <repo>
cd <repo>
conda env create -f environment.yml
conda activate etc
./scripts/run_all.sh --workers <N>
# then open notebooks/
```

No data files are committed to the repository.  All HDF5 files are generated
locally by the scripts above.
