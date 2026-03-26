# PE vs ETC: A Complexity Measure Comparison on Chaotic Dynamical Systems

This repository contains the code, notebooks, and methodology for a rigorous
comparison of **Permutation Entropy (PE)** and **Effort-To-Compress (ETC)** as
complexity measures across three canonical chaotic dynamical systems, evaluated
against the Maximum Lyapunov Exponent (MLE) as ground truth.

The study examines how well each measure tracks the onset of chaos as a
function of system parameters, and how robust each measure is under
increasing levels of measurement noise.

---

## Dynamical systems

| System | Swept parameter | Fixed parameters | Chaos onset |
|---|---|---|---|
| Logistic map | `a ∈ [3.5, 4.0]` | — | `a ≈ 3.57` |
| Hénon map | `a ∈ [0.8, 1.4]` | `b = 0.3` | `a ≈ 1.05` |
| Rössler system | `c ∈ [2.0, 8.0]` | `a = b = 0.2` | `c ≈ 5.7` |

The Rössler system is a continuous-time ODE integrated with RK45 and sampled
at a fixed interval `dt = 0.1`. All other systems are discrete maps.

---

## Complexity measures

**Permutation Entropy (PE)** — Bandt & Pompe (2002).
Embeds the time series into delay vectors of dimension `D`, maps each to its
ordinal rank pattern, and computes the normalised Shannon entropy of the
pattern distribution. Output ∈ [0, 1].

**Effort-To-Compress (ETC)** — Nagaraj et al. (2013).
Symbolises the series into `bins` equal-width symbols, then estimates
complexity via the Non-Sequential Recursive Pair Substitution (NSRPS)
algorithm. Output: Normalized ETC.

**Maximum Lyapunov Exponent (MLE)** — used as ground truth.
Computed analytically for the logistic map (exact Jacobian), via the Benettin
tangent-vector method for the Hénon map, and via the Wolf/Benettin variational
ODE method for Rössler. Positive MLE indicates chaos.

**Noise model:** measurement noise only. Each clean trajectory is generated
deterministically, then i.i.d. Gaussian noise `ε ~ N(0, σ²)` is added to
the observations. The noise does not feed back into the dynamics.

---

## Repository structure

```
.
├── src/                        Shared scientific library
│   ├── maps/
│   │   ├── logistic.py         Logistic map + MLE
│   │   ├── henon.py            Hénon map + MLE
│   │   └── rossler.py          Rössler ODE + MLE
│   ├── complexity.py           PE, ETC, and measurement noise utility
│   ├── io_utils.py             HDF5 save / load helpers
│   └── README.md               Detailed src/ documentation
│
├── scripts/                    Data generation
│   ├── _runner.py              Shared parallel runner
│   ├── generate_logistic.py    Logistic map parameter sweep
│   ├── generate_henon.py       Hénon map parameter sweep
│   ├── generate_rossler.py     Rössler system parameter sweep
│   ├── run_all.sh              Orchestrates all three scripts
│   └── README.md               Detailed scripts/ documentation
│
├── notebooks/                  Analysis and figures
│   ├── 01_sanity_check.ipynb   Verify dynamics + measures (no data needed)
│   ├── 02_logistic_analysis.ipynb
│   ├── 03_henon_analysis.ipynb
│   └── 04_rossler_analysis.ipynb
│
├── data/                       HDF5 output files (gitignored — see below)
├── figures/                    Saved plots (gitignored — see below)
├── environment.yml             Reproducible conda environment
├── .gitignore
└── README.md                   (this file)
```

`data/` and `figures/` are not tracked by git. See
[Reproducing the results](#reproducing-the-results) below.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/yojit-kumar/pe_vs_etc.git
cd pe_vs_etc
```

### 2. Create the conda environment

```bash
conda env create -f environment.yml
conda activate etc
```

### 3. Verify installation

Run the sanity-check notebook — it requires no pre-generated data:

```bash
jupyter lab notebooks/01_sanity_check.ipynb
```

Expected outputs are documented inside the notebook. If all MLE spot-checks
and PE/ETC sanity values look correct, the environment is working.

---

## Reproducing the results

All HDF5 data files are generated locally by the scripts. They are not
hosted in this repository due to file size.

### Quick test (~seconds)

Generates small HDF5 files to verify the full pipeline end-to-end:

```bash
./scripts/run_all.sh --quick
```

### Full run

```bash
./scripts/run_all.sh --workers <N>
```

Replace `<N>` with the number of CPU cores available.
Progress and timing are logged to `data/*.log`.

See [`scripts/README.md`](scripts/README.md) for full parameter documentation
and per-map run-time estimates.

### Running a single map

```bash
python scripts/generate_logistic.py --n-param 500 --L 100000 1000000 --workers 8
python scripts/generate_henon.py    --n-param 500 --L 100000 1000000 --workers 8
python scripts/generate_rossler.py  --n-param 500 --L 10000 100000   --workers 8
```

---

## Analysis notebooks

After generating data, open the analysis notebooks in any order:

| Notebook | Requires |
|---|---|
| `01_sanity_check.ipynb` | Nothing — simulates on the fly |
| `02_logistic_analysis.ipynb` | `data/logistic_L*.h5` |
| `03_henon_analysis.ipynb` | `data/henon_L*.h5` |
| `04_rossler_analysis.ipynb` | `data/rossler_L*.h5` |

Each analysis notebook produces the following figures, saved to `figures/<map>/`:

| Figure | Content |
|---|---|
| `noisefree_L{L}.png/.pdf` | PE (row 1) and ETC (row 2) vs parameter, MLE overlaid. One subplot column per D / bins value. |
| `pe_noise_L{L}.png/.pdf` | PE across all noise levels σ, one subplot per D |
| `etc_noise_L{L}.png/.pdf` | ETC across all noise levels σ, one subplot per bins |
| `correlation_vs_sigma_L{L}.png/.pdf` | Pearson correlation with MLE vs σ, PE and ETC together |
| `{map}_correlation_table.csv` | Full numerical summary table |

---

## Dependencies

Core scientific stack: `numpy`, `numba`, `scipy`, `h5py`, `matplotlib`,
`pandas`, `jupyterlab`.
Complexity libraries: [`ordpy`](https://github.com/arthurpessa/ordpy) (PE),
[`ETC`](https://github.com/pranaysy/ETCPy) (ETC).

---

## References

- Bandt, C. & Pompe, B. (2002). Permutation entropy: A natural complexity
  measure for time series. *Physical Review Letters*, 88(17), 174102.
- Nagaraj, N., Balasubramanian, K. & Dey, S. (2013). A new complexity measure
  for time series analysis and classification. *The European Physical Journal
  Special Topics*, 222(3–4), 847–860.
- Wolf, A., Swift, J. B., Swinney, H. L. & Vastano, J. A. (1985). Determining
  Lyapunov exponents from a time series. *Physica D*, 16(3), 285–317.
