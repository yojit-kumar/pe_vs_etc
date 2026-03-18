"""
_runner.py — shared experiment runner with multiprocessing.

Parallelism strategy
--------------------
The parameter sweep (N values) is the natural unit of parallelism:
  - MLE:  one task per parameter value  → Pool.imap over N tasks
  - PE/ETC: one task per (param, noise) pair, but we loop over noise levels
    sequentially in the main process and dispatch N tasks per level.
    This keeps peak memory at n_workers × L × 8 bytes (one series per worker).

For very large L (≥ 10^7) reduce n_workers to avoid OOM:
  L=1e7, n_workers=8  →  8 × 80 MB = 640 MB   (manageable)
  L=1e8, n_workers=4  →  4 × 800 MB = 3.2 GB  (check available RAM first)
"""

import sys
import logging
import time
import numpy as np
import multiprocessing as mp

## FIX
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="ETC")
##

# ── Resolve src/ regardless of where the script is run from ──────────────────
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.complexity import pe_method, etc_method, add_measurement_noise
from src.io_utils import save_results


# ─────────────────────────────────────────────────────────────────────────────
# Top-level worker functions  (must be importable → no closures / lambdas)
# ─────────────────────────────────────────────────────────────────────────────

def _mle_worker(args: tuple) -> float:
    """
    Compute MLE for one parameter value.

    args = (map_name, param_val, fixed_params, L, transient, seed)
    Returns float (MLE value, or np.nan on failure).
    """
    map_name, param_val, fixed_params, L, transient, seed = args

    try:
        if map_name == 'logistic':
            from src.maps.logistic import lyapunov_mle
            return lyapunov_mle(param_val, L, transient, seed)

        elif map_name == 'henon':
            from src.maps.henon import lyapunov_mle
            return lyapunov_mle(param_val, fixed_params['b'], L, transient, seed)

        elif map_name == 'rossler':
            from src.maps.rossler import lyapunov_mle
            return lyapunov_mle(
                fixed_params['a'], fixed_params['b'], param_val,
                L,
                dt=fixed_params['dt'],
                transient_time=fixed_params['transient_time'],
                seed=seed,
            )
        else:
            raise ValueError(f"Unknown map: {map_name}")

    except Exception as e:
        logging.error(f"MLE worker failed (map={map_name}, param={param_val:.6f}): {e}")
        return np.nan


def _sim_worker(args: tuple) -> tuple:
    """
    Simulate one parameter value at one noise level, then compute PE and ETC.

    args = (map_name, param_val, fixed_params, L, transient,
            D_array, bin_array, noise_sigma, sim_seed, noise_seed)

    Returns (pe_dict, etc_dict) where:
        pe_dict  = {D:    float}
        etc_dict = {bins: float}
    or raises on failure (caught by caller).
    """
    (map_name, param_val, fixed_params, L, transient,
     D_array, bin_array, noise_sigma, sim_seed, noise_seed) = args

    # ── Simulate clean trajectory ────────────────────────────────────────────
    if map_name == 'logistic':
        from src.maps.logistic import simulate
        series_clean = simulate(param_val, L, transient, sim_seed)

    elif map_name == 'henon':
        from src.maps.henon import simulate
        series_clean = simulate(param_val, fixed_params['b'], L, transient, sim_seed)
        # Hénon can diverge outside its basin of attraction
        if not np.all(np.isfinite(series_clean)):
            nan_pe  = {D:    np.nan for D    in D_array}
            nan_etc = {bins: np.nan for bins in bin_array}
            return nan_pe, nan_etc

    elif map_name == 'rossler':
        from src.maps.rossler import simulate
        series_clean = simulate(
            fixed_params['a'], fixed_params['b'], param_val,
            L,
            dt=fixed_params['dt'],
            transient_time=fixed_params['transient_time'],
            seed=sim_seed,
        )
    else:
        raise ValueError(f"Unknown map: {map_name}")

    # ── Add measurement noise ────────────────────────────────────────────────
    series = add_measurement_noise(series_clean, noise_sigma, noise_seed)

    # ── Complexity measures ──────────────────────────────────────────────────
    pe_dict  = {D:    pe_method(series, D)    for D    in D_array}
    etc_dict = {bins: etc_method(series, bins) for bins in bin_array}

    return pe_dict, etc_dict


# ─────────────────────────────────────────────────────────────────────────────
# Main experiment runner
# ─────────────────────────────────────────────────────────────────────────────

def run_experiment(
    map_name:      str,
    filename:      str,
    param_values:  np.ndarray,
    fixed_params:  dict,
    L:             int,
    transient,                   # int for discrete maps, float for Rössler
    D_array:       list,
    bin_array:     list,
    noise_values:  list,
    n_workers:     int,
    sim_seed:      int = 42,
    parameter_name: str = 'param',
    metadata:      dict = None,
) -> None:
    """
    Run a full parameter sweep with parallel workers and save to HDF5.

    Parameters
    ----------
    map_name       : 'logistic' | 'henon' | 'rossler'
    filename       : output HDF5 path
    param_values   : 1D array of swept parameter values
    fixed_params   : dict of non-swept parameters (map-specific)
    L              : time-series length (samples)
    transient      : burn-in (iterations for maps, time-units for Rössler)
    D_array        : list of PE embedding dimensions
    bin_array      : list of ETC bin counts
    noise_values   : list of noise sigmas (0.0 = clean)
    n_workers      : number of parallel worker processes
    sim_seed       : RNG seed for initial conditions
    parameter_name : label for the swept parameter (for HDF5 attrs)
    metadata       : optional extra attrs to store in HDF5
    """
    N = len(param_values)

    # ── Memory estimate ──────────────────────────────────────────────────────
    bytes_per_series = L * 8
    peak_mb = (n_workers * bytes_per_series) / 1024**2
    logging.info("=" * 60)
    logging.info(f"MAP          : {map_name}")
    logging.info(f"OUTPUT       : {filename}")
    logging.info(f"L            : {L:,}")
    logging.info(f"param sweep  : {N} values of '{parameter_name}'")
    logging.info(f"D_array      : {D_array}")
    logging.info(f"bin_array    : {bin_array}")
    logging.info(f"noise_values : {noise_values}")
    logging.info(f"n_workers    : {n_workers}")
    logging.info(f"Peak RAM est : ~{peak_mb:.0f} MB  ({n_workers} workers × {bytes_per_series/1024**2:.0f} MB)")
    logging.info("=" * 60)

    ctx = mp.get_context('spawn')   # Numba-safe: fresh interpreter per worker

    # ── Step 1: MLE (noise-free, computed once) ──────────────────────────────
    logging.info("Computing MLE in parallel …")
    mle_args = [
        (map_name, float(p), fixed_params, L, transient, sim_seed)
        for p in param_values
    ]
    mle_values = np.full(N, np.nan)

    t0 = time.time()
    with ctx.Pool(processes=n_workers) as pool:
        for i, result in enumerate(pool.imap(_mle_worker, mle_args)):
            mle_values[i] = result
            if (i + 1) % max(1, N // 10) == 0:
                logging.info(f"  MLE: {i+1}/{N} done")

    logging.info(f"MLE complete in {time.time()-t0:.1f}s  "
                 f"(chaotic fraction: "
                 f"{np.mean(mle_values[np.isfinite(mle_values)] > 0):.1%})")

    # ── Step 2: PE + ETC for each noise level ────────────────────────────────
    pe_all  = {}   # {sigma: {D:    np.ndarray(N)}}
    etc_all = {}   # {sigma: {bins: np.ndarray(N)}}

    for noise_idx, sigma in enumerate(noise_values):
        logging.info(f"[{noise_idx+1}/{len(noise_values)}] "
                     f"Noise σ={sigma}  — dispatching {N} workers …")

        # noise_seed is offset from sim_seed so the two RNG streams never collide
        noise_seed = sim_seed + 1 if sigma > 0.0 else sim_seed

        sim_args = [
            (map_name, float(p), fixed_params, L, transient,
             D_array, bin_array, sigma, sim_seed, noise_seed)
            for p in param_values
        ]

        pe_noise  = {D:    np.full(N, np.nan) for D    in D_array}
        etc_noise = {bins: np.full(N, np.nan) for bins in bin_array}

        t1 = time.time()
        with ctx.Pool(processes=n_workers) as pool:
            for i, (pe_d, etc_d) in enumerate(
                    pool.imap(_sim_worker, sim_args)):
                for D in D_array:
                    pe_noise[D][i] = pe_d[D]
                for bins in bin_array:
                    etc_noise[bins][i] = etc_d[bins]

                if (i + 1) % max(1, N // 10) == 0:
                    logging.info(f"  σ={sigma}: {i+1}/{N} done")

        pe_all[sigma]  = pe_noise
        etc_all[sigma] = etc_noise
        logging.info(f"  σ={sigma} complete in {time.time()-t1:.1f}s")

    # ── Step 3: Save ─────────────────────────────────────────────────────────
    logging.info("Saving results …")
    full_meta = {
        'L':         L,
        'transient': float(transient),
        'n_workers': n_workers,
        'sim_seed':  sim_seed,
    }
    if metadata:
        full_meta.update(metadata)

    save_results(
        filepath       = filename,
        map_name       = map_name,
        parameter_name = parameter_name,
        param_values   = param_values,
        mle_values     = mle_values,
        pe_results     = pe_all,
        etc_results    = etc_all,
        metadata       = full_meta,
    )
    logging.info(f"Done → {filename}")
    logging.info("=" * 60)
