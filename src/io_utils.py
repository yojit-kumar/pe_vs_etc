"""
HDF5 I/O utilities — save and load simulation results.

Usage example
-------------
>>> from src.io_utils import save_results, load_results
>>> save_results("data/logistic_L100000.h5", ...)
>>> data = load_results("data/logistic_L100000.h5")
>>> pe = data['pe'][0.0][5]          # clean run, D=5
>>> etc = data['etc'][0.01][3]       # σ=0.01, bins=3
"""

import time
import logging
import numpy as np
import h5py
from pathlib import Path


# ── Save ──────────────────────────────────────────────────────────────────────

def save_results(
    filepath: str,
    map_name: str,
    parameter_name: str,
    param_values: np.ndarray,
    mle_values: np.ndarray,
    pe_results: dict,
    etc_results: dict,
    metadata: dict = None,
) -> None:
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(filepath, 'w') as f:

        # Root attrs
        f.attrs['description']    = f"PE vs ETC comparison — {map_name}"
        f.attrs['map_name']       = map_name
        f.attrs['parameter_name'] = parameter_name
        f.attrs['created_on']     = time.ctime()
        f.attrs['n_param_values'] = len(param_values)

        if metadata:
            for k, v in metadata.items():
                f.attrs[k] = v

        # Shared datasets
        f.create_dataset('param_values', data=param_values)
        f.create_dataset('mle',          data=mle_values)

        # One group per noise level
        for noise in sorted(pe_results.keys()):
            grp = f.create_group(f"noise_{noise:.6f}")
            grp.attrs['noise_sigma'] = float(noise)

            pe_grp = grp.create_group("permutation_entropy")
            for D, arr in pe_results[noise].items():
                ds = pe_grp.create_dataset(f"D_{D}", data=arr)
                ds.attrs['embedding_dimension'] = int(D)
                ds.attrs['time_delay']          = 1

            etc_grp = grp.create_group("etc")
            for bins, arr in etc_results[noise].items():
                ds = etc_grp.create_dataset(f"bins_{bins}", data=arr)
                ds.attrs['num_bins'] = int(bins)

    logging.info(f"Saved results → {filepath}")


# ── Load ──────────────────────────────────────────────────────────────────────

def load_results(filepath: str) -> dict:
    out = {
        'pe':       {},
        'etc':      {},
        'metadata': {},
    }

    with h5py.File(filepath, 'r') as f:

        # Root attrs → metadata
        for k, v in f.attrs.items():
            out['metadata'][k] = v

        out['map_name']       = str(f.attrs.get('map_name',       'unknown'))
        out['parameter_name'] = str(f.attrs.get('parameter_name', 'param'))
        out['param_values']   = f['param_values'][:]
        out['mle']            = f['mle'][:]

        noise_groups = sorted(k for k in f.keys() if k.startswith('noise_'))
        noise_levels = []

        for ng in noise_groups:
            grp  = f[ng]
            σ    = float(grp.attrs['noise_sigma'])
            noise_levels.append(σ)

            out['pe'][σ]  = {}
            out['etc'][σ] = {}

            for ds_name, ds in grp['permutation_entropy'].items():
                D = int(ds.attrs['embedding_dimension'])
                out['pe'][σ][D] = ds[:]

            for ds_name, ds in grp['etc'].items():
                bins = int(ds.attrs['num_bins'])
                out['etc'][σ][bins] = ds[:]

        out['noise_levels'] = sorted(noise_levels)

    return out


# ── Convenience ───────────────────────────────────────────────────────────────

def list_contents(filepath: str) -> None:
    """Pretty-print the structure of a saved HDF5 file."""
    with h5py.File(filepath, 'r') as f:
        print(f"File: {filepath}")
        print("─" * 50)
        print("Root attrs:")
        for k, v in f.attrs.items():
            print(f"  {k}: {v}")
        print("\nDatasets / groups:")
        f.visititems(lambda name, obj: print(f"  /{name}"))
