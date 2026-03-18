import numpy as np
from ordpy import permutation_entropy
from ETC import partition, compute_1D


# ── Noise utility ─────────────────────────────────────────────────────────────

def add_measurement_noise(
    series: np.ndarray,
    sigma: float,
    seed: int = 42,
) -> np.ndarray:

    if sigma == 0.0:
        return series.copy()

    rng = np.random.default_rng(seed)
    return series + rng.normal(0.0, sigma, size=series.shape)


# ── Permutation Entropy ───────────────────────────────────────────────────────

def pe_method(series: np.ndarray, D: int, t: int = 1) -> float:
    return float(permutation_entropy(series, dx=D, taux=t, normalized=True))


# ── Effort-To-Compress ────────────────────────────────────────────────────────

def etc_method(series: np.ndarray, bins: int) -> float:
    symbolised = partition(series, n_bins=bins)
    result = compute_1D(symbolised)
    return float(result.get('NETC1D'))
