import numpy as np
from numba import njit


# ── Core map ──────────────────────────────────────────────────────────────────

@njit
def _logistic_step(x: float, a: float) -> float:
    return a * x * (1.0 - x)


# ── Simulation ────────────────────────────────────────────────────────────────

@njit
def _trajectory(a: float, L: int, transient: int, x0: float) -> np.ndarray:
    x = x0
    
    # 1. Burn-in without array allocation
    for _ in range(transient):
        x = _logistic_step(x, a)

    # 2. Allocate exact needed memory and run production
    series = np.zeros(L)
    for i in range(L):
        x = _logistic_step(x, a)
        series[i] = x
        
    return series


def simulate(a: float, L: int, transient: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x0 = rng.random()
    return _trajectory(a, L, transient, x0)


# ── MLE ───────────────────────────────────────────────────────────────────────

@njit
def _mle(a: float, L: int, transient: int, x0: float) -> float:
    x = x0
    for _ in range(transient):
        x = _logistic_step(x, a)

    acc = 0.0
    for _ in range(L):
        x = _logistic_step(x, a)
        deriv = np.abs(a * (1.0 - 2.0 * x))
        if deriv > 0.0:
            acc += np.log(deriv)
        else:
            return -np.inf
    return acc / L


def lyapunov_mle(a: float, L: int, transient: int, seed: int = 42) -> float:
    rng = np.random.default_rng(seed)
    x0 = rng.random()
    return _mle(a, L, transient, x0)
