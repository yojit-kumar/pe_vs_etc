import numpy as np
from numba import njit


# ── Core map ──────────────────────────────────────────────────────────────────

@njit
def _henon_step(x: float, y: float, a: float, b: float):
    return 1.0 - a * x * x + y, b * x


# ── Simulation ────────────────────────────────────────────────────────────────

@njit
def _trajectory(a: float, b: float, L: int, transient: int,
                x0: float, y0: float) -> np.ndarray:
    x, y = x0, y0
    for _ in range(transient):
        x, y = _henon_step(x, y, a, b)

    series = np.zeros(L)
    for i in range(L):
        x, y = _henon_step(x, y, a, b)
        series[i] = x          # return x-component as the scalar time series
    return series


def simulate(a: float, b: float, L: int, transient: int,
             seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x0, y0 = rng.uniform(-0.5, 0.5), rng.uniform(-0.5, 0.5)
    return _trajectory(a, b, L, transient, x0, y0)


# ── MLE ───────────────────────────────────────────────────────────────────────

@njit
def _mle(a: float, b: float, L: int, transient: int,
         x0: float, y0: float) -> float:
    # Burn-in
    x, y = x0, y0
    for _ in range(transient):
        x, y = _henon_step(x, y, a, b)

    # Initialise tangent vector on the unit sphere
    dx, dy = 1.0, 0.0

    acc = 0.0
    for _ in range(L):
        # Store x_old before taking the step to safely compute the Jacobian
        x_old = x
        x, y = _henon_step(x, y, a, b)

        # Apply Jacobian: J * [dx, dy]^T evaluated at x_old
        new_dx = -2.0 * a * x_old * dx + dy
        new_dy = b * dx

        norm = np.sqrt(new_dx * new_dx + new_dy * new_dy)
        if norm <= 0.0:
            return -np.inf

        acc += np.log(norm)
        dx, dy = new_dx / norm, new_dy / norm

    return acc / L


def lyapunov_mle(a: float, b: float, L: int, transient: int,
                 seed: int = 42) -> float:
    rng = np.random.default_rng(seed)
    x0, y0 = rng.uniform(-0.5, 0.5), rng.uniform(-0.5, 0.5)
    return _mle(a, b, L, transient, x0, y0)
