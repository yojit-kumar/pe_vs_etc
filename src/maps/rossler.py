import logging
import numpy as np
from scipy.integrate import solve_ivp


# ── ODE definitions ───────────────────────────────────────────────────────────

def _rossler(t, state, a, b, c):
    """Pure Rössler vector field (3 variables)."""
    x, y, z = state
    return [-y - z,
            x + a * y,
            b + z * (x - c)]


def _rossler_variational(t, state, a, b, c):
    x, y, z = state[0], state[1], state[2]
    dx, dy, dz = state[3], state[4], state[5]

    # Vector field
    f0 = -y - z
    f1 = x + a * y
    f2 = b + z * (x - c)

    # J · [dx, dy, dz]^T
    jd0 = -dy - dz
    jd1 = dx + a * dy
    jd2 = z * dx + (x - c) * dz

    return [f0, f1, f2, jd0, jd1, jd2]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _integrate(fun, t_span, y0, args, t_eval=None, label=""):
    """Thin wrapper around solve_ivp with consistent tolerances and error handling."""
    sol = solve_ivp(
        fun, t_span, y0,
        args=args,
        method='RK45',
        t_eval=t_eval,
        rtol=1e-9,
        atol=1e-9,
        dense_output=False,
    )
    if not sol.success:
        raise RuntimeError(
            f"Rössler integration failed{' (' + label + ')' if label else ''}: "
            f"{sol.message}"
        )
    return sol


def _burn_in(a, b, c, transient_time, x0):
    """Integrate for transient_time to reach attractor; return final state."""
    sol = _integrate(_rossler, [0, transient_time], x0, args=(a, b, c),
                     label="transient")
    return sol.y[:, -1]


# ── Simulation ────────────────────────────────────────────────────────────────

def simulate(a: float, b: float, c: float,
             L: int,
             dt: float = 0.1,
             transient_time: float = 500.0,
             seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x0 = rng.uniform(-1.0, 1.0, 3)

    # 1. Discard transient
    x_init = _burn_in(a, b, c, transient_time, x0)

    # 2. Sample production trajectory at fixed intervals
    #    Use a slight overshoot in t_end to guarantee L points from linspace.
    t_end = (L - 1) * dt
    t_eval = np.linspace(0.0, t_end, L)

    sol = _integrate(_rossler, [0.0, t_end], x_init, args=(a, b, c),
                     t_eval=t_eval, label="production")

    return sol.y[0, :L]          # x-component, exactly L points


# ── MLE ───────────────────────────────────────────────────────────────────────

def lyapunov_mle(a: float, b: float, c: float,
                 L: int,
                 dt: float = 0.1,
                 transient_time: float = 500.0,
                 renorm_interval: int = 100,
                 seed: int = 42) -> float:
    rng = np.random.default_rng(seed)
    x0 = rng.uniform(-1.0, 1.0, 3)

    # 1. Transient
    x_current = _burn_in(a, b, c, transient_time, x0)

    # 2. Random unit tangent vector
    delta0 = rng.standard_normal(3)
    delta0 /= np.linalg.norm(delta0)

    state = np.concatenate([x_current, delta0])

    # 3. Benettin accumulation
    n_blocks = L // renorm_interval
    block_time = renorm_interval * dt

    acc = 0.0
    total_time = 0.0

    for blk in range(n_blocks):
        try:
            sol = _integrate(
                _rossler_variational,
                [0.0, block_time],
                state,
                args=(a, b, c),
                label=f"MLE block {blk}",
            )
        except RuntimeError as e:
            logging.warning(str(e))
            break

        state = sol.y[:, -1]
        delta = state[3:6]
        norm = np.linalg.norm(delta)

        if norm <= 0.0:
            logging.warning(f"Rössler MLE: tangent vector collapsed at block {blk}.")
            break

        acc += np.log(norm)
        state[3:6] = delta / norm       # renormalise
        total_time += block_time

    return acc / total_time if total_time > 0.0 else np.nan
