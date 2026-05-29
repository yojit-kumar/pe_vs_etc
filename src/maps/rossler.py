import numpy as np
from numba import njit

# ── ODE Definitions & RK4 Integrators ─────────────────────────────────────────

@njit
def _rossler_deriv(x: float, y: float, z: float, 
                   a: float, b: float, c: float):
    """Pure Rössler vector field."""
    return -y - z, x + a * y, b + z * (x - c)

@njit
def _rossler_var_deriv(x: float, y: float, z: float, 
                       dx: float, dy: float, dz: float, 
                       a: float, b: float, c: float):
    """Augmented 6D vector field: Rössler + Variational equations."""
    # Base field
    fx = -y - z
    fy = x + a * y
    fz = b + z * (x - c)
    
    # Jacobian * [dx, dy, dz]^T
    fdx = -dy - dz
    fdy = dx + a * dy
    fdz = z * dx + (x - c) * dz
    
    return fx, fy, fz, fdx, fdy, fdz

@njit
def _rk4_step(x: float, y: float, z: float, 
              a: float, b: float, c: float, dt: float):
    """Single fixed-step RK4 integration for the base system."""
    k1x, k1y, k1z = _rossler_deriv(x, y, z, a, b, c)
    
    k2x, k2y, k2z = _rossler_deriv(x + 0.5*dt*k1x, y + 0.5*dt*k1y, z + 0.5*dt*k1z, a, b, c)
    k3x, k3y, k3z = _rossler_deriv(x + 0.5*dt*k2x, y + 0.5*dt*k2y, z + 0.5*dt*k2z, a, b, c)
    k4x, k4y, k4z = _rossler_deriv(x + dt*k3x, y + dt*k3y, z + dt*k3z, a, b, c)
    
    nx = x + (dt / 6.0) * (k1x + 2.0*k2x + 2.0*k3x + k4x)
    ny = y + (dt / 6.0) * (k1y + 2.0*k2y + 2.0*k3y + k4y)
    nz = z + (dt / 6.0) * (k1z + 2.0*k2z + 2.0*k3z + k4z)
    
    return nx, ny, nz

@njit
def _rk4_var_step(x: float, y: float, z: float, 
                  dx: float, dy: float, dz: float, 
                  a: float, b: float, c: float, dt: float):
    """Single fixed-step RK4 integration for the 6D variational system."""
    k1x, k1y, k1z, k1dx, k1dy, k1dz = _rossler_var_deriv(
        x, y, z, dx, dy, dz, a, b, c)
    
    k2x, k2y, k2z, k2dx, k2dy, k2dz = _rossler_var_deriv(
        x + 0.5*dt*k1x, y + 0.5*dt*k1y, z + 0.5*dt*k1z, 
        dx + 0.5*dt*k1dx, dy + 0.5*dt*k1dy, dz + 0.5*dt*k1dz, a, b, c)
    
    k3x, k3y, k3z, k3dx, k3dy, k3dz = _rossler_var_deriv(
        x + 0.5*dt*k2x, y + 0.5*dt*k2y, z + 0.5*dt*k2z, 
        dx + 0.5*dt*k2dx, dy + 0.5*dt*k2dy, dz + 0.5*dt*k2dz, a, b, c)
    
    k4x, k4y, k4z, k4dx, k4dy, k4dz = _rossler_var_deriv(
        x + dt*k3x, y + dt*k3y, z + dt*k3z, 
        dx + dt*k3dx, dy + dt*k3dy, dz + dt*k3dz, a, b, c)
    
    nx = x + (dt / 6.0) * (k1x + 2.0*k2x + 2.0*k3x + k4x)
    ny = y + (dt / 6.0) * (k1y + 2.0*k2y + 2.0*k3y + k4y)
    nz = z + (dt / 6.0) * (k1z + 2.0*k2z + 2.0*k3z + k4z)
    
    ndx = dx + (dt / 6.0) * (k1dx + 2.0*k2dx + 2.0*k3dx + k4dx)
    ndy = dy + (dt / 6.0) * (k1dy + 2.0*k2dy + 2.0*k3dy + k4dy)
    ndz = dz + (dt / 6.0) * (k1dz + 2.0*k2dz + 2.0*k3dz + k4dz)
    
    return nx, ny, nz, ndx, ndy, ndz


# ── Simulation ────────────────────────────────────────────────────────────────

@njit
def _trajectory(a: float, b: float, c: float, 
                L: int, transient_steps: int, dt: float,
                x0: float, y0: float, z0: float) -> np.ndarray:
    x, y, z = x0, y0, z0
    
    # 1. Burn-in (no array allocation)
    for _ in range(transient_steps):
        x, y, z = _rk4_step(x, y, z, a, b, c, dt)

    # 2. Production run
    series = np.zeros(L)
    for i in range(L):
        x, y, z = _rk4_step(x, y, z, a, b, c, dt)
        series[i] = x  # Retain only the x-component observation
        
    return series

def simulate(a: float, b: float, c: float,
             L: int,
             dt: float = 0.1,
             transient_time: float = 500.0,
             seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x0, y0, z0 = rng.uniform(-1.0, 1.0, 3)
    transient_steps = int(transient_time / dt)
    
    return _trajectory(a, b, c, L, transient_steps, dt, x0, y0, z0)


# ── MLE ───────────────────────────────────────────────────────────────────────

@njit
def _mle(a: float, b: float, c: float, 
         L: int, transient_steps: int, dt: float, renorm_interval: int,
         x0: float, y0: float, z0: float, 
         dx0: float, dy0: float, dz0: float) -> float:
    x, y, z = x0, y0, z0
    
    # 1. Burn-in to settle on attractor
    for _ in range(transient_steps):
        x, y, z = _rk4_step(x, y, z, a, b, c, dt)

    # 2. Setup tangent vector
    dx, dy, dz = dx0, dy0, dz0
    norm = np.sqrt(dx*dx + dy*dy + dz*dz)
    dx, dy, dz = dx / norm, dy / norm, dz / norm

    acc = 0.0
    
    # 3. Continuous Benettin accumulation
    for i in range(L):
        x, y, z, dx, dy, dz = _rk4_var_step(x, y, z, dx, dy, dz, a, b, c, dt)
        
        # Renormalize periodically
        if (i + 1) % renorm_interval == 0:
            norm = np.sqrt(dx*dx + dy*dy + dz*dz)
            if norm <= 0.0:
                return -np.inf
            acc += np.log(norm)
            dx, dy, dz = dx / norm, dy / norm, dz / norm

    total_time = L * dt
    return acc / total_time

def lyapunov_mle(a: float, b: float, c: float,
                 L: int,
                 dt: float = 0.1,
                 transient_time: float = 500.0,
                 renorm_interval: int = 100,
                 seed: int = 42) -> float:
    rng = np.random.default_rng(seed)
    x0, y0, z0 = rng.uniform(-1.0, 1.0, 3)
    dx0, dy0, dz0 = rng.standard_normal(3)
    
    transient_steps = int(transient_time / dt)
    
    return _mle(a, b, c, L, transient_steps, dt, renorm_interval,
                x0, y0, z0, dx0, dy0, dz0)
