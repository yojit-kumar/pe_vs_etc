"""
Dynamical map modules.

Each module exposes two functions:
    simulate(...)      → np.ndarray  clean time series
    lyapunov_mle(...)  → float       Maximum Lyapunov Exponent

See individual modules for parameter details.
"""

from . import logistic
from . import henon
from . import rossler

__all__ = ["logistic", "henon", "rossler"]
