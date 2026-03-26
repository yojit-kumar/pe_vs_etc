# `src/` — Library Reference

This directory contains the shared scientific library for the project.
All simulation, complexity measurement, and I/O logic is here.
The `scripts/` directory imports from here to run experiments; the `notebooks/` directory imports from here to analyse and plot results.

---

## Directory layout

```
src/
├── maps/
│   ├── __init__.py
│   ├── logistic.py     Logistic map
│   ├── henon.py        Hénon map
│   └── rossler.py      Rössler system (continuous-time ODE)
├── complexity.py       PE and ETC wrappers + measurement noise utility
├── io_utils.py         HDF5 save / load helpers
├── __init__.py
└── README.md           (this file)
```

Each map module exposes exactly two public functions:

| Function | Returns | Purpose |
|---|---|---|
| `simulate(...)` | `np.ndarray` | Clean scalar time series |
| `lyapunov_mle(...)` | `float` | Maximum Lyapunov Exponent |

Noise is applied in `complexity.add_measurement_noise`, **not** inside the map modules.
This keeps the map physics strictly separated from the observation model.

---

## Dynamical systems

### Logistic map — `maps/logistic.py`

```
x_{n+1} = a · x_n · (1 − x_n)
```

**Parameter sweep:** `a ∈ [3.5, 4.0]`

This range is chosen because it spans the full period-doubling cascade and the
onset of chaos. The system is periodic for most values below ≈ 3.57 and
chaotic for most values above it, with intermittent periodic windows (notably
the period-3 window near `a ≈ 3.83`). Restricting to `[3.5, 4.0]` means every
point in the sweep is either in a periodic window or in chaos — there is no
stable fixed-point behaviour in this range — which makes it the most
informative interval for distinguishing PE from ETC under varying complexity.

**MLE:** Computed analytically using the exact Jacobian `|f'(x)| = |a(1 − 2x)|`.
No finite-difference approximation is needed. 

---

### Hénon map — `maps/henon.py`

```
x_{n+1} = 1 − a · x_n² + y_n
y_{n+1} = b · x_n
```

**Fixed parameter:** `b = 0.3` (controls the area-contraction rate of the map).
This is the classical value at which the Hénon attractor is a strange attractor
for `a = 1.4`. Varying `b` changes the fractal dimension of the attractor but
has a less dramatic effect on the transition to chaos than varying `a`.

**Parameter sweep:** `a ∈ [0.8, 1.4]`

At `b = 0.3`, the map has a stable 2-cycle for `a ≲ 0.9`, undergoes a
period-doubling cascade, and reaches fully developed chaos by `a ≈ 1.4`
(the classical Hénon attractor). The sweep therefore covers the same
periodic → chaotic transition as the logistic sweep, making cross-map
comparisons meaningful.

**Time series:** The `x`-component is used as the scalar observable.

**MLE:** Computed via the Benettin tangent-vector method. A single tangent
vector `δ = (δx, δy)` is iterated under the linearised map at each step:

```
J · δ = [−2a·x_old · δx + δy,
          b · δx           ]
```

The vector is renormalised after every iterate to prevent overflow, and the
accumulated log-norms give the MLE. `x_old` is recovered as `y_new / b`
rather than storing it separately, keeping the Numba JIT context clean.

---

### Rössler system — `maps/rossler.py`

```
dx/dt = −y − z
dy/dt =  x + a·y
dz/dt =  b + z·(x − c)
```

**Fixed parameters:** `a = 0.2`, `b = 0.2` (standard values from the
original Rössler 1976 paper)

**Parameter sweep:** `c ∈ [2.0, 8.0]`

With `a = b = 0.2`, the Rössler system undergoes a period-doubling
route to chaos as `c` increases: period-1 near `c = 2`, period-2 near
`c = 3.5`, period-4 near `c = 4`, and broadband chaos for `c ≳ 5.7`.
This mirrors the structure of the logistic and Hénon sweeps, making
the three systems directly comparable.

**Integration:** `scipy.integrate.solve_ivp` with RK45 and tight tolerances
(`rtol = atol = 1e-9`). A transient of 500 time-units is discarded before
sampling to ensure the trajectory has settled onto the attractor.

**Continuous vs discrete:** Rössler is a continuous-time system, so it is
fundamentally different from the logistic and Hénon maps. Fixed-interval
sampling is used (rather than Poincaré sections) to produce a uniform,
evenly-spaced 1D time series that PE and ETC can consume without modification.
The downside is that the inter-sample dynamics are implicitly integrated out;
Poincaré sections would give a sparser but more dynamically faithful series.

**MLE:** Computed via the Wolf/Benettin method for continuous-time systems.
The augmented 6-variable ODE

```
[ẋ, ẏ, ż, δẋ, δẏ, δż] = [f(x,y,z), J(x,y,z)·δ]
```

is integrated in blocks of `renorm_interval · dt` time-units. After each
block the tangent vector `δ` is renormalised and its log-norm accumulated.
The MLE = total accumulated log-norm / total integration time (nats/time-unit).
The Jacobian of the Rössler field is:

```
J(x,y,z) = [[ 0,   −1,    −1  ],
             [ 1,    a,     0  ],
             [ z,    0,   x−c  ]]
```

---

## Noise model — measurement noise only

Only **measurement noise** is implemented. The system evolves under its
deterministic dynamics to produce the true trajectory `x_true[t]`, and
then independent Gaussian noise is added to each observation:

```
x_obs[t] = x_true[t] + ε[t],    ε[t] ~ N(0, σ²),  i.i.d.
```

**No clipping:** Noisy observations are not clipped to any bounded interval.
Clipping would introduce a systematic bias (truncated Gaussian tails, biased
variance) that constitutes an additional modelling assumption. The logistic
map attractor lives in `[0,1]`, but the observations with noise may transiently
exceed this range — that is fine, as we are modelling *measurement* error,
not constraining the state space.

---

## Complexity measures — `complexity.py`

### Permutation Entropy (PE)

Bandt & Pompe (2002). The series is embedded into delay vectors of
dimension `D` with time delay `t = 1`. Each vector is mapped to its
ordinal rank pattern, and the Shannon entropy of the pattern distribution
is normalised by `log(D!)`.

- Output ∈ `[0, 1]`
- Key parameter: embedding dimension `D` 
- Rule of thumb: `L >> D!` for reliable pattern distribution estimates
- At `L = 100 000` and `D = 7`: `L / 7! = 100000 / 5040 ≈ 20` samples
  per pattern on average — marginal. Prefer `D ≤ 5` for `L < 10 000`.

### Effort-To-Compress (ETC)

Nagaraj et al. (2013). Continuous values are first symbolised into
`bins` equal-width symbols via `ETC.partition`, then the Non-Sequential
Recursive Pair Substitution (NSRPS) algorithm estimates the complexity of
the resulting symbolic sequence.

- Output: ∈ `[0, 1]`
- Key parameter: `bins` (number of symbols)
- Coarser symbolisation (fewer bins) is more robust to noise but loses
  fine structure; finer symbolisation captures more detail but becomes
  noise-sensitive — this trade-off is a key comparison axis in the analysis.

---

## HDF5 file layout — `io_utils.py`

```
file.h5
  attrs: description, map_name, parameter_name, created_on, ...
  param_values          [N]
  mle                   [N]
  noise_0.000000/
    attrs: noise_sigma
    permutation_entropy/
      D_3/              [N]    attrs: embedding_dimension, time_delay
      D_5/              [N]
    etc/
      bins_2/           [N]    attrs: num_bins
      bins_4/           [N]
  noise_0.010000/
    ...
```

Noise groups are named with 6 decimal places so they sort
lexicographically in the correct numerical order.
