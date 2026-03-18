"""
Usage
-----
    python scripts/generate_rossler.py [options]

    --c-min   FLOAT       left edge of c sweep          (default: 2.0)
    --c-max   FLOAT       right edge of c sweep         (default: 8.0)
    --n-param INT         number of c values            (default: 500)
    --a-fixed FLOAT       fixed a parameter             (default: 0.2)
    --b-fixed FLOAT       fixed b parameter             (default: 0.2)
    --dt      FLOAT       sampling interval (time units)(default: 0.1)
    --transient-time FLOAT burn-in time (time units)   (default: 500.0)
    --L       INT [...]   series length(s)              (default: 10000 100000)
    --D       INT [...]   PE embedding dimensions       (default: 3 5 7)
    --bins    INT [...]   ETC bin counts                (default: 2 3 4 5)
    --noise   FLOAT [...] measurement noise σ           (default: 0.0 0.01 0.05 0.1)
    --workers INT         parallel workers              (default: all cores)
    --outdir  PATH        output directory              (default: data/)
    --seed    INT         RNG seed                      (default: 42)

Example — quick test
---------------------
    python scripts/generate_rossler.py --n-param 20 --L 5000 --workers 4

Example — full run
------------------------------
    python scripts/generate_rossler.py \\
        --n-param 500 \\
        --L 10000 100000 1000000 \\
        --D 3 5 7 \\
        --bins 2 3 4 5 \\
        --noise 0.0 0.01 0.02 0.05 0.1 \\
        --workers 16
"""

import sys
import argparse
import logging
import time
import numpy as np
from pathlib import Path

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="ETC")

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from scripts._runner import run_experiment


def parse_args():
    p = argparse.ArgumentParser(
        description="Rössler system — PE vs ETC parameter sweep",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--c-min',   type=float, default=2.0,
                   help='left edge of c sweep')
    p.add_argument('--c-max',   type=float, default=8.0,
                   help='right edge of c sweep')
    p.add_argument('--n-param', type=int,   default=500,
                   help='number of c values in sweep')
    p.add_argument('--a-fixed', type=float, default=0.2,
                   help='fixed a parameter')
    p.add_argument('--b-fixed', type=float, default=0.2,
                   help='fixed b parameter')
    p.add_argument('--dt',      type=float, default=0.1,
                   help='fixed sampling interval (time units)')
    p.add_argument('--transient-time', type=float, default=500.0,
                   help='transient integration time (time units) discarded before sampling')
    p.add_argument('--L',       type=int,   nargs='+',
                   default=[10_000, 100_000],
                   help='series length(s) — number of samples, not time units')
    p.add_argument('--D',       type=int,   nargs='+', default=[3, 5, 7],
                   help='PE embedding dimension(s)')
    p.add_argument('--bins',    type=int,   nargs='+', default=[2, 3, 4, 5],
                   help='ETC bin count(s)')
    p.add_argument('--noise',   type=float, nargs='+',
                   default=[0.0, 0.01, 0.05, 0.1],
                   help='measurement noise σ values')
    p.add_argument('--workers', type=int,   default=None,
                   help='parallel worker processes (default: all cores)')
    p.add_argument('--outdir',  type=str,   default='data',
                   help='output directory for HDF5 files')
    p.add_argument('--seed',    type=int,   default=42,
                   help='RNG seed for initial conditions')
    return p.parse_args()


def setup_logging(outdir: str) -> None:
    Path(outdir).mkdir(parents=True, exist_ok=True)
    log_path = Path(outdir) / f"rossler_{time.strftime('%Y%m%d_%H%M%S')}.log"

    fmt = logging.Formatter(
        '%(asctime)s  %(levelname)-8s  %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    root.addHandler(ch)

    fh = logging.FileHandler(log_path, mode='w')
    fh.setFormatter(fmt)
    root.addHandler(fh)

    logging.info(f"Log file: {log_path}")


def main():
    import multiprocessing as mp
    args = parse_args()

    n_workers = args.workers or mp.cpu_count()
    outdir    = Path(args.outdir)

    setup_logging(str(outdir))

    c_values = np.linspace(args.c_min, args.c_max, args.n_param)

    # fixed_params carries everything the Rössler simulate/MLE calls need
    # beyond the swept parameter c and the series length L.
    fixed_params = {
        'a':             args.a_fixed,
        'b':             args.b_fixed,
        'dt':            args.dt,
        'transient_time': args.transient_time,
    }

    logging.info(f"Rössler system sweep: c ∈ [{args.c_min}, {args.c_max}], "
                 f"{args.n_param} values")
    logging.info(f"Fixed params: a={args.a_fixed}, b={args.b_fixed}, dt={args.dt}")
    logging.info(f"Transient time: {args.transient_time} time-units")
    logging.info(f"L values (samples): {args.L}")

    # Rössler has no integer transient — pass transient_time via fixed_params
    # and set the _runner 'transient' argument to 0 (unused for Rössler;
    # the runner passes it only to discrete-map MLE/simulate calls).
    TRANSIENT_PLACEHOLDER = 0

    t_start = time.time()

    for L in args.L:
        # Rough wall-clock estimate so users can abort early if needed
        est_s = L * args.dt * args.n_param / max(n_workers, 1) / 1e4
        logging.info(f"L={L:,} — rough ODE wall-time estimate: "
                     f"~{est_s/60:.0f} min with {n_workers} workers")

        filename = str(outdir / f"rossler_L{L}.h5")
        run_experiment(
            map_name       = 'rossler',
            filename       = filename,
            param_values   = c_values,
            fixed_params   = fixed_params,
            L              = L,
            transient      = TRANSIENT_PLACEHOLDER,
            D_array        = args.D,
            bin_array      = args.bins,
            noise_values   = args.noise,
            n_workers      = n_workers,
            sim_seed       = args.seed,
            parameter_name = 'c',
            metadata       = {
                'a_fixed':        args.a_fixed,
                'b_fixed':        args.b_fixed,
                'dt':             args.dt,
                'transient_time': args.transient_time,
                'c_min':          args.c_min,
                'c_max':          args.c_max,
            },
        )

    logging.info(f"All L values complete. Total wall time: "
                 f"{(time.time()-t_start)/60:.1f} min")


if __name__ == '__main__':
    main()
