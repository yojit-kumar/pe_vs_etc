"""
Usage
-----
    python scripts/generate_henon.py [options]

    --a-min   FLOAT       left edge of a sweep      (default: 0.8)
    --a-max   FLOAT       right edge of a sweep     (default: 1.4)
    --n-param INT         number of a values        (default: 500)
    --b-fixed FLOAT       fixed b parameter         (default: 0.3)
    --L       INT [...]   series length(s)          (default: 100000 1000000)
    --transient INT       burn-in iterates          (default: 1000)
    --D       INT [...]   PE embedding dimensions   (default: 3 5 7)
    --bins    INT [...]   ETC bin counts            (default: 2 3 4 5)
    --noise   FLOAT [...] measurement noise σ       (default: 0.0 0.01 0.05 0.1)
    --workers INT         parallel workers          (default: all cores)
    --outdir  PATH        output directory          (default: data/)
    --seed    INT         RNG seed                  (default: 42)

Example — quick test
---------------------
    python scripts/generate_henon.py --n-param 50 --L 10000 --workers 4

Example — full run
------------------------------
    python scripts/generate_henon.py \\
        --n-param 500 \\
        --L 100000 1000000 10000000 \\
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
        description="Hénon map — PE vs ETC parameter sweep",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--a-min',   type=float, default=0.8,
                   help='left edge of a sweep')
    p.add_argument('--a-max',   type=float, default=1.4,
                   help='right edge of a sweep')
    p.add_argument('--n-param', type=int,   default=500,
                   help='number of a values in sweep')
    p.add_argument('--b-fixed', type=float, default=0.3,
                   help='fixed value of b (area contraction parameter)')
    p.add_argument('--L',       type=int,   nargs='+',
                   default=[100_000, 1_000_000],
                   help='series length(s) to run')
    p.add_argument('--transient', type=int, default=1000,
                   help='burn-in iterates discarded before sampling')
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
    log_path = Path(outdir) / f"henon_{time.strftime('%Y%m%d_%H%M%S')}.log"

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

    a_values = np.linspace(args.a_min, args.a_max, args.n_param)

    # fixed_params carries b, passed through to the worker
    fixed_params = {'b': args.b_fixed}

    logging.info(f"Hénon map sweep: a ∈ [{args.a_min}, {args.a_max}], "
                 f"{args.n_param} values, b = {args.b_fixed}")
    logging.info(f"L values: {args.L}")

    t_start = time.time()

    for L in args.L:
        filename = str(outdir / f"henon_L{L}.h5")
        run_experiment(
            map_name       = 'henon',
            filename       = filename,
            param_values   = a_values,
            fixed_params   = fixed_params,
            L              = L,
            transient      = args.transient,
            D_array        = args.D,
            bin_array      = args.bins,
            noise_values   = args.noise,
            n_workers      = n_workers,
            sim_seed       = args.seed,
            parameter_name = 'a',
            metadata       = {
                'b_fixed':  args.b_fixed,
                'a_min':    args.a_min,
                'a_max':    args.a_max,
                'transient': args.transient,
            },
        )

    logging.info(f"All L values complete. Total wall time: "
                 f"{(time.time()-t_start)/60:.1f} min")


if __name__ == '__main__':
    main()
