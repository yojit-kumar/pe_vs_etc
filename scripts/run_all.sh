#!/usr/bin/env bash
# =============================================================================
# run_all.sh — orchestrate all three map sweeps sequentially.
#
# Each generate_*.py script already saturates available cores via its internal
# multiprocessing Pool, so running the three scripts sequentially (not in
# background) is the right strategy: running them concurrently would cause
# n_workers × 3 processes to fight for the same CPU cores, degrading throughput.
#
# Usage
# -----
#   chmod +x scripts/run_all.sh
#   ./scripts/run_all.sh                        # defaults for all three maps
#   ./scripts/run_all.sh --workers 16           # override core count
#   ./scripts/run_all.sh --quick                # fast smoke test (small N, L)
#
# The script passes --workers and --outdir to all three generate_*.py scripts.
# Map-specific options (--a-min, --c-max, etc.) can be customised directly in
# the LOGISTIC / HENON / ROSSLER config sections below.
#
# Environment
# -----------
# Activate your conda / venv environment before running:
#   conda activate etc && ./scripts/run_all.sh
#
# Exit behaviour
# --------------
# Any failed script (non-zero exit code) aborts the whole run immediately.
# =============================================================================

set -euo pipefail      # exit on error, unbound var, or pipe failure

# ── Defaults ──────────────────────────────────────────────────────────────────
WORKERS=""             # empty → each script uses all available cores
OUTDIR="data"
QUICK=0                # 0 = production run, 1 = smoke test

# ── Parse arguments ───────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --workers)
            WORKERS="--workers $2"; shift 2 ;;
        --outdir)
            OUTDIR="$2"; shift 2 ;;
        --quick)
            QUICK=1; shift ;;
        -h|--help)
            head -n 40 "$0" | grep '^#' | sed 's/^# \?//'
            exit 0 ;;
        *)
            echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# ── Resolve project root (works regardless of where the script is called from)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$ROOT_DIR"

# ── Per-map configurations ────────────────────────────────────────────────────
# Edit these blocks to change parameter ranges, L values, etc.
# All flags here are passed verbatim to the corresponding generate_*.py script.

if [[ $QUICK -eq 1 ]]; then
    # ── Smoke-test settings (completes in seconds) ──────────────────────────
    LOGISTIC_ARGS="--n-param 30 --L 5000 --D 3 5 --bins 2 3 --noise 0.0 0.05"
    HENON_ARGS="--n-param 30   --L 5000 --D 3 5 --bins 2 3 --noise 0.0 0.05"
    ROSSLER_ARGS="--n-param 20 --L 2000 --D 3 5 --bins 2 3 --noise 0.0 0.05"
    echo ">>> QUICK mode: smoke-test parameters (small N and L) <<<"
else
    # ── Production settings ─────────────────────────────────────────────────
    LOGISTIC_ARGS="
        --a-min 3.5 --a-max 4.0
        --n-param 1000
        --L 1000000
        --transient 1000
        --D 3 4 5 6 7
        --bins 2 4 6 8
        --noise 0.0 0.01 0.05 0.1
    "
    HENON_ARGS="
        --a-min 1.0 --a-max 1.4
        --n-param 1000
        --b-fixed 0.3
        --L 1000000
        --transient 5000
        --D 3 4 5 6 7
        --bins 2 4 6 8
        --noise 0.0 0.01 0.05 0.1
    "
    ROSSLER_ARGS="
        --c-min 2.0 --c-max 8.0
        --n-param 1000
        --a-fixed 0.2 --b-fixed 0.2
        --dt 1.0 --transient-time 3000.0
        --L 100000 1000000
        --D 3 4 5 6 7
        --bins 2 4 6 8
        --noise 0.0 0.01 0.05 0.1
    "
fi

# ── Helpers ───────────────────────────────────────────────────────────────────
PYTHON="${PYTHON:-python}"     # override with PYTHON=python3 ./scripts/run_all.sh

log_banner() {
    local msg="$1"
    local len=${#msg}
    local line
    line=$(printf '=%.0s' $(seq 1 $((len + 4))))
    echo ""
    echo "$line"
    echo "  $msg"
    echo "$line"
}

timestamp() { date '+%Y-%m-%d %H:%M:%S'; }

run_script() {
    local label="$1"
    local script="$2"
    shift 2
    local args=("$@")

    log_banner "$label"
    echo "Start: $(timestamp)"
    echo "Command: $PYTHON $script ${args[*]}"
    echo ""

    time $PYTHON "$script" "${args[@]}"
    local exit_code=$?

    echo ""
    echo "End: $(timestamp)"
    if [[ $exit_code -ne 0 ]]; then
        echo "ERROR: $label exited with code $exit_code. Aborting."
        exit $exit_code
    fi
    echo "$label COMPLETE."
}

# ── Main ──────────────────────────────────────────────────────────────────────
echo "Project root : $ROOT_DIR"
echo "Output dir   : $OUTDIR"
echo "Workers flag : ${WORKERS:-'(default: all cores)'}"
echo "Start time   : $(timestamp)"

mkdir -p "$OUTDIR"

# shellcheck disable=SC2086
run_script "Logistic Map" \
    "scripts/generate_logistic.py" \
    $LOGISTIC_ARGS $WORKERS --outdir "$OUTDIR"

# shellcheck disable=SC2086
run_script "Hénon Map" \
    "scripts/generate_henon.py" \
    $HENON_ARGS $WORKERS --outdir "$OUTDIR"

# shellcheck disable=SC2086
run_script "Rössler System" \
    "scripts/generate_rossler.py" \
    $ROSSLER_ARGS $WORKERS --outdir "$OUTDIR"

log_banner "ALL JOBS COMPLETE"
echo "Results saved in: $OUTDIR/"
echo "Total finish time: $(timestamp)"
