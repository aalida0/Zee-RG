#!/bin/bash
# args: MODE LOGL SEED WALL START MARGIN      (MODE = sample | max:OBS);  environment: RUN_ID (set by the submit file)
# Runs from an IMMUTABLE per-run checkout ($Z/runs/$RUN_ID/repo, created by make_run.sh) and writes to $Z/runs/$RUN_ID/out.
set -e
Z=$HOME/zee_env
: "${RUN_ID:?RUN_ID is not set}"
R=$Z/runs/$RUN_ID
export ZEE_INPUTS=modern ZEE_REQUIRE_HIGGS=1 ZEE_DATA=$Z/zee_data RUN_ID OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
[ -d "$R/repo" ] || { echo "run directory $R/repo missing on $(hostname)" >&2; exit 2; }
$Z/venv/bin/python -I -c "import Higgs" 2>/dev/null || { echo "HiggsTools not importable on $(hostname): candidate mode only, refusing to run" >&2; exit 3; }
[ -r "$ZEE_DATA/hbdataset" ] || { echo "HiggsBounds dataset not readable on $(hostname)" >&2; exit 4; }
mkdir -p "$R/out"; cd "$R/repo"
exec $Z/venv/bin/python -I -W ignore gate2/campaign.py "$1" "$2" "$3" "$4" "$R/out" "$5" "${6:-1.0}"
