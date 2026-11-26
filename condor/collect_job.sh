#!/bin/bash
# Collector as a batch job (long queue). args: RUN_ID. Runs from the frozen checkout, writes everything to $HOME/zee_env/collect_RUN_ID.
set -e
Z=$HOME/zee_env; RUN_ID=${1:?RUN_ID}; R=$Z/runs/$RUN_ID; C=$Z/collect_$RUN_ID
mkdir -p "$C/plots"
export PYTHONDONTWRITEBYTECODE=1 ZEE_INPUTS=modern ZEE_DATA=$Z/zee_data EXPECT_FILE=$Z/condor/expected_jobs_full_$RUN_ID.json ZEE_OUT=$C ZEE_CACHE=$C/bench_cache PLOTDIR=$C/plots
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
cd "$R/repo"
exec $Z/venv/bin/python -W ignore gate2/collect_campaign.py "$R/out"
