#!/bin/bash
# targeted tan(beta) profile from one start on a worker node: profile_tb.sh START [TB list]  (results/profile_tb/START.json in the writable clone)
Z=$HOME/zee_env; cd $Z/work || exit 2
export DIRECT=1 ZEE_INPUTS=modern ZEE_DATA=$Z/zee_data OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
$Z/venv/bin/python -W ignore gate2/profile_tb.py "$@"
