#!/bin/bash
# speed test on a worker node (uses the writable clone $HOME/zee_env/work, see README)
Z=$HOME/zee_env; cd $Z/work || exit 2
export ZEE_INPUTS=modern ZEE_DATA=$Z/zee_data OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
hostname; lscpu | grep -E "Model name|^CPU\(s\)"
$Z/venv/bin/python -W ignore gate2/bench_speed.py ${1:-10}
