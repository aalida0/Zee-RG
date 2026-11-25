#!/bin/bash
# tan(beta) profile on a worker node: node_profile.sh NPROC TB1 TB2 ...   (results/refit_hw_tb<value>.json in the writable clone)
Z=$HOME/zee_env; cd $Z/work || exit 2
export ZEE_INPUTS=modern ZEE_DATA=$Z/zee_data OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
NP=$1; shift
for tb in "$@"; do TB_FIX=$tb TAG=tb$tb $Z/venv/bin/python -W ignore gate2/refit_hw.py $NP 2; done
