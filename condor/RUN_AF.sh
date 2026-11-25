#!/bin/bash
# UChicago AF: run from $HOME. Usage: bash zee_fit/RUN_AF.sh   (24 cores, ~15 min)
source ~/zee_env/bin/activate 2>/dev/null
cd ~/zee_fit && ZEE_INPUTS=modern python3 -W ignore gate2/reopt_bench.py 24 96 BM1 BM2 BM3 2>&1 | grep -v HiggsPredictions | tee reopt.log
