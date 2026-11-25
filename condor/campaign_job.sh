#!/bin/bash
# args: MODE LOGL SEED WALL START      (MODE = sample | max:OBS)
Z=$HOME/zee_env
export ZEE_INPUTS=modern ZEE_DATA=$Z/zee_data
mkdir -p $Z/out3
cd $Z/repo
exec $Z/venv/bin/python -I -W ignore gate2/campaign.py "$1" "$2" "$3" "$4" $Z/out3 "$5"
