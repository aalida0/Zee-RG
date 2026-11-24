#!/bin/bash
# args: OBS LOGL SEED WALL
Z=$HOME/zee
export ZEE_INPUTS=modern ZEE_DATA=$Z/zee_data
ln -sfn $Z/zee_data/hbdataset /tmp/hbdataset; ln -sfn $Z/zee_data/hsdataset /tmp/hsdataset
cd $Z/repo
exec $Z/venv/bin/python -I gate1/envopt.py "$1" "$2" "$3" "$4" $Z/out
