#!/bin/bash
# Usage: bash make_run.sh RUN_ID COMMIT      one-time, on login.af.uchicago.edu after setup.sh
# Creates $Z/runs/RUN_ID/repo as a read-only checkout of exactly COMMIT (no git pull), copies the NuFIT cache, records the package versions.
set -e
Z=$HOME/zee_env; RUN_ID=$1; COMMIT=$2; R=$Z/runs/$RUN_ID
[ -n "$RUN_ID" ] && [ -n "$COMMIT" ] || { echo "usage: make_run.sh RUN_ID COMMIT"; exit 1; }
[ ! -e "$R" ] || { echo "$R exists: use a new RUN_ID"; exit 1; }
mkdir -p "$R"; git clone -q https://github.com/aalida0/Zee-RG.git "$R/repo"; git -C "$R/repo" checkout -q "$COMMIT"
[ "$(git -C $R/repo rev-parse HEAD)" = "$(git -C $R/repo rev-parse $COMMIT)" ]
cp -r $Z/zee_data/_nufit_cache "$R/repo/zee/_nufit_cache"
$Z/venv/bin/pip freeze > "$R/pip_freeze.txt"; git -C "$R/repo" rev-parse HEAD > "$R/COMMIT"
for d in hbdataset hsdataset; do git -C $Z/zee_data/$d rev-parse HEAD > "$R/$d.COMMIT"; done
chmod -R a-w "$R/repo"
echo "run $RUN_ID frozen at $(cat $R/COMMIT)"
