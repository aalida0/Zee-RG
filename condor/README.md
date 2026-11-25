# AF campaign (current)
1. `bash setup.sh` once on login.af.uchicago.edu (builds venv + HiggsTools; needs `~/zee_small.tar.gz` with `zee_data/_nufit_cache`).
2. `python3 make_campaign_sub.py [WALL=1380]` -> `campaign.sub` (2000 chains; edit the loops to change the mix).
3. `condor_submit campaign.sub`; outputs `$HOME/zee_env/out3/camp_<mode>_<logL>_<seed>.json`.
4. Copy `out3/` here and run `ZEE_INPUTS=modern python3 -W ignore gate2/collect_campaign.py OUTDIR`.
Each chain applies the staged gate (gate2/campaign.py); without the Higgs module points are marked UNRESOLVED and the collector reruns the full gate.
`legacy/` holds the earlier submit files.
