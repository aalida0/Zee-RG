# AF campaign (current)
1. `bash setup.sh` once on login.af.uchicago.edu (builds venv + HiggsTools; needs `~/zee_small.tar.gz` with `zee_data/_nufit_cache`).
2. `bash make_run.sh RUN_ID COMMIT` (frozen checkout), then `python3 make_campaign_sub.py RUN_ID pilot` (20 chains) -> `campaign.sub`; submit, inspect with the collector, and only then `python3 make_campaign_sub.py RUN_ID full` (2000 chains).
3. `condor_submit campaign.sub`; outputs `$HOME/zee_env/runs/RUN_ID/out/camp_RUN_<mode>_<logL>_<seed>.json` (atomic writes, `complete` flag, manifest with code/data hashes).
4. Copy `runs/RUN_ID/out/` here and run `ZEE_INPUTS=modern python3 -W ignore gate2/collect_campaign.py OUTDIR`.
Starts: `results/campaign_starts.json` holds five gate-valid starts (index 0,1 = BM1, BM2; 2-4 from `gate2/make_starts.py`, m_A 0.73-1.25 TeV); the campaign explores around these only, and convergence (split R-hat, ESS) is judged per start, never pooled across starts.
Chain length is fixed at CAMP_NPROP=8000 proposals (about 0.1 s per proposal incl. certification on the dev machine = ~15 min of the 23-min wall); if the pilot shows chains marked INCOMPLETE (wall limit) on the worker nodes, lower CAMP_NPROP in `campaign_job.sh` or raise WALL in `make_campaign_sub.py` before the full run.
Pilot checklist (20 chains): every chain `complete`, no `start fails the gate`/HiggsTools errors (exit 3/4), acceptance of post-burn-in proposals 0.2-0.7, stored points > 20 per sample chain, collector prints verified/meet-cutoff counts and finalists.
Each chain applies the staged gate (gate2/campaign.py); without the Higgs module points are marked UNRESOLVED and the collector reruns the full gate.
`legacy/` holds the earlier submit files.

## What a chain does (gate2/campaign.py)
* `sample`: Metropolis walk with adaptive-covariance Gaussian proposals (learned in the first 25% of the wall time, then frozen), proposals outside the
  parameter box rejected. Stationary measure: uniform in the sampling coordinates on the region R_L = {staged gate, coarse default-input cutoff >= LOGL}.
  Every 40th accepted post-burn-in step is stored and certified over 5 matching variants (gate2/cert.py).
* `max:OBS`: hill climb; a new best point is recorded only if its certified scale (5 variants) is >= LOGL; the walk uses LOGL+1 as a prefilter.
* The collector reports split R-hat and ESS per (logL, start) basin, per-basin fractions with a chain-bootstrap interval, re-verifies with the
  full gate (HiggsTools, 200-start vacuum search, 13-variant certification) and writes `results/campaign_finalists.json` for
  `BM_INPUT=results/campaign_finalists.json BM_OUTPUT=results/campaign_bench.json python3 gate2/benchmarks.py` (100 scales/decade, 17 pieces).

## Speed test and tan(beta) profile on worker nodes (writable clone)
```
Z=$HOME/zee_env; git clone https://github.com/aalida0/Zee-RG.git $Z/work && cp -r $Z/zee_data/_nufit_cache $Z/work/zee/   # once; later: git -C $Z/work pull
cd $Z/work/condor
condor_submit node_speed.sub        # prints CPU model, evaluate and gate times for 1 and 10 processes -> $Z/logs/speed.out.<cluster>
condor_submit node_profile.sub      # 3 jobs x 3 tan(beta) values (10 cores each); results in $Z/work/results/refit_hw_tb<value>.json
```
`+queue = "short"` and the 23-minute wall of the campaign suggest a short wall limit: if the profile jobs are killed, submit with fewer tan(beta) values per job.
