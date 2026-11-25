# AF campaign (current)
1. `bash setup.sh` once on login.af.uchicago.edu (builds venv + HiggsTools; needs `~/zee_small.tar.gz` with `zee_data/_nufit_cache`).
2. `bash make_run.sh RUN_ID COMMIT` (frozen checkout), then `python3 make_campaign_sub.py RUN_ID pilot` (20 chains) -> `campaign.sub`; submit, inspect with the collector, and only then `python3 make_campaign_sub.py RUN_ID full` (2000 chains).
3. `condor_submit campaign.sub`; outputs `$HOME/zee_env/runs/RUN_ID/out/camp_RUN_<mode>_<logL>_<seed>.json` (atomic writes, `complete` flag, manifest with code/data hashes).
4. Copy `runs/RUN_ID/out/` here and run `ZEE_INPUTS=modern python3 -W ignore gate2/collect_campaign.py OUTDIR`.
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
