# HANDOFF: items from the V12 post-scan referee report that need execution

Status of the edits made in the "do not run anything" pass: source and manuscript
text were edited only. Nothing below has been run, tested, compiled or committed.
`ZEE_INPUTS=modern` throughout; heavy jobs go to UChicago AF/Condor; validation
runs need `ZEE_CACHE=<writable dir>`.

## 0. First, verify the edited code
1. `pytest tests/test_validate_finalist.py tests/test_matched_scales.py tests/test_prod1_results.py`
   (new fail-closed classification: `cert.variant_status`/`eligible`,
   `matched_scales.summarize`, `yukawa_run.summarize_variants`, rewritten
   `validate_finalist.py`). Expect to fix small bugs: none of it has been executed.
2. `python gate2/make_table_prod1.py` -> paste regenerated rows into Table `tab:bm`
   of `main.tex` (the body is stale; the caption was already rewritten).
3. `python gate2/make_fig_prod1.py` -> new `figures/fig_prod1.pdf` (stale until run).
4. Compile `main.tex`; only then send the PDF.

## 1. Revalidation (P0 3.1-3.4)
- BM1 through the current common gate (`benchmarks.py`, same code/data hashes as BP3).
- `validate_finalist.py` for BM1, BM2, BP1, BP2, BP3 (+C43): 13 fine-grid tree
  variants, all 7 combined matching+all-Yukawa modes (`--yukawa-modes`), exact
  coverage assessment; `--reassess FILE` on the existing records.
  Check that the statuses quoted in the draft are unchanged (BM1/BP3 robust;
  BP1/BP2/BM2 EP-clean only; BM1 stress-inclusive minimum 10^11.37, BP3 10^11.31).
- Combined run for the four prescriptions not yet done for BP3 (draft says so).

## 2. Screening
- EP screen of the 22 finalists not flagged by the proxy.
- Full matching of a Pareto shortlist: C28, C41, C53, C56, C58, the best mu->e
  candidates, and BM1.

## 3. Physics checks listed under "Items not done"
- Two-loop PyR@TE for BM1 and BP3 with nominal and limiting matched boundaries.
- Input sensitivity for BM1, BP1-BP3: exact z95 = 1.95996, DESI-DR2 with a
  consistent refit, RPP 2026 STU, no-cosmology fit, latest HiggsTools data.
- Mechanism numerics: H/A decomposition of the internal-flip term, lambda5 variation,
  direction of the matched BFB minimum, paired fixed-low-energy comparisons
  (C25, C40, BM1, BP3). The new mechanism paragraph is worded as an expectation
  until this is done.
- Conservative mu->e conversion bound (coherent amplitude with non-dipole terms).
- High-mass charged-scalar Drell-Yan pair-rate estimate (> 500 GeV).

## 4. Reproducibility release
Manifest including the 106 top-up jobs, code/data hashes, full-precision parameter
vectors, regeneration commands. Note: production commit 646e3597 differs from the
current validation code; state this in the release.

## Update after the text re-scoping (main.tex rewritten, nothing run)
- Title, abstract, introduction, Results, Discussion, Conclusions and appendix remnants were rewritten.
  BP2 and C43 are no longer benchmarks (one disclosure sentence on C57/C43 remains); BP1 is conditional; BM2 is EP-clean only.
- The Table `tab:bm` body in main.tex was trimmed by hand from the OLD records (4 columns: BM1 BM2 BP1 BP3). Regenerate it with
  `python gate2/make_table_prod1.py > rows.tex` (defaults are now BP1 BP3) and `gate2/make_fig_prod1.py` (BP dict is now BP1, BP3);
  `tests/test_prod1_results.py` was updated for the 4-column header. All untested.
- Check after regeneration: the hand-written rows "conservative tree-level margin" (BM1 0.026, BM2 0.020, BP1 0.058, BP3 0.077)
  and "complete prescriptions" (7, 5, 6, 7) must agree with the generator output; BM2's "(24)" dagger count was inferred from the text.
- CORRECTION to the earlier referee analysis: the z95 factor (1.949 vs 1.95996) raises chi2 and chi2_min by about the same ~0.035,
  so margins change by ~0.003, not ~0.035. The thin BP3 margin (0.07) is not threatened by that alone; DESI-DR2, RPP-2026, no-cosmology
  and HiggsTools updates for BP1/BP3 still need to be run.
- Fixed factual slip: stress minima of the largest clean rates are mu->e gamma 10^13.1 (C42), tau->mu gamma 10^10.2 (C28), h->tau mu 10^11.6 (C53).
- Before submission: BM1 must be revalidated with the same code/data hashes as BP3 (text says its record is from the first release);
  recompute the refit minima 10.72/10.86/10.74 with the final universality likelihood (appendix says they predate it).

## Status after execution pass (2026-11-27)
- DONE: full test suite `ZEE_INPUTS=modern ZEE_CACHE=<dir> pytest tests` -> 181 passed, 1 skipped (one bug fixed: validate_finalist.assess
  now tolerates yukawa records without 'variants').
- DONE: Table `tab:bm` rows regenerated with `gate2/make_table_prod1.py` (BM1 BM2 BP1 BP3) and Fig. 1 with `gate2/make_fig_prod1.py` (legend fixed);
  both pasted into the draft; the draft compiles. New numbers from the regenerated table: smallest matched V_red margins BM1 0.015, BM2 -0.123, BP1 -0.003, BP3 0.025.
- STILL OPEN: revalidate BM1 (and BM2) with validate_finalist.py (13 fine-grid tree variants, 7 combined matching+all-Yukawa modes; about 30-40 min per point on 2 cores);
  input updates for BP1/BP3; EP screen of the 22 unscreened finalists; two-loop for BM1/BP3 with matched boundaries.


## Input-sensitivity and BM1 revalidation (2026-11-27)
- BM1 revalidated with the current `gate2/validate_finalist.py` (7/7 prescriptions complete, robust scale 10^11.369 GeV incl. combined all-Yukawa, min matched margin 0.0149); Table 1 rows identical to the old record. BM2 not re-run (first-release record).
- `gate2/input_sensitivity.py` (+ `ZEE_STU=rpp2026` hook in `zee/hgmodel.py`) -> `results/input_sensitivity.json`: margins chi2_min+3.84-chi2 per variation with a refit minimum for each (`refit_hw_{z95,rpp2026,nocosmo,allnew}.json`, local refits from the 21 stored minima, NPERT=0; DR2 is the earlier 63-start refit).
  - exact z95: BM1/BM2/BP1/BP3 +0.85/+0.53/+0.52/+0.07; DR2: +0.79/+0.47/+0.46/+0.007; no cosmology: +0.84/+0.52/+0.50/+0.07.
  - RPP-2026 oblique fit (stu.py constants, Table 10.8; re-verification fetch of the PDF was blocked by a redirect loop): BM1 -0.81, BM2 -7.5 (outside), BP1 +0.37, BP3 +0.80. All three updates together: -0.80/-7.54/+0.37/+0.80.
  - The earlier draft statement "RPP 2026 oblique chi2 0.3-1.2" was wrong (actual 3.9/11.6/2.4/1.6 for BM1/BM2/BP1/BP3); corrected.
- HiggsTools: hbdataset master (3539f9a) and hsdataset v1.1 (3c6c253) are the remote release heads. hbdataset `develop` (d709915, Mar 2026) adds ATLAS 138/fb h->mutau/etau (2302.05225) stored as 1.8e-4/2.0e-4 but the paper says 0.18%/0.20%; as stored BP1/BP3 come out excluded (h ratio 1.27/1.10), with the limit x10 all four allowed (0.13/0.11). Worth reporting upstream.
- Still open: refit of BM1/BM2 under RPP-2026; BM2 revalidation; EP screen of 22 finalists; two-loop for BM1/BP3.
