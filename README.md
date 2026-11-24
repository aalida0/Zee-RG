# Zee-RG: general Zee model global fit with RG running (work in progress)

Anomaly-agnostic, post-MEG II fit of the general Zee model, with one-loop RG
running and stability (BFB/perturbativity) checks. Gate 1 reproduces Herrero-García
et al. (HG), JHEP04(2017)130, with HG's 2016 inputs; `ZEE_INPUTS=modern` switches to 2025-26 inputs.

## Layout
- `zee/` model code: `scalars`, `neutrino`, `observables`, `dipoles`, `bz` (Barr-Zee), `stu`, `nufit`
  (NuFIT 6.1 chi2 tables; the cache `zee/_nufit_cache` is built by `nufit.build_cache` from the NuFIT files, not committed),
  `hgmodel` (18-parameter HG-slice likelihood), `rge`/`rgeq`/`rgfast` (Yukawa and quartic RGEs), `bfb`.
- `gate1/` scripts: `slog.py` (signed-log multistart fit), `polish.py`, `walk2.py` (random-walk region),
  `ns.py` (dynesty nested sampling, HG-style priors), `final_filter4.py` (RG stability filter). `m_*.py` are
  modern-input copies of the same scripts with different pickle paths.
- `tests/` pytest suite for the physics modules (none yet for hgmodel/bz/rg*).

## Assumptions (read before using results)
1. HG slice: f_emu = 0; Y2^{ee,emu,mue,mumu} = 0; f and Y2^{etau} real; lambda4 = lambda7 = lambda8 = lambda9 = lambda10 = lambda_h = 0;
   lambda2 = sqrt(4 pi); m_h = 125.5 GeV; lambda6 derived. Free: lambda3, lambda5, lambda6, mixing.
2. Potential: the 1/2 multiplies lambda5 only (M12 = lambda6 v^2), as verified against HG.
3. Neutrino masses at leading order (`mnu_hg_leading`), HG convention. Diagonal dipoles use physical normalisation (HG 3.21/3.23 differ by 2).
4. mu-e conversion uses the monopole term only (HG 3.24-3.25); the extracted formula was garbled, so Cr is NOT validated against HG.
5. Higgs signal strengths are approximate (fixed SM BRs, 87/13 production split); Higgs inputs are still HG-era.
6. Priors: signs and log-uniform |x| in [1e-12, 0.1] for f and Y2; log-uniform tan(beta) in [0.3, 50] (a guess); log10 masses in [2, 4.3]; quartics uniform.
7. RG: one-loop (Machacek-Vaughn), rough gauge/top inputs; `rgfast` is an 11-parameter closed form matching the full tensor to ~1%, leptonic Yukawas dropped.
   Stability = bounded from below + perturbativity (|lambda| <= 4 pi) at each scale. Yukawa running is negligible in the HG region.
8. Modern inputs (`ZEE_INPUTS=modern`): MEG II mu->e gamma < 1.5e-13, Belle II tau->3mu < 1.9e-8, tau->mu gamma < 4.2e-8 (Belle), ATLAS h->mu tau < 0.18%,
   h->e tau < 0.20%, HFLAV 2023 (g_tau/g_mu) = 1.0011(14), KATRIN 0.45 eV, DESI sum m < 0.072 eV (LCDM-specific), NuFIT 6.1, Delta a_mu = 3.7(6.4)e-10 (typed from memory, unverified).

## Status (2026-11-24)
- Best fits, modern inputs: chi2min NO 11.4, IO 33.6 (IO disfavoured by ~22, mostly DESI sum m).
- Quartic stability filter: ~25% of NO survives to 1e4 GeV, ~4% to 1e8, ~0.3% to 1e16; HG's fixed lambda2 has a Landau pole near 1 TeV.
- Ranges depend on the sampling measure: random-walk region vs nested-sampling credible interval differ by up to an order of magnitude
  (e.g. Br(tau->mu gamma) max 2.7e-8 vs 95% upper 2.8e-9). Mass range (NO) 2.25 TeV (NS, 300 live points) vs HG 1.7 TeV: unresolved.
- Open: converge NS (more live points, independent runs), stability filter on weighted samples, IO NS, Cr formula, tests for hgmodel/bz/rg*.

## Usage
`ZEE_INPUTS=modern python gate1/slog.py NO 11 out.pkl`; see script headers. Scripts read/write pickles in /tmp.
