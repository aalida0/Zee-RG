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
   h->e tau < 0.20%, HFLAV 2023 (g_tau/g_mu) = 1.0011(14), KATRIN 0.45 eV, DESI sum m < 0.072 eV (LCDM-specific), NuFIT 6.1, Delta a_mu = 3.8(6.3)e-10 (WP25, arXiv:2505.21476).

## Status (2026-11-24)
- Best fits, modern inputs: chi2min NO 11.4, IO 33.6 (IO disfavoured by ~22, mostly DESI sum m).
- Quartic stability filter: ~25% of NO survives to 1e4 GeV, ~4% to 1e8, ~0.3% to 1e16; HG's fixed lambda2 has a Landau pole near 1 TeV.
- Ranges depend on the sampling measure: random-walk region vs nested-sampling credible interval differ by up to an order of magnitude
  (e.g. Br(tau->mu gamma) max 2.7e-8 vs 95% upper 2.8e-9). Mass range (NO) 2.25 TeV (NS, 300 live points) vs HG 1.7 TeV: unresolved.
- Open: converge NS (more live points, independent runs), stability filter on weighted samples, IO NS, Cr formula, tests for hgmodel/bz/rg*.

## Profile region vs nested sampling (modern inputs, NO)
Nested sampling is not converged (300 vs 800 live points: log Z -53.4 vs -48.8, m_A 95% upper 2.2 vs 3.5 TeV); do not quote credible intervals.
Profile region (Delta chi2 <= 3.84 above 11.4, multistart from 480 polished seeds incl. NS samples, then random walk) is stable under added seeds:
Br(tau->mu gamma) 2e-11..1.6e-8, Br(mu->e gamma) up to 1.5e-13, Br(h->tau mu) 2.5e-10..1.6e-3, m_A 0.26..3.4 TeV, tan(beta) 0.3..0.73.
No seed beat chi2 = 11.4, so it is the likely global minimum.

## Off-slice test: lambda4 free (stop-loss, 2026-11-24)
`hgmodel` accepts an optional 19th parameter x[18] = lambda4 (default 0 = HG slice; m_A^2 = M_H+^2 + (lambda4-lambda5) v^2/2).
Profile region with |lambda4| <= sqrt(4 pi): chi2min 10.1 (slice 9.8, same inputs); Br(tau->mu gamma) up to 1.2e-8, Br(mu->e gamma) up to 1.3e-13,
Br(h->tau mu) up to 1.5e-3, m_A up to 3.8 TeV. Stability: EW-BFB 44%, 43% to 1e4 GeV, 2.7% to 1e8 GeV, 0 to 1e16 GeV (600 pts).
The LFV signals and the stability cut survive freeing lambda4. Walk explored |lambda4| <~ 1.1 only (step size); full range not yet scanned.

## Future-experiment reach (NO, lambda4 free, 600 profile-region points; docs/reach.png)
mu->3e is a dipole-dominated estimate (alpha/3pi (ln(m_mu^2/m_e^2) - 11/4) Br(mu->e gamma)); Cr(Al) adds the dipole-dominance term alpha/3 Br (Au: alpha/2) incoherently to the HG monopole term; median CR(Al)/Br = 3.2e-3; monopole formula still unvalidated against HG.
Muonium-antimuonium needs Y_emu^2 at tree level, which vanishes on this slice (not computed). Fraction of region above each future reach
(all / stable to 1e4 GeV): MEG II 6e-14: 6% / 5%; Mu3e II 1e-16: 18% / 14%; Mu2e-COMET II 1e-16 (Al, CR = monopole + alpha/3 Br(mu->e gamma)): 18% / 16% (6e-17: 20% / 18%); Mu3e I 1e-15, COMET I 7e-15: 0%; Belle II tau->mu gamma 1e-9: 24% / 26%;
Belle II tau->3mu 5e-10: 0%; h->tau mu > 1e-4: 33% / 23%. Reach numbers from the literature (MEG II 6e-14; Mu3e 1e-15..1e-16; Mu2e/COMET 1e-16..1e-17; Belle II tau->mu gamma 1e-9), (MEG II 6e-14, Mu3e I 1e-15 / II 1e-16, Mu2e/COMET II ~6e-17-1e-16, Belle II tau->mu gamma 1e-9, tau->3mu ~5e-10; checked against literature summaries, not experiment documents). Higgs signals: HiggsTools 1.3.0 (built from gitlab.com/higgsbounds/higgstools, datasets hbdataset/hsdataset; `gate1/higgstools_chi2.py`) replaces the HG-era signal-strength term for post-fit filtering: HiggsSignals chi2 (159 obs, SM = 153.8) and HiggsBounds for H, A with SM-like decays only (charged scalars not tested; missing exotic decays make limits conservative).

## HiggsTools filter (NO, lambda4 free, 600 points)
HS chi2 min 151.8 (SM 153.8); 1.5% of points have Delta chi2 > 3.84 from the region minimum; HiggsBounds excludes 3.2% (m_H ~ m_A ~ 320-350 GeV at tan(beta) = 0.3, ttbar searches).

## Usage
`ZEE_INPUTS=modern python gate1/slog.py NO 11 out.pkl`; see script headers. Scripts read/write pickles in /tmp.
