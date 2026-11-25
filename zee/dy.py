"""Drell-Yan pair production of the charged scalars, pp -> h_k+ h_k-, against LHC searches for l nu l nu (+ tau) final states.

Physics input.  A weak-singlet charged scalar (SCS) has the quantum numbers of a right-handed slepton, so its pair cross
section is sigma_R (13 TeV NLO+NLL, Fuks et al. 1310.2621 Table 3; also the LHC SUSY cross-section Twiki, CT10 table).  The
published recast of the 139/fb ATLAS slepton search to an SCS (Das, Howe, Shuve, Tucker-Smith, Yager, PRD 112, 055003,
arXiv:2506.05258) gives the excluded mass ranges for several branching ratios (B_e, B_mu, B_tau):
    B_e = 1           107 - 425 GeV      (LEP covers 100 - 107)
    B_mu = 1          93.5 - 450 GeV     (LEP covers < 97.3)
    B_tau = 1         ~100 - 350 GeV     (ATLAS stau search; LEP < 90.1)
    (1/4, 1/4, 1/2)   125 - 185 GeV      (allowed: 80 - 125 and > 185)
    B_mu = B_tau = 1/2  ~120 - 255 GeV   (rough estimate in the paper)
We turn this into a flavour-resolved screen: a weight w = (B_mu^2 + 0.85 B_e^2) sigma(m,d)/sigma_R(m) is compared with a
mass-dependent threshold W_THR(m) interpolated through the published exclusion edges (it reproduces the e, mu and mixed edges
used in the construction, see the test; the 0.85 for electrons reproduces the 425 GeV edge).  The tau channel is a window rule between the two published points: excluded for
100 <= m <= 350 GeV when B_tau^2 > 1/2 (published: allowed at B_tau = 1/2, excluded at B_tau = 1).
Approximations (documented, not a full recast): the limit is a function of the same-flavour weight only (e mu and ll-tau
channels neglected); the e-mu-tau interpolation between the published benchmarks is not a measurement; each charged state is
screened separately; the doublet fraction d enters through sigma = (1-d) sigma_R + d sigma_L, which neglects gamma-Z
interference, so the screen is UNRESOLVED when m < 500 GeV and d > 0.3.  Masses above 500 GeV (end of the cross-section table) are screened as passing (sigma_L(500) = 0.47 fb).
A pass is a statement about this screen, not an experimental certification."""
import numpy as np

_M = np.array([100., 150., 200., 250., 300., 350., 400., 450., 500.])
_SL = np.array([270.79, 63.34, 21.81, 9.21, 4.43, 2.33, 1.31, 0.77, 0.47])   # fb, l_L+ l_L-, 13 TeV NLO+NLL
_SR = np.array([96.51, 23.32, 8.15, 3.47, 1.68, 0.89, 0.50, 0.30, 0.18])      # fb, l_R+ l_R-


def _interp(m, s):
    ls = np.log(s)
    if m <= _M[0]:
        return float(np.exp(ls[0] + (m - _M[0])*(ls[1] - ls[0])/(_M[1] - _M[0])))
    return float(np.exp(np.interp(m, _M, ls)))


def sigma_dy(m, d):
    """sigma(pp -> h+ h-) [fb] per flavour-equivalent for doublet fraction d; 0 above 500 GeV."""
    if m > _M[-1]: return 0.0
    d = min(max(float(d), 0.0), 1.0)
    return (1 - d)*_interp(m, _SR) + d*_interp(m, _SL)


def _sig_R(m):
    return _interp(m, _SR)


# effective same-flavour limit anchors (GeV, fb): the cross section sigma_R * w at each published exclusion edge, w = B_e^2 + B_mu^2
# Weight-threshold model fitted to the published exclusion edges: the point is excluded when
#     w_eff = [B_mu^2 + KAPPA_E B_e^2] * sigma(m, d)/sigma_R(m)  >=  W_THR(m).
# Anchors (GeV, w): B_mu = 1 edges (93.5 and 450 GeV), B_e = 1 upper edge (425 GeV, fixes KAPPA_E), mu+tau 1/2:1/2 (120, 255 GeV,
# w = 1/4), and the (1/4, 1/4, 1/2) benchmark (125, 185 GeV, w = 1/16 + KAPPA_E/16).  Log-linear interpolation of ln W_THR.
KAPPA_E = 0.85
_A = [(93.5, 1.0), (120., 0.25), (125., 1/16 + KAPPA_E/16), (185., 1/16 + KAPPA_E/16), (255., 0.25), (450., 1.0)]
_AM = np.array([m for m, w in _A]); _AW = np.array([w for m, w in _A])


def w_thr(m):
    """threshold weight; None below 93.5 GeV (no LHC sensitivity, LEP applies); extrapolated above 450 GeV."""
    if m < _AM[0]: return None
    if m <= _AM[-1]: return float(np.exp(np.interp(m, _AM, np.log(_AW))))
    sl = (np.log(_AW[-1]) - np.log(_AW[-2]))/(_AM[-1] - _AM[-2])
    return float(np.exp(np.log(_AW[-1]) + sl*(m - _AM[-1])))


TAU_WINDOW = (100., 350.); TAU_B2 = 0.5


def screen_state(m, d, be, bmu, btau):
    """One charged mass eigenstate. Returns dict(status, w_eff, w_thr, tau, reason)."""
    if m > _M[-1]:
        return dict(status='PASS', w_eff=0.0, w_thr=None, tau=False, reason='above 500 GeV')
    if d > 0.3:
        return dict(status='UNRESOLVED', w_eff=None, w_thr=None, tau=False, reason='doublet fraction > 0.3: linear interpolation not trusted')
    w = (bmu**2 + KAPPA_E*be**2)*sigma_dy(m, d)/_sig_R(m); thr = w_thr(m)
    tau = (TAU_WINDOW[0] <= m <= TAU_WINDOW[1]) and (btau**2 > TAU_B2)
    bad = thr is not None and w >= thr*(1 - 1e-9)
    return dict(status='FAIL' if (bad or tau) else 'PASS', w_eff=float(w), w_thr=thr, tau=bool(tau),
                S_fb=float(sigma_dy(m, d)*(be**2 + bmu**2)), reason=('same-flavour' if bad else '') + (' tau-window' if tau else ''))


def decide(masses, dfracs, bre, brmu, brtau):
    """All charged states; FAIL if any state fails; UNRESOLVED if none fails but one is unresolved."""
    st = [screen_state(m, d, a, b, c) for m, d, a, b, c in zip(masses, dfracs, bre, brmu, brtau)]
    s = [x['status'] for x in st]
    return dict(status='FAIL' if 'FAIL' in s else ('UNRESOLVED' if 'UNRESOLVED' in s else 'PASS'), states=st,
                S_fb=float(sum(x.get('S_fb') or 0.0 for x in st)))
