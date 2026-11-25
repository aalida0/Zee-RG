"""Drell-Yan pair production of the charged scalars, pp -> h_k+ h_k-, confronted with the LHC slepton-pair searches.

Why a recast: the HiggsBounds dataset has only LEP limits for H+H-.  The LHC searches for l+ l- + MET (massless
invisible state) constrain a charged scalar that decays to l nu, because (h+ h- -> l nu l nu) has the same topology as
(slepton pair -> l l chi0 chi0) with a massless neutralino.

Inputs: 13 TeV NLO+NLL slepton-pair cross sections per flavour (Fuks, Klasen, Lamprea, Rothering, arXiv:1310.2621,
Table 3).  A weak-singlet charged scalar (hypercharge coupling only) is the right-handed slepton; a weak-doublet one
(the Higgs-basis H+) is the left-handed slepton.  For a mixture with doublet fraction d we interpolate linearly,
sigma = (1-d) sigma_R + d sigma_L (the gamma/Z interference is neglected: approximation, tested at the end points).
For h1+ = cos(phi) H+ + sin(phi) h+ the doublet fraction is cos^2(phi); for h2+ it is sin^2(phi).

Limit (approximation, documented): ATLAS excludes right-handed e/mu sleptons with massless LSP up to ~420 GeV with 139/fb
(arXiv:1908.08215; quoted e.g. in arXiv:2103.15328), i.e. 2 sigma_R(420) = 0.82 fb for the e+mu (same-flavour) sum at
100% branching ratio.  Taking this cross-section limit as flat in the mass is conservative (over-excluding) at low mass,
where the real limit is weaker.  The signal is S = sum_k sigma_k (B_e,k^2 + B_mu,k^2) (same-flavour pairs) and the point is
excluded when S > S_LIM.  tau nu final states are not constrained (stau searches are weaker); this is why
Babu et al. (arXiv:1907.09498) find the region evaded for large BR(tau nu).
Masses above 500 GeV (table end) can never reach S_LIM and are given S=0 (sigma_L(500)=0.47 fb < 0.82 fb)."""
import numpy as np
_M = np.array([100., 150., 200., 250., 300., 350., 400., 450., 500.])
_SL = np.array([270.79, 63.34, 21.81, 9.21, 4.43, 2.33, 1.31, 0.77, 0.47])   # fb, l_L+ l_L-, 13 TeV NLO+NLL
_SR = np.array([96.51, 23.32, 8.15, 3.47, 1.68, 0.89, 0.50, 0.30, 0.18])      # fb, l_R+ l_R-
S_LIM = 0.82                                                                 # fb, = 2 sigma_R(420)  (see docstring)


def _interp(m, s):
    ls = np.log(s)
    if m <= _M[0]:   # log-linear extrapolation of the first interval (LEP removes m < ~100 anyway)
        return float(np.exp(ls[0] + (m - _M[0])*(ls[1] - ls[0])/(_M[1] - _M[0])))
    return float(np.exp(np.interp(m, _M, ls)))


def sigma_dy(m, d):
    """sigma(pp -> h+ h-) [fb] per flavour-equivalent, 13 TeV, for doublet fraction d in [0,1]; 0 above 500 GeV."""
    if m > _M[-1]: return 0.0
    d = min(max(float(d), 0.0), 1.0)
    return (1 - d)*_interp(m, _SR) + d*_interp(m, _SL)


def dy_signal(masses, dfracs, bre, brmu):
    """S = sum_k sigma_k (B_e^2 + B_mu^2) [fb]; arguments are sequences over the charged mass eigenstates."""
    return sum(sigma_dy(m, d)*(be**2 + bm**2) for m, d, be, bm in zip(masses, dfracs, bre, brmu))


def dy_allowed(masses, dfracs, bre, brmu):
    S = dy_signal(masses, dfracs, bre, brmu)
    return bool(S <= S_LIM), float(S)
