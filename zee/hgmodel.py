"""HG (JHEP04(2017)130) slice of the general Zee model and its full chi^2 (HG sec. 4, 2016 inputs).

Slice: f_emu=0, Y2^{ee,emu,mue,mumu}=0, f and Y2^{e tau} real, lam4=lam7=lam8=lam9=lam10=lam_h=0,
lam2 = sqrt(4 pi) (irrelevant except stability), m_h = 125.5 GeV fixed.
Parameter vector x (18): f_et, f_mt, Y2_et, Y2_tt(re,im), Y2_tm(re,im), Y2_te(re,im), Y2_mt(re,im),
  ln tb, log10 M_H+, log10 M_33, u (mu = u sqrt2 M_H+ M_33 / v), lam3, lam5, lam6.
lam1 follows from m_h = 125.5 (eq. 2.13); lam4 = 0 gives m_A^2 = M_H+^2 - v^2 lam5 / 2.
APPROXIMATE inputs (documented in the code): Higgs signal strengths (ATLAS+CMS Run-1 combination
from memory, not HG's refs [105-108]); HFLAV-Tau 2023 universality with its correlation matrix (MODERN; HG-era values uncorrelated).
"""
import numpy as np

from . import V_EW as V
from . import neutrino as N, scalars as S, observables as O, stu as T, bz as B, dipfull as DF
from .observables import M_LEP, GF, ALPHA, BR_TAU_LNUNU

MH = 125.5
ME_, MMU, MTAU = M_LEP
SQ4PI = np.sqrt(4 * np.pi)
NAMES = ["f_et", "f_mt", "Y_et", "Y_tt_re", "Y_tt_im", "Y_tm_re", "Y_tm_im", "Y_te_re", "Y_te_im",
         "Y_mt_re", "Y_mt_im", "ln_tb", "lMHp", "lM33", "u", "lam3", "lam5", "lam6"]
LO = np.array([-.1, -.1, -.1] + [-.1] * 8 + [np.log(.3), 2.0, 2.0, 0.0, -SQ4PI, -SQ4PI, -SQ4PI])
HI = np.array([.1, .1, .1] + [.1] * 8 + [np.log(50.), 4.3, 4.3, 0.999, SQ4PI, SQ4PI, SQ4PI])

OSC = {"NO": dict(s12=(0.306, .012), s13=(0.02166, .00075), s23=(0.441, .027), dm21=(7.50e-5, .19e-5),
                  dm3=(2.524e-3, .040e-3), d=(261., 59.)),
       "IO": dict(s12=(0.306, .012), s13=(0.02179, .00076), s23=(0.587, .024), dm21=(7.50e-5, .19e-5),
                  dm3=(-2.514e-3, .041e-3), d=(277., 46.))}
# Charged-charged term of T: the general Zee/multi-Higgs result is -2 w1 w2 F (Grimus et al. 0711.4022 Eq. 70); the printed HG expression has -w1 w2 F.
# The physical default is the general result; the printed HG form is kept only as a named historical reproduction (ZEE_STU_LEGACY_HG=1; V13 referee report P0).
import os as _os
STU_HG_LEGACY = _os.environ.get("ZEE_STU_LEGACY_HG") == "1"
STU_C = np.array([0.05, 0.09, 0.01]); STU_S = np.array([.11, .13, .11])
STU_COV = np.array([[1, .90, -.59], [.90, 1, -.83], [-.59, -.83, 1]]) * np.outer(STU_S, STU_S)
UNIV = {"tm": (1.0011, 0.0015), "te": (1.0030, 0.0015), "me": (1.0018, 0.0014)}   # HG-era inputs (older HFLAV); MODERN overrides
UNIV_CORR = None
UNIV_KEYS = ("tm", "te", "me")
HIGGS = {"gg": (1.14, .185), "ZZ": (1.29, .245), "WW": (1.09, .17), "tt": (1.11, .23), "bb": (0.70, .28)}
BR_SM = dict(bb=.5809, WW=.2152, gg=.0818, tt=.0627, cc=.0289, ZZ=.0264, gg_=.00227, Zg=.00154)
# bounds: (limit, normalisation of C.L.)
BOUNDS = {"tau_mu_g": (4.4e-8, 1.645), "tau_e_g": (3.3e-8, 1.645), "mu_e_g": (4.2e-13, 1.645),
          "tau_3mu": (2.1e-8, 1.645), "tau_mumue": (2.7e-8, 1.645), "Cr_Au": (7e-13, 1.645),
          "da_e": (2e-12, 1.645), "d_mu": (1e-19, 1.645), "d_e": (4.1e-30, 1.645),
          "h_tau_mu": (0.0120, 1.949), "h_tau_e": (0.0069, 1.949),
          "m_ee": (190e-3, 1.949), "mu_3e": (1.0e-12, 1.645), "m_beta": (2.05, 1.949), "sum_m": (0.23, 1.949)}
H_TAU_MU_LIMIT = 0.0015   # CMS HIG-20-009 (arXiv:2105.03007), 137 fb^-1: observed 95% CL B(H->mu tau) < 0.15%, stronger than ATLAS 0.18%; h->e tau stays at ATLAS 0.20% (CMS 0.22%)
KAPPA = 1.0
KAPPA3 = 0.25  # set by calibration in tests/test_texture.py (generic trilepton normalisation)
import os as _os
# ZEE_INPUTS=modern2026: the single coherent modern input set (V13 referee): "modern" plus the RPP 2026 free-U oblique fit, the exact half-normal factor z_0.975 = 1.95996 and the DESI-DR2 sum m_nu < 0.0642 eV.
# Each can still be overridden by its variable (ZEE_STU=fit2016, ZEE_Z95=1.949, ZEE_SUMM_LIMIT=0.072) for sensitivity runs; "modern" keeps the frozen inputs of the archived campaign.
if _os.environ.get("ZEE_INPUTS", "hg") == "modern2026":
    for _k, _v in (("ZEE_STU", "rpp2026"), ("ZEE_Z95", "1.95996"), ("ZEE_SUMM_LIMIT", "0.0642")): _os.environ.setdefault(_k, _v)
MODERN = _os.environ.get("ZEE_INPUTS", "hg") in ("modern", "modern2026")
# single ingredients for the controlled reproduction of HG (ZEE_ING=osc,lfv,cr,cosmo,amu,univ); "modern" switches all on
ING = set("osc lfv cr cosmo amu univ".split()) if MODERN else set(filter(None, _os.environ.get("ZEE_ING", "").split(",")))
if "lfv" in ING:
    BOUNDS.update(mu_e_g=(1.5e-13, 1.645), tau_3mu=(1.9e-8, 1.645), tau_mu_g=(4.2e-8, 1.645), h_tau_mu=(H_TAU_MU_LIMIT, 1.949), h_tau_e=(0.0020, 1.949))
if "cosmo" in ING: BOUNDS.update(m_beta=(0.45, 1.949), sum_m=(float(_os.environ.get("ZEE_SUMM_LIMIT", 0.072)), 1.949))  # ZEE_SUMM_LIMIT: sensitivity runs, e.g. 0.0642 (DESI DR2 combination)
if "univ" in ING: UNIV["tm"] = (1.0011, 0.0014)
if "osc" in ING:
    from zee import nufit as _nf
    _NF = {}
if MODERN:  # 2025-26 inputs: MEG II, Belle II, ATLAS h->l tau, HFLAV 2023, KATRIN 2025, DESI DR2
    BOUNDS.update(mu_e_g=(1.5e-13, 1.645), tau_3mu=(1.9e-8, 1.645), tau_mu_g=(4.2e-8, 1.645),
                  h_tau_mu=(H_TAU_MU_LIMIT, 1.949), h_tau_e=(0.0020, 1.949), m_beta=(0.45, 1.949), sum_m=(float(_os.environ.get("ZEE_SUMM_LIMIT", 0.072)), 1.949))  # ZEE_SUMM_LIMIT: sensitivity runs, e.g. 0.0642 (DESI DR2 combination)
    if _os.environ.get("ZEE_Z95"):   # sensitivity: replace the 95% CL half-normal factor 1.949 of the frozen fit (exact value z_0.975 = 1.95996) by the given one
        BOUNDS.update({k: (v[0], float(_os.environ["ZEE_Z95"])) for k, v in BOUNDS.items() if v[1] == 1.949})
    if _os.environ.get("ZEE_STU") == "rpp2026":   # sensitivity: RPP 2026 free-U oblique fit (zee/stu.py) instead of the frozen fit values above
        from zee import stu as _stu_rpp
        STU_C = _stu_rpp.STU_CENTRAL.copy(); STU_S = _stu_rpp.STU_SIGMA.copy()
        _c = np.array([[1, _stu_rpp.RHO["ST"], _stu_rpp.RHO["SU"]], [_stu_rpp.RHO["ST"], 1, _stu_rpp.RHO["TU"]], [_stu_rpp.RHO["SU"], _stu_rpp.RHO["TU"], 1]])
        STU_COV = _c * np.outer(STU_S, STU_S)
    # HFLAV-Tau 2023 (2411.18639; hflav-eos .../end-2023/lepton-univ.html eqs. 6-8, pure leptonic): g_tau/g_mu, g_tau/g_e, g_mu/g_e,
    # with the Table-5 correlations (+0.67, -0.41, +0.40; transcribed from a garbled page, positive-definite, near-singular since
    # g_mu/g_e = (g_tau/g_e)/(g_tau/g_mu)).  Keys: tm = A(tau->e)/A(mu->e) = g_tau/g_mu, te = A(tau->mu)/A(mu->e) = g_tau/g_e,
    # me = A(tau->mu)/A(tau->e) = g_mu/g_e.
    UNIV.update(tm=(1.0016, 0.0014), te=(1.0018, 0.0014), me=(1.0002, 0.0011))
    UNIV_CORR = np.array([[1, .67, -.41], [.67, 1, .40], [-.41, .40, 1]])
    from zee import nufit as _nf
    _NF = {}


def unpack(x):
    f = np.zeros((3, 3), complex); f[0, 2], f[1, 2] = x[0], x[1]; f -= f.T
    Y = np.zeros((3, 3), complex)
    Y[0, 2] = x[2]; Y[2, 2] = x[3] + 1j * x[4]; Y[2, 1] = x[5] + 1j * x[6]
    Y[2, 0] = x[7] + 1j * x[8]; Y[1, 2] = x[9] + 1j * x[10]
    if len(x) > 19:  # relaxed texture: f_emu, Y2^{ee}, Y2^{emu}, Y2^{mue}, Y2^{mumu} (complex)
        f[0, 1] = x[19]; f[1, 0] = -x[19]
        Y[0, 0] = x[20] + 1j * x[21]; Y[0, 1] = x[22] + 1j * x[23]; Y[1, 0] = x[24] + 1j * x[25]; Y[1, 1] = x[26] + 1j * x[27]
    return f, Y


def rescale_fixed_nu(x, tbv):
    """Carry a point to tan(beta)=tbv with the leading neutrino mass matrix unchanged (scalar masses and mixing fixed): M_nu depends on
    f*tan(beta) and Y2/sin(beta) only (Eq. mnu_hg_leading), so f -> f tb0/tb (indices 0,1 = the two singlet couplings) and
    Y2 -> Y2 sin(beta)/sin(beta0) (indices 2..10 = the doublet couplings, real and imaginary parts); ln tan(beta) is index 11."""
    x1 = np.array(x, float); tb0 = np.exp(x1[11]); sb = lambda t: t / np.sqrt(1 + t * t)
    x1[:2] *= tb0 / tbv; x1[2:11] *= sb(tbv) / sb(tb0); x1[11] = np.log(tbv)
    return x1


def scalar_sector(x):
    """Return dict or None if unphysical; plus 'pen' (hinge sum for stability/perturbativity/>100 GeV)."""
    tb = np.exp(x[11]); MHp = 10 ** x[12]; M33 = 10 ** x[13]
    mu = x[14] * np.sqrt(2) * MHp * M33 / V
    lam3, lam5, lam6 = x[15], x[16], x[17]
    lam4 = x[18] if len(x) > 18 else 0.0   # off-slice: lambda4 free (default 0 = HG slice)
    m1sq, m2sq, phi = S.charged_spectrum(MHp**2, M33**2, mu)
    mA2 = MHp**2 + 0.5 * V**2 * (lam4 - lam5)
    M22 = mA2 + lam5 * V**2; M12 = lam6 * V**2
    d = M22 - MH**2
    degenerate = d < 1e-3 * V**2           # 125.5 GeV must be the LIGHTER CP-even eigenvalue (M22 > m_h^2) and not degenerate with M22;
    if degenerate:                         # otherwise the algebraic root below is not the physical state: the point is penalised
        d = 1e-3 * V**2
    M11 = MH**2 + M12**2 / d
    lam1 = M11 / V**2
    mH2 = M11 + M22 - MH**2
    r = -M12 / d
    sba = 1 / np.sqrt(1 + r**2); cba = r * sba
    lam2 = SQ4PI
    pen = 1.0 if degenerate else 0.0
    hinge = lambda v: max(0.0, v)
    masses = np.sqrt(np.array([max(m1sq, 1e-6), max(m2sq, 1e-6), max(mA2, 1e-6), max(mH2, 1e-6)]))
    pen += sum(hinge(100.0 - m) / 100.0 for m in masses)
    pen += hinge(-lam1) + hinge(lam1 - SQ4PI) + hinge(abs(lam3) - SQ4PI) + hinge(abs(lam5) - SQ4PI) + hinge(abs(lam4) - SQ4PI)
    sq = np.sqrt(max(lam1, 0) * lam2)
    pen += hinge(-(lam3 + sq)) + hinge(-(lam3 + min(lam4, 0.0) - abs(lam5) + sq)) + hinge(2 * abs(lam6) - ((lam1 + lam2) / 2 + lam3 + lam5))
    mu2sq = MHp**2 - 0.5 * V**2 * lam3
    pen += hinge(-mu2sq) / V**2
    return dict(tb=tb, mu=mu, m1=masses[0], m2=masses[1], mA=masses[2], mH=masses[3], phi=phi, sba=sba, cba=cba,
                lam1=lam1, lam6=lam6, lam3=lam3, lam4=lam4, lam5=lam5, MHp=MHp, M33=M33, pen=pen)


def yukawa_max(x, sc):
    """Largest Yukawa-type coupling that must stay perturbative (|y| < sqrt(4 pi) ~ 3.54): lepton couplings of the neutral scalars |Dd_ij|,
    the Zee couplings |f_ij| (charged-scalar coupling 2f) and the top coupling tan(beta) sqrt(2) m_t / v of H+."""
    f, Y = unpack(x); Dd = lepton_couplings(Y, sc)[1]
    return float(max(np.abs(Dd).max(), 2*np.abs(f).max(), np.sqrt(2)*sc['tb']*172.5/V))
YUK_MAX = float(np.sqrt(4*np.pi))


def lepton_couplings(Y, sc):
    tb, sba, cba = sc["tb"], sc["sba"], sc["cba"]
    cb = 1 / np.sqrt(1 + tb**2)
    mv = np.diag(M_LEP) / V
    Dd = -np.sqrt(2) * tb * mv + Y.conj().T / cb
    return {"h": sba * mv + cba * Dd / np.sqrt(2), "H": cba * mv - sba * Dd / np.sqrt(2), "A": 1j * Dd / np.sqrt(2)}, Dd


def charged_lepton_widths(f, Dd, sc):
    """Gamma(h_k+ -> l_j nu) summed over neutrinos, k = 1, 2 (rows), j = e, mu, tau (columns).
    -L = nubar U^dag Dd e_R (c h1+ - s h2+) + nu^T C (2f) e_L (s h1+ + c h2+)  (HG 3.26-3.27; hep-ph/0604012):
    the width into l_j is the column-j norm of Dd (doublet part) and of 2f (singlet part)."""
    c, s = np.cos(sc["phi"]), np.sin(sc["phi"]); y = 2 * f
    nD = np.sum(abs(Dd) ** 2, axis=0); nf = np.sum(abs(y) ** 2, axis=0)
    return np.array([sc["m1"] / (16 * np.pi) * (c * c * nD + s * s * nf), sc["m2"] / (16 * np.pi) * (s * s * nD + c * c * nf)])


def univ_resid(uni):
    """whitened universality residuals (independent pulls if no correlation is set, else L^-1 (x - mu) with the HFLAV covariance)"""
    d = np.array([(uni[k] - UNIV[k][0]) / UNIV[k][1] for k in UNIV_KEYS])
    return d if UNIV_CORR is None else np.linalg.solve(np.linalg.cholesky(UNIV_CORR), d)


def universality(f, sc):
    """amplitude ratios A(tau->e)/A(mu->e), A(tau->mu)/A(mu->e), A(tau->mu)/A(tau->e) and the G_F shift from h1+, h2+ exchange"""
    eps = np.sqrt(2) / GF * (np.sin(sc["phi"]) ** 2 / sc["m1"] ** 2 + np.cos(sc["phi"]) ** 2 / sc["m2"] ** 2)
    gme, gte, gtm = (np.sqrt(1 + eps * abs(f[a, b]) ** 2) for a, b in ((0, 1), (0, 2), (1, 2)))
    return {"tm": gte / gme, "te": gtm / gme, "me": gtm / gte}, gme - 1


def dipoles(i, j, Lam, mass, bz_c):
    """LEGACY (HG eq. 3.13-3.19 leading terms): kept only as a regression approximation; production uses zee.dipfull."""
    """(C_L, C_R) for decay j -> i, HG normalisation (|C|^2 form of eq. 3.12). One-loop neutral (tau inside) + BZ."""
    mj = M_LEP[j]
    cl = cr = 0j
    for p, L in Lam.items():
        m2 = mass[p] ** 2
        lg = np.log(m2 / MTAU**2) - 1.5
        Ld = L.conj().T
        cr += (MTAU / mj) * L[i, 2] * L[2, j] / m2 * lg
        cl += (MTAU / mj) * Ld[i, 2] * Ld[2, j] / m2 * lg
        if j == 2:
            cl += (1.0 / 6.0) * Ld[i, 2] * L[2, j] / m2
    cl, cr = cl / (16 * np.pi**2), cr / (16 * np.pi**2)
    cr += 2 * bz_c[i, j] / mj
    cl += 2 * np.conj(bz_c[j, i]) / mj
    return cl, cr


def charged_c(i, j, f, Dd, sc):
    s2, c2 = np.sin(sc["phi"]) ** 2, np.cos(sc["phi"]) ** 2
    m1, m2 = sc["m1"], sc["m2"]
    Kf = s2 / m1**2 + c2 / m2**2; KH = c2 / m1**2 + s2 / m2**2
    y = 2 * f
    Sf = np.sum(np.conj(y[:, i]) * y[:, j])
    D = Dd.conj().T
    SH = (D @ D.conj().T)[i, j]
    k = -1 / (192 * np.pi**2)
    return k * KH * SH, k * Kf * Sf


def bz_matrix(Lam, mass, sc):
    tb, sba, cba = sc["tb"], sc["sba"], sc["cba"]
    kh = sba - tb * cba; kH = cba + tb * sba
    out = {}
    def top(mf, sgn, nq2):
        return (mf, nq2, {"h": (mf / V * kh, 0.0), "H": (mf / V * kH, 0.0), "A": (0.0, sgn * tb * mf / V)})
    tau = (MTAU, 1.0, {p: (Lam[p][2, 2].real, Lam[p][2, 2].imag) for p in Lam})
    return B.cR_bz(Lam, mass, [top(172.5, +1, 3 * 4 / 9), top(4.18, -1, 3 / 9), tau])


def htogg(sc, mt=172.5, mw=80.379):
    """R_gamma_gamma, HG eq. (3.42)-(3.46) (lam7=lam8=lam10=0)."""
    mh2 = MH**2
    def fx(x):
        return np.arcsin(np.sqrt(x)) ** 2 if x <= 1 else -0.25 * (np.log((1 + np.sqrt(1 - 1 / x + 0j)) / (1 - np.sqrt(1 - 1 / x + 0j))) - 1j * np.pi) ** 2
    # loop functions in terms of tau = 4 m^2 / m_h^2, x = tau
    def A12(t): return 2 * t + 2 * t * (1 - t) * fx(1 / t)
    def A1(t): return -2 - 3 * t - 3 * t * (2 - t) * fx(1 / t)
    def A00(t): return -t + t**2 * fx(1 / t)
    sba, cba, tb = sc["sba"], sc["cba"], sc["tb"]
    kt = sba - tb * cba
    s2, c2 = np.sin(sc["phi"]) ** 2, np.cos(sc["phi"]) ** 2
    s2phi = 2 * np.sin(sc["phi"]) * np.cos(sc["phi"])
    lam = (sba * sc["lam3"] * c2 - sba * sc["mu"] * s2phi / (np.sqrt(2) * V),
           sba * sc["lam3"] * s2 + sba * sc["mu"] * s2phi / (np.sqrt(2) * V))
    tW, tt = 4 * mw**2 / mh2, 4 * mt**2 / mh2
    num = sba * A1(tW) + (4 / 3) * kt * A12(tt)
    for l, m in zip(lam, (sc["m1"], sc["m2"])):
        num += l * V**2 / (2 * m**2) * A00(4 * m**2 / mh2)
    den = A1(tW) + (4 / 3) * A12(tt)
    return float(abs(num / den) ** 2)


def rgg_general(sc, trilin, mt=172.5, mw=80.379):
    """R_gamma_gamma with the charged-scalar trilinears of dipfull.charged_trilinears (HG eqs. 3.42-3.46 for l7=l8=l10=0)."""
    mh2 = MH**2
    def fx(x):
        return np.arcsin(np.sqrt(x)) ** 2 if x <= 1 else -0.25 * (np.log((1 + np.sqrt(1 - 1 / x + 0j)) / (1 - np.sqrt(1 - 1 / x + 0j))) - 1j * np.pi) ** 2
    def A12(t): return 2 * t + 2 * t * (1 - t) * fx(1 / t)
    def A1(t): return -2 - 3 * t - 3 * t * (2 - t) * fx(1 / t)
    def A00(t): return -t + t**2 * fx(1 / t)
    kt = sc["sba"] - sc["tb"] * sc["cba"]
    tW, tt = 4 * mw**2 / mh2, 4 * mt**2 / mh2
    num = sc["sba"] * A1(tW) + (4 / 3) * kt * A12(tt)
    for l, m in zip(trilin["h"], (sc["m1"], sc["m2"])):
        num += l * V**2 / (2 * m**2) * A00(4 * m**2 / mh2)
    return float(abs(num / (A1(tW) + (4 / 3) * A12(tt))) ** 2)


def evaluate(x, ordering="NO", with_stu=True, eta=None):
    """eta = (l2, l7, l8, l9, l10, lh): free quartics entering R_gamma_gamma and the charged-scalar Barr-Zee loops
    (None = HG slice, l7 = l8 = l10 = 0). The same chi2 is used for search, acceptance and reporting."""
    sc = scalar_sector(x)
    f, Y = unpack(x)
    tb = sc["tb"]
    Lam, Dd = lepton_couplings(Y, sc)
    mass = {"h": MH, "H": sc["mH"], "A": sc["mA"]}
    M = N.mnu_hg_leading(f, M_LEP, Y, tb, sc["m1"], sc["m2"], sc["mu"])
    o = N.oscillation_observables(M, ordering)
    t = OSC[ordering]
    dd = (o["delta_deg"] - t["d"][0] + 180.) % 360. - 180.
    if "osc" in ING:
        if ordering not in _NF: _NF[ordering] = _nf.NuFit(ordering)
        c_ = _NF[ordering].chi2(o["s12sq"], o["s13sq"], o["s23sq"], o["dm21_eV2"], o["dm3l_eV2"], o["delta_deg"] % 360.)
        osc = np.array([np.sqrt(max(c_, 0.0))])
    else:
      osc = np.array([(o["s12sq"] - t["s12"][0]) / t["s12"][1], (o["s13sq"] - t["s13"][0]) / t["s13"][1],
                    (o["s23sq"] - t["s23"][0]) / t["s23"][1], (o["dm21_eV2"] - t["dm21"][0]) / t["dm21"][1],
                    (o["dm3l_eV2"] - t["dm3"][0]) / t["dm3"][1], dd / t["d"][1]])
    ob = {}
    l7, l8, l10 = (0.0, 0.0, 0.0) if eta is None else (eta[1], eta[2], eta[4])
    trilin = DF.charged_trilinears(sc, l7, l8, l10)
    AL, AR = DF.amplitudes(Lam, mass, f, Dd, sc, trilin)
    for name, (i, j) in {"mu_e_g": (0, 1), "tau_e_g": (0, 2), "tau_mu_g": (1, 2)}.items():
        norm = 1.0 if name == "mu_e_g" else BR_TAU_LNUNU[i]
        ob[name] = norm * 48 * np.pi**3 * ALPHA / GF**2 * (abs(AL[i, j]) ** 2 + abs(AR[i, j]) ** 2)
    ALs, ARs = DF.sm_reference(sc)
    amom, dmom = DF.diagonal_moments(AL - ALs, AR - ARs)   # new physics only
    for nm, i in (("e", 0), ("mu", 1)):
        ob["da_" + nm] = abs(amom[i]); ob["das_" + nm] = amom[i]; ob["d_" + nm] = abs(dmom[i])
    # trilepton (3.9)
    cb = 1 / np.sqrt(1 + tb**2)
    g1 = {"h": sc["sba"] - tb * sc["cba"], "H": sc["cba"] + tb * sc["sba"], "A": -1j * tb}
    g2 = {"h": sc["cba"] / (np.sqrt(2) * cb), "H": -sc["sba"] / (np.sqrt(2) * cb), "A": 1j / (np.sqrt(2) * cb)}
    # HG (2.19): g1 = (-s_a/c_b, c_a/c_b, -i t_b), with s_a/c_b = s_ba... use kappa forms above
    for name, (Yd, Yr, xi) in {"tau_3mu": (Y[1, 2], Y[2, 1], 0.5), "tau_mumue": (Y[0, 2], Y[2, 0], 1.0)}.items():
        DLL = DRL = DLR = DRR = 0
        for p in ("h", "H", "A"):
            m2 = mass[p] ** 2
            DLL += -0.5 / m2 * np.conj(g2[p]) * Yd * np.conj(g1[p]) * MMU / V
            DRL += -0.5 / m2 * np.conj(g2[p]) * Yd * g1[p] * MMU / V
            DRR += -0.5 / m2 * g2[p] * np.conj(Yr) * g1[p] * MMU / V
            DLR += -0.5 / m2 * g2[p] * np.conj(Yr) * np.conj(g1[p]) * MMU / V
        ob[name] = BR_TAU_LNUNU[1] * (1 / (32 * GF**2)) * (xi * abs(DLL) ** 2 + xi * abs(DRR) ** 2 + abs(DLR) ** 2 + abs(DRL) ** 2)
    # generic tree-level l_I -> 3 l_j through h, H, A with the full coupling matrices (normalisation calibrated so that
    # it equals the HG (3.9) expression above on the HG slice; tests/test_texture.py); used for the relaxed texture
    def l3(I, j, xi):
        C = np.zeros((2, 2), complex)
        for p in ("h", "H", "A"):
            A_ = Lam[p]; m2 = mass[p] ** 2
            vI = (A_[j, I], np.conj(A_[I, j])); vj = (A_[j, j], np.conj(A_[j, j]))
            for a in range(2):
                for b in range(2): C[a, b] += vI[a] * vj[b] / m2
        return (1 / (32 * GF**2)) * (xi * abs(C[0, 0]) ** 2 + xi * abs(C[1, 1]) ** 2 + abs(C[0, 1]) ** 2 + abs(C[1, 0]) ** 2) * KAPPA3
    ob["tau_3mu_hg"] = ob["tau_3mu"]; ob["tau_3mu_gen"] = BR_TAU_LNUNU[1] * l3(2, 1, 0.5); ob["tau_3mu"] = ob["tau_3mu_gen"]; ob["mu_3e_tree"] = l3(1, 0, 0.5)
    # mu-e conversion (3.24)-(3.25)
    gLV = gRV = 0
    for p in ("h", "H", "A"):
        m2 = mass[p] ** 2
        w = abs(g2[p]) ** 2 * (ALPHA / (2 * np.pi)) * (-1.0 / (9 * m2)) * (4 + 3 * np.log(MTAU**2 / m2))
        for k in range(3):   # internal lepton k; only k=tau on the HG slice. Log cut at max(m_k, m_mu) for light k
            wk = w if k == 2 else abs(g2[p]) ** 2 * (ALPHA / (2 * np.pi)) * (-1.0 / (9 * m2)) * (4 + 3 * np.log(max(M_LEP[k], MMU) ** 2 / m2))
            gLV += wk * Y[k, 1] * np.conj(Y[k, 0]); gRV += wk * np.conj(Y[1, k]) * Y[0, k]
    # Monopole (photon-penguin) term, HG eq. (3.24)-(3.25); loop function checked numerically (gate1/cr_loop_check.py)
    cr_au_hg = (0.0974 * MMU**2.5) ** 2 * (abs(gLV) ** 2 + abs(gRV) ** 2) / 8.6e-18
    ob["Cr_Au_HG"] = cr_au_hg
    # Matching with Kitano-Koike-Okada: contact C (e gamma P mu)(p gamma p) gives omega = 4 |C|^2 V_p^2 with C = 4 pi alpha A,
    # which reproduces HG (3.24)-(3.25) exactly (an earlier version used omega = |C|^2 V^2/4, a factor 16 too small).
    ob["Cr_Au"] = cr_au_hg + 3.9e-3 * ob["mu_e_g"] if "cr" in ING else cr_au_hg   # "cr": add the dipole term
    # Al: V_p=0.0161, Gamma_capt=0.7054e6/s -> Cr_Al = 0.51 Cr_Au(monopole); dipole ratio 2.67e-3
    ob["Cr_Al"] = 0.51 * cr_au_hg + 2.67e-3 * ob["mu_e_g"]
    # tree-level scalar exchange (relaxed texture only: needs Y2^{e mu}, Y2^{mu e}); coherent N-bar N coupling kappa_p m_N f_N / v,
    # f_N = 0.30, quark couplings type-I-like (HG quark sector), KKO overlaps (Al: S_p+S_n = 0.0322; Au: 0.1532), added incoherently
    gS = np.zeros(2, complex)
    for p, kq in (("h", sc["sba"] - tb * sc["cba"]), ("H", sc["cba"] + tb * sc["sba"])):
        gS += np.array([Lam[p][0, 1], np.conj(Lam[p][1, 0])]) * kq * 0.939 * 0.30 / (V * mass[p] ** 2) * np.sqrt(2) / GF
    wS = 2 * GF**2 * MMU**5 * np.sum(abs(gS) ** 2)
    ob["Cr_Al_S"] = wS * 0.0322**2 / 4.643e-19; ob["Cr_Au_S"] = wS * 0.1532**2 / 8.6e-18
    if "cr" in ING: ob["Cr_Au"] += ob["Cr_Au_S"]
    ob["Cr_Al"] += ob["Cr_Al_S"]
    ob["h_tau_mu"] = O.higgs_lfv_br(1, 2, Y, tb, sc["cba"]); ob["h_tau_e"] = O.higgs_lfv_br(0, 2, Y, tb, sc["cba"])
    ob["m_ee"] = abs(M[0, 0]) * 1e9
    # mu->3e: dipole-dominated estimate (photon penguin), Br ~ alpha/(3 pi) (ln(m_mu^2/m_e^2) - 11/4) Br(mu->e gamma); no tree-level Y_ee, Y_mu e on this slice
    ob["mu_3e"] = ALPHA / (3 * np.pi) * (np.log(MMU**2 / ME_**2) - 11 / 4) * ob["mu_e_g"] + ob["mu_3e_tree"]   # tree part vanishes on the HG slice
    ob["m_beta"] = float(np.sqrt(np.sum(abs(o["U"][0]) ** 2 * (o["masses_eV"]) ** 2)))
    ob["sum_m"] = float(np.sum(o["masses_eV"]))
    # chi2 pieces
    chi_b = {}
    for k, (lim, nrm) in BOUNDS.items():
        B_ = lim / nrm
        # continuous one-sided Gaussian surrogate centred at zero with sigma = limit/z (z = 1.645 for 90% CL, 1.949 for 95% CL)
        chi_b[k] = (ob[k] / B_) ** 2
    nat1 = 0.5 * V**2 * abs(Y[2, 1] * Y[1, 2]) * (tb / np.sqrt(1 + tb**2)) ** 2
    nat2 = 0.5 * V**2 * abs(Y[2, 0] * Y[0, 2]) * (tb / np.sqrt(1 + tb**2)) ** 2
    chi_b["nat_mu"] = max(0.0, (nat1 / (MMU * MTAU)) ** 2 - 1)   # continuous naturalness hinges (HG)
    chi_b["nat_e"] = max(0.0, (nat2 / (ME_ * MTAU)) ** 2 - 1)
    muB = KAPPA * 4 * np.pi * MH / sc["sba"]
    chi_b["nat_mu_trilinear"] = max(0.0, (sc["mu"] / muB) ** 2 - 1)
    # tree-level h1+, h2+ exchange in l_I -> l_J nu nu (HG eqs. 3.28-3.33; Crivellin et al. 2012.09845 eq. 2):
    # |A/A_SM|^2 = 1 + eps |f_IJ|^2, eps = sqrt2/(G_F Mt2^2), 1/Mt2^2 = s^2/m1^2 + c^2/m2^2
    uni, ob["delta_mu_e"] = universality(f, sc)   # delta_mu_e: G_F shift from muon decay (CKM unitarity), off-slice only
    chi_am = ((ob["das_mu"] - 3.8e-10) / 6.3e-10) ** 2 if "amu" in ING else 0.0   # a_mu^exp - a_mu^SM(WP25)
    chi_b["amu"] = chi_am
    chi_u = float(np.sum(univ_resid(uni) ** 2))
    chi_stu = 0.0; stu_v = (np.nan,) * 3
    if with_stu:
        stu_v = T.stu(sc["m1"], sc["m2"], sc["phi"], sc["mH"], sc["mA"], sc["cba"], mh=MH, hg_factor=STU_HG_LEGACY)
        d = np.array(stu_v) - STU_C
        chi_stu = float(d @ np.linalg.solve(STU_COV, d))
    # Higgs signals (approximate)
    kV = sc["sba"]; kt = sc["sba"] - tb * sc["cba"]; ktau = Lam["h"][2, 2].real * V / MTAU
    rgg = rgg_general(sc, trilin)
    gam = (BR_SM["bb"] * kt**2 + BR_SM["WW"] * kV**2 + BR_SM["gg"] * kt**2 + BR_SM["tt"] * ktau**2 + BR_SM["cc"] * kt**2
           + BR_SM["ZZ"] * kV**2 + BR_SM["gg_"] * rgg + BR_SM["Zg"] * 1.0)
    prod = 0.87 * kt**2 + 0.13 * kV**2
    mus = {"gg": prod * rgg / gam, "ZZ": prod * kV**2 / gam, "WW": prod * kV**2 / gam, "tt": prod * ktau**2 / gam,
           "bb": prod * kt**2 / gam}
    chi_h = sum(((mus[k] - HIGGS[k][0]) / HIGGS[k][1]) ** 2 for k in mus)
    chi_osc = float(np.sum(osc**2))
    # smooth residual vector for least squares (surrogate for the hinge-type bound terms)
    res = list(osc)
    for k, (lim, nrm) in BOUNDS.items():
        res.append(ob[k] / (lim / nrm))
    res.append(np.sqrt(chi_b["nat_mu"])); res.append(np.sqrt(chi_b["nat_e"]))
    res.append(np.sqrt(chi_b["nat_mu_trilinear"]))
    res.append((ob["das_mu"] - 3.8e-10) / 6.3e-10 if "amu" in ING else 0.0)
    res.extend(univ_resid(uni))
    if with_stu:
        res.extend(np.linalg.solve(np.linalg.cholesky(STU_COV), np.array(stu_v) - STU_C))
    res.extend([(mus[k] - HIGGS[k][0]) / HIGGS[k][1] for k in mus])
    res.append(np.sqrt(1e3 * sc["pen"]))   # same weight as in chi2
    total = chi_osc + sum(chi_b.values()) + chi_u + chi_stu + chi_h + 1e3 * sc["pen"]
    return dict(chi2=total, chi_osc=chi_osc, chi_bounds=chi_b, chi_univ=chi_u, chi_stu=chi_stu, chi_higgs=chi_h,
                pen=sc["pen"], res=np.array(res, float), obs=ob, sc=sc, osc=osc, stu=stu_v, mus=mus, rgg=rgg)
