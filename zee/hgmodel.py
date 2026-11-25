"""HG (JHEP04(2017)130) slice of the general Zee model and its full chi^2 (HG sec. 4, 2016 inputs).

Slice: f_emu=0, Y2^{ee,emu,mue,mumu}=0, f and Y2^{e tau} real, lam4=lam7=lam8=lam9=lam10=lam_h=0,
lam2 = sqrt(4 pi) (irrelevant except stability), m_h = 125.5 GeV fixed.
Parameter vector x (18): f_et, f_mt, Y2_et, Y2_tt(re,im), Y2_tm(re,im), Y2_te(re,im), Y2_mt(re,im),
  ln tb, log10 M_H+, log10 M_33, u (mu = u sqrt2 M_H+ M_33 / v), lam3, lam5, lam6.
lam1 follows from m_h = 125.5 (eq. 2.13); lam4 = 0 gives m_A^2 = M_H+^2 - v^2 lam5 / 2.
APPROXIMATE inputs (documented in the code): Higgs signal strengths (ATLAS+CMS Run-1 combination
from memory, not HG's refs [105-108]); HFLAV universality treated as uncorrelated.
"""
import numpy as np

from . import V_EW as V
from . import neutrino as N, scalars as S, observables as O, stu as T, bz as B
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
STU_C = np.array([0.05, 0.09, 0.01]); STU_S = np.array([.11, .13, .11])
STU_COV = np.array([[1, .90, -.59], [.90, 1, -.83], [-.59, -.83, 1]]) * np.outer(STU_S, STU_S)
UNIV = {"tm": (1.0011, 0.0015), "te": (1.0030, 0.0015), "me": (1.0018, 0.0014)}
HIGGS = {"gg": (1.14, .185), "ZZ": (1.29, .245), "WW": (1.09, .17), "tt": (1.11, .23), "bb": (0.70, .28)}
BR_SM = dict(bb=.5809, WW=.2152, gg=.0818, tt=.0627, cc=.0289, ZZ=.0264, gg_=.00227, Zg=.00154)
# bounds: (limit, normalisation of C.L.)
BOUNDS = {"tau_mu_g": (4.4e-8, 1.645), "tau_e_g": (3.3e-8, 1.645), "mu_e_g": (4.2e-13, 1.645),
          "tau_3mu": (2.1e-8, 1.645), "tau_mumue": (2.7e-8, 1.645), "Cr_Au": (7e-13, 1.645),
          "da_e": (2e-12, 1.645), "d_mu": (1e-19, 1.645), "d_e": (4.1e-30, 1.645),
          "h_tau_mu": (0.0120, 1.949), "h_tau_e": (0.0069, 1.949),
          "m_ee": (190e-3, 1.949), "mu_3e": (1.0e-12, 1.645), "m_beta": (2.05, 1.949), "sum_m": (0.23, 1.949)}
KAPPA = 1.0
KAPPA3 = 0.25  # set by calibration in tests/test_texture.py (generic trilepton normalisation)
import os as _os
MODERN = _os.environ.get("ZEE_INPUTS", "hg") == "modern"
# single ingredients for the controlled reproduction of HG (ZEE_ING=osc,lfv,cr,cosmo,amu,univ); "modern" switches all on
ING = set("osc lfv cr cosmo amu univ".split()) if MODERN else set(filter(None, _os.environ.get("ZEE_ING", "").split(",")))
if "lfv" in ING:
    BOUNDS.update(mu_e_g=(1.5e-13, 1.645), tau_3mu=(1.9e-8, 1.645), tau_mu_g=(4.2e-8, 1.645), h_tau_mu=(0.0018, 1.949), h_tau_e=(0.0020, 1.949))
if "cosmo" in ING: BOUNDS.update(m_beta=(0.45, 1.949), sum_m=(0.072, 1.949))
if "univ" in ING: UNIV["tm"] = (1.0011, 0.0014)
if "osc" in ING:
    from zee import nufit as _nf
    _NF = {}
if MODERN:  # 2025-26 inputs: MEG II, Belle II, ATLAS h->l tau, HFLAV 2023, KATRIN 2025, DESI DR2
    BOUNDS.update(mu_e_g=(1.5e-13, 1.645), tau_3mu=(1.9e-8, 1.645), tau_mu_g=(4.2e-8, 1.645),
                  h_tau_mu=(0.0018, 1.949), h_tau_e=(0.0020, 1.949), m_beta=(0.45, 1.949), sum_m=(0.072, 1.949))
    UNIV["tm"] = (1.0011, 0.0014)
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
    if abs(d) < 1e-3 * V**2:
        d = 1e-3 * V**2 * (1 if d >= 0 else -1)
    M11 = MH**2 + M12**2 / d
    lam1 = M11 / V**2
    mH2 = M11 + M22 - MH**2
    r = -M12 / d
    sba = 1 / np.sqrt(1 + r**2); cba = r * sba
    lam2 = SQ4PI
    pen = 0.0
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


def lepton_couplings(Y, sc):
    tb, sba, cba = sc["tb"], sc["sba"], sc["cba"]
    cb = 1 / np.sqrt(1 + tb**2)
    mv = np.diag(M_LEP) / V
    Dd = -np.sqrt(2) * tb * mv + Y.conj().T / cb
    return {"h": sba * mv + cba * Dd / np.sqrt(2), "H": cba * mv - sba * Dd / np.sqrt(2), "A": 1j * Dd / np.sqrt(2)}, Dd


def dipoles(i, j, Lam, mass, bz_c):
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


def evaluate(x, ordering="NO", with_stu=True):
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
    bzc = bz_matrix(Lam, mass, sc)
    for name, (i, j) in {"mu_e_g": (0, 1), "tau_e_g": (0, 2), "tau_mu_g": (1, 2)}.items():
        cl, cr = dipoles(i, j, Lam, mass, bzc)
        clc, crc = charged_c(i, j, f, Dd, sc)
        cl += clc; cr += crc
        norm = 1.0 if name == "mu_e_g" else BR_TAU_LNUNU[i]
        ob[name] = norm * 48 * np.pi**3 * ALPHA / GF**2 * (abs(cl) ** 2 + abs(cr) ** 2)
    # diagonal moments (C_L = C_R^*): a = 2 m^2 Re C_R, d/e = m Im C_R, from c = e m C/2 and L = (e a/4m) lbar sigma l F;
    # HG (3.21), (3.23) are a factor 2 larger (convention of the h.c. for i=j); physical normalisation used here
    for nm, i in (("e", 0), ("mu", 1)):
        cl, cr = dipoles(i, i, Lam, mass, bzc)
        cl = np.conj(cr)
        clc, crc = charged_c(i, i, f, Dd, sc)
        crt = cr + crc
        ob["da_" + nm] = abs(2 * M_LEP[i] ** 2 * crt.real); ob["das_" + nm] = 2 * M_LEP[i] ** 2 * crt.real
        ob["d_" + nm] = abs(M_LEP[i] * crt.imag) * 1.973e-14  # GeV^-1 -> cm
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
    ob["tau_3mu_gen"] = BR_TAU_LNUNU[1] * l3(2, 1, 0.5); ob["tau_3mu"] = ob["tau_3mu_gen"]; ob["mu_3e_tree"] = l3(1, 0, 0.5)
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
    eps = np.sqrt(2) / (GF * (np.sin(sc["phi"]) ** 2 / sc["m1"] ** 2 + np.cos(sc["phi"]) ** 2 / sc["m2"] ** 2))
    uni = {"tm": np.sqrt(1 + eps * abs(f[0, 2]) ** 2), "te": np.sqrt(1 + eps * abs(f[1, 2]) ** 2),
           "me": np.sqrt(abs(1 + eps * (abs(f[1, 2]) ** 2 - abs(f[0, 2]) ** 2)))}
    chi_am = ((ob["das_mu"] - 3.8e-10) / 6.3e-10) ** 2 if "amu" in ING else 0.0   # a_mu^exp - a_mu^SM(WP25)
    chi_b["amu"] = chi_am
    chi_u = sum(((uni[k] - UNIV[k][0]) / UNIV[k][1]) ** 2 for k in uni)
    chi_stu = 0.0; stu_v = (np.nan,) * 3
    if with_stu:
        stu_v = T.stu(sc["m1"], sc["m2"], sc["phi"], sc["mH"], sc["mA"], sc["cba"], mh=MH, hg_factor=True)
        d = np.array(stu_v) - STU_C
        chi_stu = float(d @ np.linalg.solve(STU_COV, d))
    # Higgs signals (approximate)
    kV = sc["sba"]; kt = sc["sba"] - tb * sc["cba"]; ktau = Lam["h"][2, 2].real * V / MTAU
    rgg = htogg(sc)
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
    res.extend([(uni[k] - UNIV[k][0]) / UNIV[k][1] for k in uni])
    if with_stu:
        res.extend(np.linalg.solve(np.linalg.cholesky(STU_COV), np.array(stu_v) - STU_C))
    res.extend([(mus[k] - HIGGS[k][0]) / HIGGS[k][1] for k in mus])
    res.append(30.0 * np.sqrt(sc["pen"]))
    total = chi_osc + sum(chi_b.values()) + chi_u + chi_stu + chi_h + 1e3 * sc["pen"]
    return dict(chi2=total, chi_osc=chi_osc, chi_bounds=chi_b, chi_univ=chi_u, chi_stu=chi_stu, chi_higgs=chi_h,
                pen=sc["pen"], res=np.array(res, float), obs=ob, sc=sc, osc=osc, stu=stu_v, mus=mus, rgg=rgg)
