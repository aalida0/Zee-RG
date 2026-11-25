"""Complete l_I -> l_J gamma dipole amplitudes for the Zee model (production kernel).

Convention (Omura-Senaha-Tobe 1511.08880, same as HG eq. 3.12):
  -L = phi ebar_L Lam_phi e_R + h.c.,  T = e eps* ubar_J m_I i sigma q (A_L P_L + A_R P_R) u_I,
  Br(I->J gamma)/Br(I->J nu nu) = 48 pi^3 alpha (|A_L|^2 + |A_R|^2)/G_F^2.
Matrices are indexed [J, I] (final, initial).
Contents
  n1 : one-loop neutral h, H, A; all internal leptons k; internal flip (m_k/m_I) log term and both external
       flips (m_I and m_J), Omura Eqs. 18, 30-31 completed with the m_J/m_I mirror terms of Lavoura hep-ph/0302221.
  ch : one-loop charged h1+, h2+ (doublet part Dd and singlet part 2f), both external flips.
  bzf: Barr-Zee fermion loops t, b, tau through phi-gamma-gamma and phi-Z-gamma (Omura Eq. 20).
  bzW: Barr-Zee W loops for h, H (Omura Eq. 20).
  bzC: Barr-Zee charged-scalar loops h1+, h2+ for h, H (Ilisie 1502.04199 Eq. 20 mapped to LFV).
Checks (tests/test_dipfull.py): the diagonal limit gives the textbook one-loop a_mu for scalar
(ln - 7/6), pseudoscalar (-(ln - 11/6)) and charged scalar (-1/96 pi^2); Barr-Zee signs: CP-even top loop
negative, CP-odd positive, W loop opposite to top.
Diagonal moments: a_I = m_I^2 Re(A_L + A_R), d_I/e = m_I Im(A_R - A_L)/2.
  bzHW: H+- W-+ gamma Barr-Zee sets (Ilisie 1502.04199 Eqs. 22-24) for h1+, h2+ with the H+ content of each state.  It is
       LARGE (it raises tau -> mu gamma by a factor ~2 at the earlier benchmarks) and rests on the transcription of the
       published formulas plus the mapping Dd -> varsigma_l; the gate therefore requires all limits to hold for HW_SCALE = 0, +1, -1.
Not included: two-loop terms beyond these.
"""
import numpy as np
from . import V_EW as V
from . import loops as LF
from .observables import M_LEP, ALPHA

MW, MZ, SW2 = 80.379, 91.1876, 0.231
CW2 = 1 - SW2; TW2 = SW2 / CW2
PI2 = np.pi**2
M = np.asarray(M_LEP, float)
MJI = M[:, None] / M[None, :]          # m_J/m_I
MKI = M[:, None] / M[None, :]          # m_k/m_I as [k, I]


def charged_trilinears(sc, l7=0.0, l8=0.0, l10=0.0):
    """Couplings lambda_{phi h_k+ h_k-} (in units of v, -L contains lambda v phi h+ h-), code convention
    (off-diagonal -mu v/sqrt2, light state c H+ + s h+), the same as hgmodel.rgg."""
    c, s = np.cos(sc["phi"]), np.sin(sc["phi"])
    out = {}
    for p, (a, b) in (("h", (sc["sba"], sc["cba"])), ("H", (sc["cba"], -sc["sba"]))):
        dHH = a * sc["lam3"] + b * l7
        dhh = a * l8 + b * l10
        dHh = -a * sc["mu"] / (np.sqrt(2) * V)
        out[p] = (c * c * dHH + s * s * dhh + 2 * s * c * dHh, s * s * dHH + c * c * dhh - 2 * s * c * dHh)
        out["raw_" + p] = (dHH, dhh, dHh)      # couplings in the (H+, h+) basis, used by the H+ W gamma loops
    return out


def quark_couplings(sc):
    """(S, P) of -L = phi fbar (S + i P gamma5) f for t, b (HG type-I-like quark sector)."""
    tb, sba, cba = sc["tb"], sc["sba"], sc["cba"]
    kh, kH = sba - tb * cba, cba + tb * sba
    return {"t": (172.5, 3, 2 / 3, 0.5, {"h": (172.5 / V * kh, 0.0), "H": (172.5 / V * kH, 0.0), "A": (0.0, tb * 172.5 / V)}),
            "b": (4.18, 3, -1 / 3, -0.5, {"h": (4.18 / V * kh, 0.0), "H": (4.18 / V * kH, 0.0), "A": (0.0, -tb * 4.18 / V)})}


def bz_kernels(Lam, mass, sc, trilin):
    """K_phi (GeV^-1) such that A_R^BZ[J,I] = sum_phi Lam_phi[J,I] K_phi / m_I, A_L^BZ = conj(...)^T."""
    ferm = quark_couplings(sc)
    ferm["tau"] = (M[2], 1, -1.0, -0.5, {p: (Lam[p][2, 2].real, Lam[p][2, 2].imag) for p in Lam})
    gW = {"h": sc["sba"], "H": sc["cba"]}
    K = {}
    for p in Lam:
        mp = mass[p]; kf = kw = kc = 0j
        for mf, nc, q, t3, cp in ferm.values():
            S, P = cp[p]; x, xz = mf**2 / mp**2, mf**2 / MZ**2
            zc = (1 - 4 * SW2) * (2 * t3 - 4 * q * SW2) / (16 * SW2 * CW2)
            kf += -ALPHA * nc * q / (8 * np.pi**3 * mf) * (q * (S * LF.FH(x) + 1j * P * LF.FA(x))
                                                           + zc * (S * LF.FHt(x, xz) + 1j * P * LF.FAt(x, xz)))
        if p in gW:
            x, xz = MW**2 / mp**2, MW**2 / MZ**2
            FH, FA, G = LF.FH(x), LF.FA(x), LF.G(x)
            br = 3 * FH + 23 / 4 * FA + 3 / 4 * G + mp**2 / (2 * MW**2) * (FH - FA)
            br += (1 - 4 * SW2) / (8 * SW2) * ((5 - TW2 + (1 - TW2) / (2 * x)) * LF.FHt(x, xz)
                                              + (7 - 3 * TW2 - (1 - TW2) / (2 * x)) * LF.FAt(x, xz) + 1.5 * (FA + G))
            kw = ALPHA / (16 * np.pi**3 * V) * gW[p] * br
            for lk, mk in zip(trilin[p], (sc["m1"], sc["m2"])):
                kc += ALPHA * V / (16 * np.pi**3 * mp**2) * lk * LF.F2(mk**2 / mp**2)
        K[p] = dict(f=kf, W=kw, C=kc)
    return K


def _panels():
    e = np.concatenate([[0.0], 10.0 ** np.arange(-9, -0.5, 1.0), [0.5]])
    e = np.unique(np.concatenate([e, 1 - e]))                 # geometric panels towards both endpoints
    g, w = np.polynomial.legendre.leggauss(16)
    xs = np.concatenate([0.5 * (b - a) * g + 0.5 * (a + b) for a, b in zip(e[:-1], e[1:])])
    ws = np.concatenate([0.5 * (b - a) * w for a, b in zip(e[:-1], e[1:])])
    return xs, ws
_XG, _WG = _panels()
HW_ON = True
HW_SCALE = 1.0   # robustness variants of the H+-W-+ gamma set: 0 (off), 1 (default), -1 (opposite overall sign)
NC, QT, QB = 3, 2.0 / 3, -1.0 / 3
MT, MB = 172.5, 4.18


def _G(wa, wb, x):
    """Loop function of Ilisie 1502.04199 Eq. (29), read as ln(X/y)/(y - X) with X = wa x + wb (1-x), y = x(1-x): it is the
    k^2 integral of the two propagators of the Barr-Zee construction; finite at X = y."""
    X = wa * x + wb * (1 - x); y = x * (1 - x); t = X / y
    d = np.where(np.abs(t - 1) < 1e-7, -1.0, np.log(t) / (1 - t + (np.abs(t - 1) < 1e-7)))
    return d / y


def bz_HW(Dd, mass, sc, trilin):
    """H+- W-+ gamma Barr-Zee amplitudes (Ilisie 1502.04199 Eqs. 22-24, S^(4), S^(5), S^(6)), A_L[J,I], A_R[J,I].
    The charged mass eigenstates h_k+ contain H+ with amplitude u = (cos phi, -sin phi) and the diagrams are summed over k
    (outer) and, for the contact-type loop, over the internal mass eigenstate j. The lepton vertex is conj(Dd[I,J]) (the neutrino
    carries the flavour of the final lepton J, so that the SM W vertex is flavour diagonal; the singlet coupling 2f is neglected, f ~ 1e-6).
    Checks (tests): diagonal limit against an independent quadrature of Ilisie's typeset formulae, charged-mass-basis independence
    for degenerate states, decoupling.  NOT checked against an independent two-loop calculation: the overall sign and size rest on
    the transcription, so the production gate varies HW_SCALE in {0, 1, -1}.  The definition of G1 is inferred as G(1, .)."""
    if "raw_h" not in trilin:
        return np.zeros((3, 3), complex), np.zeros((3, 3), complex)
    c, s = np.cos(sc["phi"]), np.sin(sc["phi"])
    u = np.array([c, -s]); e = np.array([[c, s], [-s, c]])        # H+ content and (H+, h+) content of h1+, h2+
    mk = np.array([sc["m1"], sc["m2"]]); tb = sc["tb"]; sba, cba = sc["sba"], sc["cba"]
    x = _XG; w = _WG; y1 = x * (1 - x)
    pref = ALPHA / (128 * np.sqrt(2) * np.pi**3 * SW2)
    zu = -tb                                                         # varsigma_u = varsigma_d (type-I-like quarks)
    wa, wb, sg, kind, cf = [], [], [], [], []                         # one row per G evaluation: G(wa, wb) with sign sg
    for k in range(2):
        mH2 = mk[k] ** 2; dm = mH2 - MW**2
        wa += [MT**2 / mH2, MT**2 / MW**2]; wb += [MB**2 / mH2, MB**2 / MW**2]; sg += [1.0, -1.0]; kind += [4, 4]
        cf += [2 * NC * u[k] ** 2 / (V * dm)] * 2
        for ph, Z5, R2 in (("h", sba * cba, cba), ("H", -cba * sba, -sba)):
            mp2 = mass[ph] ** 2
            wa += [MW**2 / mH2, 1.0]; wb += [mp2 / mH2, mp2 / MW**2]; sg += [1.0, -1.0]; kind += [5, 5]
            cf += [u[k] ** 2 * Z5 / (V * dm) * 1.0] * 2
            dHH, dhh, dHh = trilin["raw_" + ph]; Lam2 = np.array([[dHH, dHh], [dHh, dhh]])
            for j in range(2):
                lam = e[k] @ Lam2 @ e[j]
                wa += [mk[j] ** 2 / mH2, mk[j] ** 2 / MW**2]; wb += [mp2 / mH2, mp2 / MW**2]; sg += [1.0, -1.0]; kind += [6, 6]
                cf += [V * R2 * u[k] * lam * u[j] / dm] * 2
    wa, wb, sg, kind, cf = (np.array(a_) for a_ in (wa, wb, sg, kind, cf))
    Gm = _G(wa[:, None], wb[:, None], x[None, :]) * sg[:, None]       # [rows, x]
    # x-dependent weights of the three sets; (5) depends on k and phi through (mH2 + MW^2 - mp2)
    poly4 = (QT * x + QB * (1 - x)) * zu * (MB**2 * y1 + MT**2 * x * (1 + x))
    poly6 = x**2 * (x - 1)
    P = np.empty_like(Gm)
    r = 0
    for k in range(2):
        mH2 = mk[k] ** 2
        P[r:r + 2] = poly4; r += 2
        for ph in ("h", "H"):
            P[r:r + 2] = x**2 * ((mH2 + MW**2 - mass[ph] ** 2) * (1 - x) - 4 * MW**2); r += 2
            for j in range(2):
                P[r:r + 2] = poly6; r += 2
    coef = float(np.sum(cf * (Gm * P * w[None, :]).sum(1)))
    AL = pref * coef * Dd.conj().T / M[None, :]                  # A_L[J, I] = pref coef conj(Dd[I, J]) / m_I
    AR = pref * coef * Dd / M[None, :]                           # hermiticity: X_R[J, I] = conj(X_L[I, J]), X = m_I A
    return AL, AR


def amplitudes(Lam, mass, f, Dd, sc, trilin, parts=False):
    """Return (A_L, A_R) 3x3 [J, I]; with parts=True also a dict of the components."""
    P = {k: [np.zeros((3, 3), complex), np.zeros((3, 3), complex)] for k in ("n1", "ch", "bzf", "bzW", "bzC", "bzHW")}
    for p, L in Lam.items():
        m2 = mass[p] ** 2
        lg = np.log(m2 / M**2) - 1.5
        Lc = L.conj()
        W = MKI * lg[:, None]                                    # [k, I]
        E1 = Lc.T @ L                                            # sum_k L*_kJ L_kI
        E2 = L @ Lc.T                                            # sum_k L_Jk L*_Ik
        aL = Lc.T @ (Lc.T * W) + (E1 + MJI * E2) / 6
        aR = L @ (L * W) + (E2 + MJI * E1) / 6
        P["n1"][0] += aL / (16 * PI2 * m2); P["n1"][1] += aR / (16 * PI2 * m2)
    c2, s2 = np.cos(sc["phi"]) ** 2, np.sin(sc["phi"]) ** 2
    KH = c2 / sc["m1"] ** 2 + s2 / sc["m2"] ** 2; Kf = s2 / sc["m1"] ** 2 + c2 / sc["m2"] ** 2
    y = 2 * f
    SH = Dd.conj().T @ Dd; Sf = y.conj().T @ y
    k = -1 / (192 * PI2)
    P["ch"][0] += k * (KH * SH + MJI * Kf * Sf); P["ch"][1] += k * (Kf * Sf + MJI * KH * SH)
    K = bz_kernels(Lam, mass, sc, trilin)
    for p, L in Lam.items():
        for key, lab in (("f", "bzf"), ("W", "bzW"), ("C", "bzC")):
            R = L * K[p][key] / M[None, :]
            P[lab][1] += R
            P[lab][0] += (L * K[p][key]).conj().T / M[None, :]
    if HW_ON:
        hl, hr = bz_HW(Dd, mass, sc, trilin); P["bzHW"][0], P["bzHW"][1] = HW_SCALE*hl, HW_SCALE*hr
    AL = sum(v[0] for v in P.values()); AR = sum(v[1] for v in P.values())
    return (AL, AR, P) if parts else (AL, AR)


def diagonal_moments(AL, AR):
    """a_I and d_I (e cm) from the diagonal amplitudes."""
    a = M**2 * np.real(np.diag(AL) + np.diag(AR))
    d = M * np.imag(np.diag(AR) - np.diag(AL)) / 2 * 1.973e-14
    return a, d


def sm_reference(sc):
    """Diagonal amplitudes of the SM Higgs alone (one loop + top/b/tau/W Barr-Zee); subtracted from a_l and d_l,
    which are already part of a_l^SM. LFV amplitudes are unaffected (the SM part is flavour diagonal)."""
    Z = np.zeros((3, 3), complex)
    Lam = {"h": np.diag(M / V).astype(complex), "H": Z, "A": Z}
    scs = dict(sc, sba=1.0, cba=0.0)
    return amplitudes(Lam, {"h": 125.5, "H": 1e3, "A": 1e3}, Z, Z, scs, {"h": (0.0, 0.0), "H": (0.0, 0.0)})
