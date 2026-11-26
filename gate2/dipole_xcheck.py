"""Independent cross-check of the one-loop l_I -> l_J gamma dipole amplitudes (neutral-scalar and charged-scalar loops).

Nothing here is shared with zee/dipfull.py: the amplitude is built from the Feynman rules with explicit 4x4 Dirac matrices and
on-shell spinors, the Feynman-parameter integral is done numerically (nested Gauss-Legendre panels, no closed-form loop
functions), and the dipole coefficients are extracted by a least-squares fit of the spinor-sandwiched vertex to the operators
  gamma.eps P_{L,R},   i sigma^{mu nu} eps_mu q_nu P_{L,R}
(over the four spin combinations and both transverse polarisations; the fit residual is returned and must vanish).

Conventions (those of dipfull): -L = phi ebar_L Lam e_R + h.c. (real phi), i.e. the vertex ebar_a (Lam_ab P_R + conj(Lam_ba) P_L) e_b;
vertex function Gamma^mu = gamma^mu F1 + i sigma^{mu nu} q_nu (c_L P_L + c_R P_R) with q = p' - p; A_{L,R} = c_{L,R}/m_I.
Feynman parameters, q^2 = 0, loop momentum shifted; only the l-independent part of the numerator is kept (the l^2 part
is proportional to gamma^mu and cannot feed the dipole).
"""
import numpy as np

g0 = np.diag([1, 1, -1, -1]).astype(complex)
sig = [np.array([[0, 1], [1, 0]], complex), np.array([[0, -1j], [1j, 0]]), np.array([[1, 0], [0, -1]], complex)]
gam = [g0] + [np.block([[np.zeros((2, 2)), s], [-s, np.zeros((2, 2))]]) for s in sig]
g5 = np.block([[np.zeros((2, 2)), np.eye(2)], [np.eye(2), np.zeros((2, 2))]]).astype(complex)
PR, PL = (np.eye(4) + g5) / 2, (np.eye(4) - g5) / 2
METRIC = np.diag([1., -1, -1, -1])


def slash(p):
    return sum(METRIC[m, m] * p[m] * gam[m] for m in range(4))


def spinor(p, m, s):
    chi = np.zeros(4, complex); chi[s] = 1
    return (slash(p) + m * np.eye(4)) @ chi / np.sqrt(p[0] + m)


def _gl(n, a, b):
    x, w = np.polynomial.legendre.leggauss(n)
    return 0.5 * (b - a) * x + 0.5 * (a + b), 0.5 * (b - a) * w


def _panels(n=12, lo=1e-16):
    """geometric panels towards 0 and 1 (the integrands have scales m_k^2/m_phi^2 down to ~1e-9)."""
    e = np.concatenate([[0.0], 10.0 ** np.arange(np.log10(lo), -0.4, 1.0), [0.5]])
    e = np.unique(np.concatenate([e, 1 - e]))
    xs, ws = [], []
    for a, b in zip(e[:-1], e[1:]):
        x, w = _gl(n, a, b); xs.append(x); ws.append(w)
    return np.concatenate(xs), np.concatenate(ws)


_X, _W = _panels()


def _tri(f):
    """integral over x1+x2+x3 = 1 of f(x2, x3); x3 on geometric panels, x2 in [0, 1 - x3] likewise (scaled)."""
    tot = 0.0
    for x3, w3 in zip(_X, _W):
        t = _X * (1 - x3)                       # x2 = t
        tot = tot + w3 * (1 - x3) * np.sum(_W * f(t, x3))
    return tot


def _kin(mI, mJ):
    w = (mI**2 - mJ**2) / (2 * mI)
    p = np.array([mI, 0, 0, 0.]); kg = np.array([w, 0, 0, w]); pp = p - kg
    return p, pp, pp - p                          # p, p', q = p' - p


def _fit(Nmat, p, pp, q, mI, mJ):
    """Nmat[mu] = 4x4 matrices N^mu (contravariant). The 1PI vertex of a single diagram is not gauge invariant by itself, so it is fitted
    with ALL four values of mu to the six on-shell structures {gamma^mu, (p+p')^mu, q^mu} x {P_L, P_R}.  The Gordon identities
        i sigma^{mu nu} q_nu P_L = -(p+p')^mu P_L + m_I gamma^mu P_R + m_J gamma^mu P_L   (and L <-> R)
    show that the dipole coefficient of the gauge-invariant sum is c = -b, b = coefficient of (p+p')^mu (the gamma^mu and q^mu pieces
    are cancelled by the leg diagrams / Ward identity).  Returns (c_L, c_R, fit residual)."""
    P = p + pp
    rows, rhs = [], []
    for sJ in range(2):
        for sI in range(2):
            uJ = spinor(pp, mJ, sJ); uI = spinor(p, mI, sI); ubJ = uJ.conj() @ g0
            for mu in range(4):
                rhs.append(ubJ @ Nmat[mu] @ uI)
                rows.append([ubJ @ gam[mu] @ PL @ uI, ubJ @ gam[mu] @ PR @ uI, P[mu] * (ubJ @ PL @ uI), P[mu] * (ubJ @ PR @ uI),
                             q[mu] * (ubJ @ PL @ uI), q[mu] * (ubJ @ PR @ uI)])
    A = np.array(rows); b = np.array(rhs)
    x, *_ = np.linalg.lstsq(A, b, rcond=None)
    res = np.linalg.norm(A @ x - b) / max(np.linalg.norm(b), 1e-300)
    sv = np.linalg.svd(A, compute_uv=False)
    assert sv[-1] > 1e-9 * sv[0], "operator basis is degenerate"
    return -x[2], -x[3], res


_MONO = [lambda x2, x3: 1.0 + 0 * x2, lambda x2, x3: x2, lambda x2, x3: x3, lambda x2, x3: x2 * x3, lambda x2, x3: x2**2, lambda x2, x3: x3**2]
_PTS = [(0., 0.), (1., 0.), (0., 1.), (1., 1.), (2., 0.), (0., 2.)]


def _grid():
    X3 = np.repeat(_X, len(_X)); W3 = np.repeat(_W, len(_X))
    T = np.tile(_X, len(_X)); WT = np.tile(_W, len(_X))
    return T * (1 - X3), X3, W3 * WT * (1 - X3)           # x2, x3, weights on the simplex


_X2, _X3, _WT = _grid()


def _poly_integrals(D):
    """int dx2 dx3 mono_m / D for the six monomials."""
    return np.array([np.sum(_WT * m(_X2, _X3) / D) for m in _MONO])


def _coeff_matrices(Nfun):
    """Nfun(x2, x3) is a polynomial of degree <= 2 in (x2, x3) with matrix values: recover its six monomial coefficient matrices."""
    A = np.array([[m(*pt) for m in _MONO] for pt in _PTS])
    vals = np.array([Nfun(*pt) for pt in _PTS])                     # [6, ...]
    return np.tensordot(np.linalg.inv(A), vals, axes=(1, 0))        # [6, ...]


def neutral_loop(L, mphi, masses):
    """one real neutral scalar of mass mphi with coupling matrix L (-L = phi ebar_L L e_R + h.c.), charged lepton k in the loop.
    Returns (A_L, A_R, worst fit residual) as 3x3 [J, I] arrays (J != I)."""
    K = 1 / (16 * np.pi**2)
    AL = np.zeros((3, 3), complex); AR = np.zeros((3, 3), complex); worst = 0.0
    G = lambda a, b: L[a, b] * PR + np.conj(L[b, a]) * PL                          # vertex ebar_a ... e_b
    for I in range(3):
        for J in range(3):
            if I == J:
                continue
            mI, mJ = masses[I], masses[J]
            p, pp, q = _kin(mI, mJ)
            N = [np.zeros((4, 4), complex) for _ in range(4)]
            for k in range(3):
                mk = masses[k]
                gJ, gI = G(J, k), G(k, I)
                D = (1 - _X3) * mk**2 + _X3 * mphi**2 - (1 - _X2 - _X3) * _X3 * mI**2 - _X2 * _X3 * mJ**2
                ints = _poly_integrals(D)
                for mu in range(4):
                    def Nf(x2, x3):
                        a = x3 * p - x2 * q; ap = x3 * p + (1 - x2) * q
                        return gJ @ (slash(ap) + mk * np.eye(4)) @ gam[mu] @ (slash(a) + mk * np.eye(4)) @ gI
                    C = _coeff_matrices(Nf)
                    N[mu] += K * np.tensordot(ints, C, axes=(0, 0))
            cL, cR, res = _fit(N, p, pp, q, mI, mJ)
            AL[J, I] = cL / mI; AR[J, I] = cR / mI; worst = max(worst, res)
    return AL, AR, worst


def charged_loop(a, b, mS, masses):
    """charged scalar S^- (charge of the lepton) of mass mS, massless neutral fermion in the loop, photon on the scalar.
    -L = S^+ nubar (a_{nu e} P_R + b_{nu e} P_L) e + h.c., a, b: 3x3 [nu, e]. Returns (A_L, A_R, worst residual) [J, I]."""
    K = 1 / (16 * np.pi**2)
    AL = np.zeros((3, 3), complex); AR = np.zeros((3, 3), complex); worst = 0.0
    for I in range(3):
        for J in range(3):
            if I == J:
                continue
            mI, mJ = masses[I], masses[J]
            p, pp, q = _kin(mI, mJ)
            N = [np.zeros((4, 4), complex) for _ in range(4)]
            for nu in range(3):
                gI = a[nu, I] * PR + b[nu, I] * PL                                    # nubar ... e_I
                gJ = np.conj(a[nu, J]) * PL + np.conj(b[nu, J]) * PR                  # ebar_J ... nu
                if not (np.any(gI) and np.any(gJ)):
                    continue
                x2, x3 = _X2, _X3; x1 = 1 - x2 - x3
                D = (x2 + x3) * mS**2 - x1 * (x2 * mI**2 + x3 * mJ**2)
                # N0 = gJ (x2 p + x3 p')slash gI [(1 - 2 x2) p + (1 - 2 x3) p']^mu : polynomial of degree 2
                ints = _poly_integrals(D)
                for mu in range(4):
                    def Nf(y2, y3):
                        return gJ @ slash(y2 * p + y3 * pp) @ gI * ((1 - 2 * y2) * p[mu] + (1 - 2 * y3) * pp[mu])
                    # here the roles of (x2, x3) are (weight of p, weight of p'); the Feynman parametrisation below uses the same labels
                    C = _coeff_matrices(Nf)
                    N[mu] += K * np.tensordot(ints, C, axes=(0, 0))
            cL, cR, res = _fit(N, p, pp, q, mI, mJ)
            AL[J, I] = cL / mI; AR[J, I] = cR / mI; worst = max(worst, res)
    return AL, AR, worst


def compare(seed=11, masses=(125.5, 300., 1000., 3000.), ntrial=3):
    """Compare dipfull's one-loop n1 / ch parts with the independent calculation for random complex flavour-violating couplings.
    Returns dict with the worst relative deviation |dA| / max(|A_L|, |A_R|) per transition."""
    from zee import dipfull as D
    from zee.observables import M_LEP
    M = np.asarray(M_LEP); Z = np.zeros((3, 3), complex)
    sc0 = dict(phi=0., m1=400., m2=900., sba=1., cba=0., tb=1., lam3=0., mu=0.)
    rng = np.random.default_rng(seed); out = {"neutral": [], "charged": []}
    for mphi in masses:
        for _ in range(ntrial):
            L = (rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))) * 10 ** rng.uniform(-4, 0, size=(3, 3))
            AL, AR, res = neutral_loop(L, mphi, M)
            for p in ("H", "A"):
                Lam = {"h": Z.copy(), "H": Z.copy(), "A": Z.copy()}; Lam[p] = L
                c = D.amplitudes(Lam, {"h": mphi, "H": mphi, "A": mphi}, Z, Z, sc0, {"h": (0, 0), "H": (0, 0)}, parts=True)[2]["n1"]
                w = max(abs(x - y) / max(abs(AL[J, I]), abs(AR[J, I])) for J in range(3) for I in range(3) if I != J
                        for x, y in ((AL[J, I], c[0][J, I]), (AR[J, I], c[1][J, I])))
                out["neutral"].append(dict(mphi=mphi, field=p, fit_resid=res, worst=float(w)))
    for phi in (0.0, 0.4, 1.1):
        sc = dict(sc0, phi=phi, m1=250., m2=700.)
        cx = lambda: (rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))) * 10 ** rng.uniform(-3, -0.3, size=(3, 3))
        Dd = cx(); f = cx(); f = f - f.T; y = 2 * f
        P = D.amplitudes({k: Z.copy() for k in "hHA"}, {"h": 500., "H": 500., "A": 500.}, f, Dd, sc, {"h": (0, 0), "H": (0, 0)}, parts=True)[2]["ch"]
        c, s = np.cos(phi), np.sin(phi); ALx = np.zeros((3, 3), complex); ARx = np.zeros((3, 3), complex); res = 0.0
        for mk, u, v in ((sc["m1"], c, s), (sc["m2"], -s, c)):
            for a, b in ((u * Dd, Z), (Z, v * y)):
                al, ar, r = charged_loop(a, b, mk, M); ALx += al; ARx += ar; res = max(res, r)
        w = max(abs(x - y_) / max(abs(ALx[J, I]), abs(ARx[J, I])) for J in range(3) for I in range(3) if I != J
                for x, y_ in ((ALx[J, I], P[0][J, I]), (ARx[J, I], P[1][J, I])))
        out["charged"].append(dict(phi=phi, fit_resid=res, worst=float(w)))
    return out


if __name__ == "__main__":
    import json, sys
    r = compare()
    json.dump(r, open(sys.argv[1] if len(sys.argv) > 1 else "results/dipole_xcheck.json", "w"), indent=1)
    print("neutral worst by mphi:", {m: max(x["worst"] for x in r["neutral"] if x["mphi"] == m) for m in sorted({x["mphi"] for x in r["neutral"]})})
    print("charged worst:", [x["worst"] for x in r["charged"]])
