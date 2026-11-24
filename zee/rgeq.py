"""One-loop quartic RGEs (Machacek-Vaughn, real scalars, 2-comp. fermions) for the general Zee scalar sector.
Real scalars: (H1+,H10,H2+,H20,eta+) each as (Re,Im). V4 = 1/4! lam_abcd phi^4."""
import itertools, numpy as np
from scipy.integrate import solve_ivp
from . import rge
NS = 10; KAP = 0.5
PERMS = list(itertools.permutations(range(4)))
def _real(t):
    """complex hermitian generator t (n x n) -> real 2n x 2n acting on (Re,Im) pairs."""
    n = t.shape[0]; R = np.zeros((2 * n, 2 * n)); tr, ti = t.real, t.imag
    for i in range(n):
        for j in range(n):
            R[2 * i, 2 * j] = -ti[i, j]; R[2 * i, 2 * j + 1] = -tr[i, j]
            R[2 * i + 1, 2 * j] = tr[i, j]; R[2 * i + 1, 2 * j + 1] = -ti[i, j]
    return R
def _gens(g1, g2):
    sig = [np.array([[0, 1], [1, 0]], complex) / 2, np.array([[0, -1j], [1j, 0]]) / 2, np.array([[1, 0], [0, -1]], complex) / 2]
    A = np.zeros((NS, NS))
    A[0:4, 0:4] = g1 * 0.5 * _real(np.eye(2, dtype=complex)); A[4:8, 4:8] = A[0:4, 0:4]
    A[8:10, 8:10] = g1 * 1.0 * _real(np.eye(1, dtype=complex))
    th = [A]
    for s in sig:
        B = np.zeros((NS, NS))
        B[0:4, 0:4] = g2 * _real(s); B[4:8, 4:8] = B[0:4, 0:4]
        th.append(B)
    return th
def quartic_beta(lam, Y, g):
    g1, g2 = g[0], g[1]
    th = _gens(g1, g2)
    C2 = -sum(t @ t for t in th)
    L2 = (np.einsum('abef,efcd->abcd', lam, lam) + np.einsum('acef,efbd->abcd', lam, lam)
          + np.einsum('adef,efbc->abcd', lam, lam))
    Yc = np.conj(np.transpose(Y, (0, 2, 1)))
    ab = np.einsum('aij,bjk->abik', Y, Yc); cd = np.einsum('cij,djk->cdik', Y, Yc)
    T = np.einsum('abik,cdki->abcd', ab, cd)
    H = sum(np.transpose(T, p) for p in PERMS) / 4.0
    Y2S = np.array([[0.5 * np.trace(Yc[a] @ Y[b] + Yc[b] @ Y[a]) for b in range(NS)] for a in range(NS)])
    def lead(M, X):
        return (np.einsum('ak,kbcd->abcd', M, X) + np.einsum('bk,akcd->abcd', M, X) + np.einsum('ck,abkd->abcd', M, X) + np.einsum('dk,abck->abcd', M, X))
    LY = lead(Y2S.real, lam); LS = lead(C2, lam)
    anti = [[t1 @ t2 + t2 @ t1 for t2 in th] for t1 in th]
    A = np.zeros((NS,) * 4)
    for i in range(len(th)):
        for j in range(len(th)):
            M = anti[i][j]
            A += np.einsum('ab,cd->abcd', M, M) + np.einsum('ac,bd->abcd', M, M) + np.einsum('ad,bc->abcd', M, M)
    return (L2 - 8 * KAP * H.real + 2 * KAP * LY - 3 * LS + 3 * A) / (16 * np.pi**2)
def rhs(t, z):
    zy = z[:58]; lam = z[58:].reshape((NS,) * 4)
    Y1, Y2, f, yt, g = rge.unpack(zy)
    Y = rge.full_Y(Y1, Y2, f, yt)
    dq = quartic_beta(lam, Y, g)
    return np.concatenate([rge.rhs(t, zy), dq.ravel()])
# ---- potential tensor from HG-basis couplings (real) ----
def V4(phi, c):
    """HG quartic potential, c=(l1..l10, lh), real couplings; phi real vector of 10: (H1+,H10,H2+,H20,eta+)xRe,Im."""
    z = (phi[0::2] + 1j * phi[1::2]) / np.sqrt(2)
    H1, H2, h = z[0:2], z[2:4], z[4]
    l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, lh = c
    d = lambda a, b: np.vdot(a, b)
    a11, a22, a12, a21 = d(H1, H1).real, d(H2, H2).real, d(H1, H2), d(H2, H1)
    hh = abs(h)**2
    return (0.5 * l1 * a11**2 + 0.5 * l2 * a22**2 + l3 * a11 * a22 + l4 * (a12 * a21).real
            + (l5 * a12**2).real + 2 * ((l6 * a11 + l7 * a22) * a12).real + lh * hh**2 + l8 * hh * a11 + l9 * hh * a22 + 2 * (l10 * hh * a12).real)
def tensor_from(c, cache={}):
    """lam_abcd = d^4 V4 / dphi^4 via polarization of a homogeneous quartic (exact, finite differences on polynomial)."""
    # use symmetric multilinear extraction: lam_abcd = sum over sign vectors (inclusion-exclusion)
    lam = np.zeros((NS,) * 4); E = np.eye(NS)
    for a, b, cc, d in itertools.combinations_with_replacement(range(NS), 4):
        # polarization: coefficient of x_a x_b x_c x_d
        idx = (a, b, cc, d); tot = 0.0
        for m in range(1, 16):
            sel = [i for i in range(4) if (m >> i) & 1]
            phi = sum(E[idx[i]] for i in sel)
            tot += (-1)**(4 - len(sel)) * V4(phi, c)
        val = tot                                   # = d^4V/(da db dc dd) up to symmetric factor
        for p in set(itertools.permutations(idx)): lam[p] = val
    return lam
def min_quartic(lam, n=300, seed=0):
    """min of V4/|phi|^4 over the sphere (negative => unbounded below)."""
    from scipy.optimize import minimize
    rng = np.random.default_rng(seed); best = 1e9
    f = lambda p: np.einsum('abcd,a,b,c,d', lam, p, p, p, p) / 24 / (p @ p)**2
    for _ in range(n):
        r = minimize(f, rng.normal(size=NS), method='BFGS', options={'gtol': 1e-9})
        best = min(best, r.fun)
    return best
