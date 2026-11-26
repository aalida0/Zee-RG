"""Momentum-dependent one-loop scalar self-energies of the Zee scalar sector (Landau gauge, MSbar), report v11.

Implements the one-loop self-energy matrix Pi_ij(s), s = -p^2, of the ten real scalars in the general-theory form of S. P. Martin, Phys. Rev. D70 (2004) 016005 [hep-ph/0312092],
eq. (3.1)-(3.8), at xi = 0:

  Pi_ij = 1/2 lam^{ijkk} A(m_k^2) - 1/2 lam^{ikl} lam^{jkl} B(m_k^2, m_l^2)              (scalar tadpole, scalar bubble)
        + Re[y^{KLi} y*_{KLj}] B_FF(m_K^2, m_L^2) + Re[y^{KLi} y^{K'L'j} M*_{KK'} M*_{LL'}] B_FFbar   (fermions; colour factor 3)
        + g^{aik} g^{ajk} B_SV(m_k^2, m_a^2) + 1/2 g^{aaij} A_V(m_a^2) + 1/2 g^{abi} g^{abj} B_VV(m_a^2, m_b^2)   (vectors)

with the overall factor 1/(16 pi^2) outside.  Field conventions: R_i real scalars (zee/rgeq ordering p0..p9), vector couplings from the gauge generators theta^a (rgeq._gens),
g^{aij} = theta^a_{ij}, g^{abi} = d_i M_V^2{}_{ab}, g^{abij} = d_i d_j M_V^2{}_{ab}, M_V^2{}_{ab} = (theta^a phi).(theta^b phi).  The top-bottom system is the only fermion sector
(M^{t tc} = y_t H1^0 + rho_t H2^0, M^{b tc} = -(y_t H1^+ + rho_t H2^+), mass-basis rotation U with M^dagger M diagonal).  Only the real part is kept (no widths).

Validation (tests/test_selfenergy.py): Pi_ij(0)/(16 pi^2) equals the Hessian of the one-loop effective potential at a generic non-tachyonic field point (tests every piece except
B_SV, which vanishes at s = 0 in Landau gauge); the B_SV normalisation is tested against the scalar-QED result and the RG invariance of its pole mass.
"""
import numpy as np
from . import rgeq
from . import matching as M

XMIN = 1e-7


def lnb(x, q2):
    return np.log(x/q2) if x > 0 else 0.0


def A_(x, q2):
    return x*(np.log(x/q2) - 1) if x > 0 else 0.0


def B_(x, y, s, q2):
    """Re B(x,y;s) = -Re int_0^1 dt ln[(t x + (1-t) y - t(1-t) s)/Q^2]"""
    x = max(x, 0.0); y = max(y, 0.0)
    if abs(s) < 1e-8*max(x + y, 1.0):
        if abs(x - y) < 1e-9*max(x + y, 1.0):
            return -(np.log(x/q2) if x > 0 else 0.0) if x > 0 else 0.0
        return -((x*lnb(x, q2) - y*lnb(y, q2))/(x - y) - 1.0)
    # arg(t) = s t^2 + (x - y - s) t + y = s (t - t1)(t - t2)
    disc = (x - y - s)**2 - 4*s*y
    sq = np.sqrt(complex(disc))
    t1, t2 = (-(x - y - s) + sq)/(2*s), (-(x - y - s) - sq)/(2*s)
    def F(a):          # Re int_0^1 ln(t - a) dt
        if abs(a.imag) < 1e-14*max(1, abs(a.real)):
            a = a.real
            t1_ = (1 - a)*np.log(abs(1 - a)) if abs(1 - a) > 0 else 0.0
            t2_ = a*np.log(abs(a)) if abs(a) > 0 else 0.0
            return t1_ + t2_ - 1.0
        return ((1 - a)*np.log(1 - a) + a*np.log(-a)).real - 1.0
    return -(np.log(abs(s)/q2) + F(t1) + F(t2))


def Lx(f, x):
    """L_x f(x) = [f(x) - f(0)]/x in Landau gauge; at x -> 0 the derivative at zero (finite difference with a tiny photon mass)"""
    if x > XMIN: return (f(x) - f(0.0))/x
    h = 1e-4
    return (f(h) - f(0.0))/h


def A_V(x, q2):
    return 4*A_(x, q2) + 2*x - Lx(lambda z: z*A_(z, q2), x)


def B_SV(x, y, s, q2):
    """x scalar mass^2, y vector mass^2"""
    if y <= XMIN:
        return 3*(x + s)*B_(0.0, x, s, q2) + 3*A_(x, q2) - 2*s
    f = lambda z: (x + z - s)*A_(z, q2) - (x - s)**2*B_(x, z, s, q2)
    return (2*x - y + 2*s)*B_(x, y, s, q2) + A_(x, q2) - 2*A_(y, q2) + Lx(f, y)


def B_VV(x, y, s, q2):
    g = lambda u, v: u*A_(v, q2) + v*A_(u, q2) + (2*s*(u + v) - u*u - v*v - s*s)*B_(u, v, s, q2)
    t1 = -3.5*B_(x, y, s, q2) + 2.0
    t2 = 0.5*Lx(lambda u: u*B_(u, y, s, q2), x) + 0.5*Lx(lambda v: v*B_(x, v, s, q2), y)
    # L_x L_y g = [g(x,y) - g(0,y) - g(x,0) + g(0,0)]/(x y) with the zero-mass limits taken as derivatives
    def Ly(u):
        return Lx(lambda v: g(u, v), y)
    t3 = 0.25*Lx(Ly, x)
    return t1 + t2 + t3


def _fermion_sector(phi, yt, rt):
    """Weyl basis (t, b, tc): M (3x3 complex symmetric) and its derivative tensor y[I,J,k] = dM^{IJ}/dR_k (real field index k)"""
    H1p, H10 = (phi[0] + 1j*phi[1])/np.sqrt(2), (phi[2] + 1j*phi[3])/np.sqrt(2)
    H2p, H20 = (phi[4] + 1j*phi[5])/np.sqrt(2), (phi[6] + 1j*phi[7])/np.sqrt(2)
    a = yt*H10 + rt*H20; b = -(yt*H1p + rt*H2p)
    Mm = np.zeros((3, 3), complex); Mm[0, 2] = Mm[2, 0] = a; Mm[1, 2] = Mm[2, 1] = b
    y = np.zeros((3, 3, 10), complex)
    s2 = 1/np.sqrt(2)
    for k, (coef_a, coef_b) in {0: (0, -yt*s2), 1: (0, -1j*yt*s2), 2: (yt*s2, 0), 3: (1j*yt*s2, 0), 4: (0, -rt*s2), 5: (0, -1j*rt*s2), 6: (rt*s2, 0), 7: (1j*rt*s2, 0)}.items():
        y[0, 2, k] = y[2, 0, k] = coef_a; y[1, 2, k] = y[2, 1, k] = coef_b
    return Mm, y


class Pi:
    """self-energy engine at a field point phi for given tree tensors T=(mu2,kappa,lambda), couplings g=(g',g), top Yukawas.  Goldstone masses may be set to zero (goldstone0)."""

    def __init__(self, phi, T, g, yt, rt, mu0, mass2=None):
        self.q2 = mu0**2; self.phi = np.asarray(phi, float); mu2, kap, lam = T
        M2 = mass2_at(self.phi, T) if mass2 is None else mass2
        w, O = np.linalg.eigh(M2); self.O = O
        self.m2 = np.where(np.abs(w) < 1e-3, 0.0, w)               # exact Goldstone zero modes (tree-level tadpole-free point)
        c3 = kap + np.einsum('ijkl,l->ijk', lam, self.phi)
        self.l3 = np.einsum('ijk,ia,jb,kc->abc', c3, O, O, O)
        self.l4 = np.einsum('ijkl,ia,jb,kc,ld->abcd', lam, O, O, O, O)
        th = M._GENS.get(tuple(g)) or M._GENS.setdefault(tuple(g), rgeq._gens(*g))
        w4 = np.array([t @ self.phi for t in th]); MV = w4 @ w4.T
        mv, Ov = np.linalg.eigh(MV); self.mv = np.clip(mv, 0.0, None); self.Ov = Ov
        thp = np.einsum('ba,bij->aij', Ov, np.array(th))                 # theta'^a in the vector mass basis (field basis for scalars)
        self.thS = np.einsum('aij,ik,jl->akl', thp, O, O)                # g^{a k l} in scalar mass basis
        wp = np.einsum('ba,bi->ai', Ov, w4)                              # theta'^a phi
        gab = np.einsum('aim,bm->abi', np.transpose(thp, (0, 2, 1)), wp) + np.einsum('bim,am->abi', np.transpose(thp, (0, 2, 1)), wp)
        self.gabi = np.einsum('abi,ik->abk', gab, O)                     # g^{abi}: sum over m of theta^a_{mi} (theta^b phi)_m + (a<->b)
        gaaij = np.einsum('aim,ajm->aij', np.transpose(thp, (0, 2, 1)), np.transpose(thp, (0, 2, 1)))  # (theta^T theta)_{ij} per vector
        self.gaa = np.einsum('aij,ik,jl->akl', gaaij, O, O)
        Mm, y = _fermion_sector(self.phi, yt, rt)
        Hm = Mm.conj().T @ Mm; mf, U = np.linalg.eigh(Hm); self.mf = np.clip(mf, 0.0, None)
        self.Mf = U.T @ Mm @ U; self.yf = np.einsum('IK,IJk,JL->KLk', U, y, U)
        self.yf = np.einsum('KLk,ki->KLi', self.yf, O)

    def at(self, s, parts='SFV'):
        q2 = self.q2; n = 10; P = np.zeros((n, n))
        if 'S' in parts:
            Ak = np.array([A_(m, q2) if m > 0 else 0.0 for m in self.m2])
            P += 0.5*np.einsum('ijkk,k->ij', self.l4, Ak)
            Bkl = np.array([[B_(self.m2[k], self.m2[l], s, q2) for l in range(n)] for k in range(n)])
            P -= 0.5*np.einsum('ikl,jkl,kl->ij', self.l3, self.l3, Bkl)
        if 'F' in parts:
            mf = self.mf; nf = len(mf)
            Bf = np.array([[B_(mf[K], mf[L], s, q2) for L in range(nf)] for K in range(nf)])
            BFF = (mf[:, None] + mf[None, :] - s)*Bf - np.array([A_(m, q2) for m in mf])[:, None] - np.array([A_(m, q2) for m in mf])[None, :]
            F1 = np.einsum('KLi,KLj,KL->ij', self.yf, self.yf.conj(), BFF).real
            Z = np.einsum('KP,PQj,LQ->KLj', self.Mf.conj(), self.yf, self.Mf.conj())
            F2 = np.einsum('KLi,KLj,KL->ij', self.yf, Z, 2*Bf).real
            P += 3*(F1 + F2)
        if 'V' in parts:
            nv = len(self.mv); mv = self.mv
            SV = np.array([[B_SV(self.m2[k], mv[a], s, q2) for a in range(nv)] for k in range(n)])
            P += np.einsum('aik,ajk,ka->ij', self.thS, self.thS, SV)
            AV = np.array([A_V(m, q2) for m in mv])
            P += np.einsum('aij,a->ij', self.gaa, AV)
            BV = np.array([[B_VV(mv[a], mv[b], s, q2) for b in range(nv)] for a in range(nv)])
            P += 0.5*np.einsum('abi,abj,ab->ij', self.gabi, self.gabi, BV)
        return self.O @ P @ self.O.T


def mass2_at(phi, T):
    return M.mass2_at(phi, T)
