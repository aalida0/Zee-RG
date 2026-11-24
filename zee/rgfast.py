"""Fast closed 11-parameter form of the quartic RGE (lam1..lam10, lamh), leptonic Yukawas dropped (|y_l|<<1).
beta_c = A.c.c + (yt^2 B_t + g1^2 B_1 + g2^2 B_2).c + C(g1,g2,yt); built numerically from zee.rgeq and cached."""
import os, numpy as np
from scipy.integrate import solve_ivp
from . import rge, rgeq
_F = os.path.join(os.path.dirname(__file__), '_rgfast.npz')
NC = 11
def _build():
    Ts = [rgeq.tensor_from(np.eye(NC)[i]) for i in range(NC)]
    M = np.array([t.ravel() for t in Ts]).T
    pinv = np.linalg.pinv(M)
    def proj(T):
        c = pinv @ T.ravel(); res = np.abs(M @ c - T.ravel()).max(); return c, res
    Z = np.zeros((3, 3), complex)
    def beta(c, g, yt):
        T = sum(ci * Ti for ci, Ti in zip(c, Ts))
        Y = rge.full_Y(Z, Z, Z, yt)
        return proj(rgeq.quartic_beta(T, Y, np.array([g[0], g[1], 1.0])))
    g0 = np.zeros(2); worst = 0
    Q = lambda c: beta(c, g0, 0.0)[0]
    A = np.zeros((NC, NC, NC))
    for i in range(NC):
        for j in range(i, NC):
            ei, ej = np.eye(NC)[i], np.eye(NC)[j]
            v = (Q(ei + ej) - Q(ei) - Q(ej)) / 2 if i != j else Q(ei)
            A[:, i, j] = A[:, j, i] = v
    c0 = np.zeros(NC)
    Ct = beta(c0, g0, 1.0)[0]; C11 = beta(c0, np.array([1.0, 0]), 0.0)[0]; C22 = beta(c0, np.array([0, 1.0]), 0.0)[0]
    C12 = beta(c0, np.array([1.0, 1.0]), 0.0)[0] - C11 - C22
    Bt = np.zeros((NC, NC)); B1 = Bt.copy(); B2 = Bt.copy()
    for i in range(NC):
        e = np.eye(NC)[i]
        Bt[:, i] = beta(e, g0, 1.0)[0] - Ct - A[:, i, i]
        B1[:, i] = beta(e, np.array([1.0, 0]), 0.0)[0] - C11 - A[:, i, i]
        B2[:, i] = beta(e, np.array([0, 1.0]), 0.0)[0] - C22 - A[:, i, i]
        for b in (beta(e, np.array([1.0, 1.0]), 0.0), beta(e, g0, 1.0)): worst = max(worst, b[1])
    np.savez(_F, A=A, Bt=Bt, B1=B1, B2=B2, C11=C11, C12=C12, C22=C22, Ct=Ct, worst=worst)
if not os.path.exists(_F): _build()
_d = np.load(_F); A, Bt, B1, B2, C11, C12, C22, Ct = (_d[k] for k in ('A', 'Bt', 'B1', 'B2', 'C11', 'C12', 'C22', 'Ct'))
BG = np.array([41 / 6 + 1 / 6 + 1 / 3, -19 / 6 + 1 / 6, -7.0])
def rhs(t, z):
    g1, g2, g3, yt = z[:4]; c = z[4:]
    k = 1 / (16 * np.pi**2)
    dc = (np.einsum('ijk,j,k->i', A, c, c) + (yt**2 * Bt + g1**2 * B1 + g2**2 * B2) @ c + C11 * g1**4 + C12 * g1**2 * g2**2 + C22 * g2**4 + Ct * yt**4)
    dyt = k * yt * (4.5 * yt**2 - 8 * g3**2 - 2.25 * g2**2 - 17 / 12 * g1**2)
    return np.concatenate([k * BG * np.array([g1, g2, g3])**3, [dyt], dc])
def run(c0, t0, t1=np.log(1e19), g=(0.36, 0.64, 1.05), yt=0.88, cmax=4 * np.pi, scales=None):
    z0 = np.concatenate([g, [yt], c0])
    ev = lambda t, z: cmax - np.abs(z[4:]).max(); ev.terminal = True
    return solve_ivp(rhs, (t0, t1), z0, events=ev, rtol=1e-8, atol=1e-10, t_eval=scales)
