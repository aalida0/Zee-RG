"""rgfast with a second-doublet top Yukawa rho = r*yt (r = -tan(beta) when Phi2 has no quark couplings in the generic basis).
beta_c = A c c + (yt^2 Ba + rho^2 Bb + yt*rho Bc + g1^2 B1 + g2^2 B2) c + C(g) + sum_m yt^(4-m) rho^m D_m ; r is RG-invariant at one loop for a single top."""
import os, numpy as np
from scipy.integrate import solve_ivp
from . import rge, rgeq, rgfast as F0
_F = os.path.join(os.path.dirname(__file__), '_rgfast2.npz')
NC = 11
def _build():
    Ts = [rgeq.tensor_from(np.eye(NC)[i]) for i in range(NC)]
    M = np.array([t.ravel() for t in Ts]).T; pinv = np.linalg.pinv(M)
    Z = np.zeros((3, 3), complex)
    def beta(c, yt, rho):
        T = sum(ci * Ti for ci, Ti in zip(c, Ts)); Y = rge.full_Y(Z, Z, Z, yt, rho)
        return pinv @ rgeq.quartic_beta(T, Y, np.array([0., 0., 1.0])).ravel()
    pts = [(1, 0), (0, 1), (1, 1), (1, -1), (1, 2), (2, 1), (1, .5)]
    mono = lambda y, r: [y**4, y**3*r, y**2*r**2, y*r**3, r**4]
    Cm = np.array([mono(*p) for p in pts]); Cv = np.array([beta(np.zeros(NC), *p) for p in pts])
    D = np.linalg.lstsq(Cm, Cv, rcond=None)[0]                  # (5,NC)
    res = np.abs(Cm @ D - Cv).max()
    A = F0.A; Bs = []
    pts2 = [(1, 0), (0, 1), (1, 1), (1, -1)]; Bm = np.array([[p[0]**2, p[1]**2, p[0]*p[1]] for p in pts2])
    Bt = np.zeros((3, NC, NC))
    for i in range(NC):
        e = np.eye(NC)[i]
        v = np.array([beta(e, *p) - beta(np.zeros(NC), *p) - 0 for p in pts2])   # includes A_ii (y-independent part cancels below)
        base = beta(e, 0.0, 0.0) - beta(np.zeros(NC), 0.0, 0.0)
        co = np.linalg.lstsq(Bm, v - base, rcond=None)[0]; Bt[:, :, i] = co[:, :]
    np.savez(_F, D=D, Bt=Bt, res=res)
    return res
if not os.path.exists(_F): _build()
_d = np.load(_F); D, Bt = _d['D'], _d['Bt']
A, B1, B2, C11, C12, C22 = F0.A, F0.B1, F0.B2, F0.C11, F0.C12, F0.C22
BG = F0.BG
def rhs_r(r):
    def rhs(t, z):
        g1, g2, g3, yt = z[:4]; c = z[4:]; k = 1/(16*np.pi**2); rho = r*yt
        Bm = yt**2*Bt[0] + rho**2*Bt[1] + yt*rho*Bt[2]
        Cy = sum(m*Dm for m, Dm in zip([yt**4, yt**3*rho, yt**2*rho**2, yt*rho**3, rho**4], D))
        dc = (np.einsum('ijk,j,k->i', A, c, c) + (Bm + g1**2*B1 + g2**2*B2) @ c + C11*g1**4 + C12*g1**2*g2**2 + C22*g2**4 + Cy)
        dyt = k*yt*(4.5*yt**2*(1 + r**2) - 8*g3**2 - 2.25*g2**2 - 17/12*g1**2)
        return np.concatenate([k*BG*np.array([g1, g2, g3])**3, [dyt], dc])
    return rhs
def run(c0, t0, t1=np.log(1e19), g=(0.36, 0.64, 1.05), yt=0.88, cmax=4*np.pi, scales=None, r=0.0):
    z0 = np.concatenate([g, [yt], c0]); ev = lambda t, z: cmax - np.abs(z[4:]).max(); ev.terminal = True
    return solve_ivp(rhs_r(r), (t0, t1), z0, events=ev, rtol=1e-8, atol=1e-10, t_eval=scales)
