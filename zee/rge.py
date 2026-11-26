"""One-loop Yukawa RGEs of the general Zee model (Machacek-Vaughn, Weyl/real-scalar form).
Fields: L_i (nu,e), e^c_i, third-gen quarks (t_L,b_L,t^c; 3 colours), scalars H1,H2 (doublets), eta+.
Couplings run: Y1 (H1 lepton), Y2 (H2 lepton), f (antisym), y_t (H1 only), g1,g2,g3."""
import numpy as np
from scipy.integrate import solve_ivp
# Weyl index map
NU, E, EC = 0, 3, 6; TL, BL, TC = 9, 12, 15; NF = 18
# real scalars: complex fields H1+,H10,H2+,H20,eta+ -> index 2k (Re), 2k+1 (Im)
SC = {"H1p": 0, "H10": 1, "H2p": 2, "H20": 3, "eta": 4}
def _tensor(Y1, Y2, f, yt):
    T = np.zeros((5, NF, NF), complex); Tc = np.zeros((5, NF, NF), complex)   # coeff of phi, phi*
    def add(M, k, a, b, c):
        M[k, a, b] += c; M[k, b, a] += c
    for k, Yl in ((0, Y1), (2, Y2)):                       # -L = Y_ij e^c_j (H0* e_i + H- nu_i)
        for i in range(3):
            for j in range(3):
                add(Tc, k + 1, EC + j, E + i, Yl[i, j]); add(Tc, k, EC + j, NU + i, Yl[i, j])
    for c in range(3):                                      # top: t^c (t_L H10 - b_L H1+), colours as 3 copies
        pass
    for i in range(3):
        for j in range(3):
            add(T, 4, NU + i, E + j, f[i, j]); add(T, 4, E + i, NU + j, -f[i, j])
    return T, Tc
def _real(T, Tc):
    Y = np.zeros((10, NF, NF), complex)
    for k in range(5):
        Y[2 * k] = (T[k] + Tc[k]) / np.sqrt(2); Y[2 * k + 1] = 1j * (T[k] - Tc[k]) / np.sqrt(2)
    return Y
def _top(yt):
    # three colours: use summed contribution analytically via extra Weyl states
    return yt
def beta_Y(Y, g, C):
    """Y: (S,NF,NF). C: diag of sum_G g_G^2 C2_G per Weyl fermion."""
    Y2F = sum(Y[b].conj().T @ Y[b] for b in range(len(Y)))
    Y2Ft = sum(Y[b] @ Y[b].conj().T for b in range(len(Y)))
    Y2S = np.array([[0.5 * np.trace(Y[a].conj().T @ Y[b] + Y[b].conj().T @ Y[a]) for b in range(len(Y))] for a in range(len(Y))])
    out = []
    for a in range(len(Y)):
        t = 0.5 * (Y2Ft @ Y[a] + Y[a] @ Y2F)
        t = t + sum(2 * Y[b] @ Y[a].conj().T @ Y[b] + Y[b] * Y2S[a, b] for b in range(len(Y)))
        t = t - 3 * (C[:, None] + C[None, :]) * Y[a]
        out.append(t)
    return np.array(out) / (16 * np.pi**2)
def _C(g1, g2, g3):
    C = np.zeros(NF); Yh = {NU: -.5, E: -.5, EC: 1, TL: 1 / 6, BL: 1 / 6, TC: -2 / 3}
    for base, y in Yh.items():
        for i in range(3 if base in (NU, E, EC, TL, BL, TC) else 0):
            pass
    for i in range(3):
        for base, y, c2 in ((NU, -.5, .75), (E, -.5, .75), (EC, 1, 0)):
            C[base + i] = g1**2 * y**2 + g2**2 * c2
    for base, y, c2, c3 in ((TL, 1 / 6, .75, 4 / 3), (BL, 1 / 6, .75, 4 / 3), (TC, -2 / 3, 0, 4 / 3)):
        for i in range(3):
            C[base + i] = g1**2 * y**2 + g2**2 * c2 + g3**2 * c3
    return C
def _top_tensor(yt, rho=0.0):
    Tt = np.zeros((5, NF, NF), complex)
    if rho:                                                 # H2 top coupling (Phi2-free quark sector in the generic basis): rho = -yt tan(beta)
        for c in range(3):
            for (a, b, k, s_) in ((TC + c, TL + c, 3, 1.0), (TC + c, BL + c, 2, -1.0)):
                Tt[k, a, b] += s_ * rho; Tt[k, b, a] += s_ * rho
    for c in range(3):                                      # t^c_c t_L,c H10 - t^c_c b_L,c H1+
        for (a, b, k, s) in ((TC + c, TL + c, 1, 1.0), (TC + c, BL + c, 0, -1.0)):
            Tt[k, a, b] += s * yt; Tt[k, b, a] += s * yt
    return Tt
def full_Y(Y1, Y2, f, yt, rho=0.0):
    T, Tc = _tensor(Y1, Y2, f, yt); T = T + _top_tensor(yt, rho)
    return _real(T, Tc)
def pack(Y1, Y2, f, yt, g):
    return np.concatenate([Y1.ravel(), Y2.ravel(), f.ravel(), [yt], g]).view(float) if False else np.concatenate([np.concatenate([Y1.ravel(), Y2.ravel(), f.ravel()]).view(float), [yt], g])
def unpack(z):
    c = z[:54].view(complex); return c[:9].reshape(3, 3), c[9:18].reshape(3, 3), c[18:27].reshape(3, 3), z[54], z[55:58]
BG = np.array([41 / 6 + 1 / 6 + 1 / 3, -19 / 6 + 1 / 6, -7.0])        # above the scalar threshold
def rhs(t, z):
    Y1, Y2, f, yt, g = unpack(z)
    Y = full_Y(Y1, Y2, f, yt); b = beta_Y(Y, g, _C(*g))
    # project beta back to couplings
    def get(T, k, a, c):  # coefficient extraction
        return T[k, a, c]
    # invert real->complex: T_k = (Y_re - i Y_im)/sqrt2 + ... ; we need T (phi) and Tc (phi*) parts
    def part(b):
        Tk = np.zeros((5, NF, NF), complex); Tck = np.zeros((5, NF, NF), complex)
        for k in range(5):
            re, im = b[2 * k], b[2 * k + 1]
            Tk[k] = (re - 1j * im) / np.sqrt(2); Tck[k] = (re + 1j * im) / np.sqrt(2)
        return Tk, Tck
    Tk, Tck = part(b)
    dY1 = np.array([[Tck[1][EC + j, E + i] for j in range(3)] for i in range(3)])
    dY2 = np.array([[Tck[3][EC + j, E + i] for j in range(3)] for i in range(3)])
    G = np.array([[Tk[4][NU + i, E + j] for j in range(3)] for i in range(3)])
    df = (G - G.T)/4           # the tensor entry is y^{nu_i e_j} = f_ij - f_ji = 2 f_ij (antisymmetric f): beta_f = antisymmetric part of G / 2 (report v11 fix; was G)
    dyt = Tk[1][TC, TL].real
    dg = g**3 * BG / (16 * np.pi**2)
    return pack(dY1, dY2, df, dyt, dg)
def run(Y1, Y2, f, yt, g, t0, t1, ymax=4 * np.pi):
    z0 = pack(Y1, Y2, f, yt, np.asarray(g, float))
    ev = lambda t, z: ymax - max(np.abs(z[:54]).max(), abs(z[54])); ev.terminal = True
    s = solve_ivp(rhs, (t0, t1), z0, events=ev, rtol=1e-7, atol=1e-12, dense_output=False)
    return s
