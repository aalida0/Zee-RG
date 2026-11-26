"""Z2-basis (lambda_1..7, m12^2, tan beta) <-> Higgs-basis bookkeeping for the GM2Calc comparison away from alignment.
Conventions of GM2Calc 2 (V = m11^2|P1|^2 + m22^2|P2|^2 - m12^2 (P1^+ P2 + h.c.) + l1/2 |P1|^4 + l2/2 |P2|^4 + l3 |P1|^2|P2|^2 + l4 |P1^+P2|^2
+ [l5/2 (P1^+P2)^2 + l6 |P1|^2 P1^+P2 + l7 |P2|^2 P1^+P2 + h.c.]) with <P_i^0> = v_i/sqrt2.  Higgs basis H1 = cb P1 + sb P2, H2 = -sb P1 + cb P2.
mass_matrices() reproduces the GM2Calc formulas (without the gauge parts, which only affect the Goldstones)."""
import numpy as np
from scipy.optimize import fsolve
V = 246.22


def tadpole_masses(l, tb, m122, v=V):
    l1, l2, l3, l4, l5, l6, l7 = l; b = np.arctan(tb); v1, v2 = v*np.cos(b), v*np.sin(b)
    m112 = (m122*v2 - l1*v1**3/2 - l7*v2**3/2 - 3*l6*v2*v1**2/2 - (l3 + l4 + l5)*v1*v2**2/2) / v1
    m222 = (m122*v1 - l6*v1**3/2 - l2*v2**3/2 - (l3 + l4 + l5)*v2*v1**2/2 - 3*l7*v1*v2**2/2) / v2
    return m112, m222


def mass_matrices(l, tb, m122, v=V):
    l1, l2, l3, l4, l5, l6, l7 = l; b = np.arctan(tb); v1, v2 = v*np.cos(b), v*np.sin(b)
    m112, m222 = tadpole_masses(l, tb, m122, v)
    hh = np.array([[m112 + 3*l6*v1*v2 + 1.5*l1*v1**2 + 0.5*(l3 + l4 + l5)*v2**2,
                    -m122 + (l3 + l4 + l5)*v1*v2 + 1.5*l6*v1**2 + 1.5*l7*v2**2],
                   [0, m222 + 3*l7*v1*v2 + 0.5*(l3 + l4 + l5)*v1**2 + 1.5*l2*v2**2]])
    hh[1, 0] = hh[0, 1]
    Ah = np.array([[m112 + l6*v1*v2 + 0.5*l1*v1**2 + 0.5*(l3 + l4 - l5)*v2**2, -m122 + l5*v1*v2 + 0.5*l6*v1**2 + 0.5*l7*v2**2],
                   [0, m222 + l7*v1*v2 + 0.5*(l3 + l4 - l5)*v1**2 + 0.5*l2*v2**2]]); Ah[1, 0] = Ah[0, 1]
    Hm = np.array([[m112 + l6*v1*v2 + 0.5*l1*v1**2 + 0.5*l3*v2**2, -m122 + 0.5*(l4 + l5)*v1*v2 + 0.5*l6*v1**2 + 0.5*l7*v2**2],
                   [0, m222 + l7*v1*v2 + 0.5*l3*v1**2 + 0.5*l2*v2**2]]); Hm[1, 0] = Hm[0, 1]
    return hh, Ah, Hm


def physical(l, tb, m122, v=V):
    """masses and s_{b-a}, c_{b-a} (GM2Calc: h = -s_a P1 + c_a P2 in the (rho1, rho2) basis, alpha in [-pi/2, pi/2])."""
    hh, Ah, Hm = mass_matrices(l, tb, m122, v); b = np.arctan(tb)
    w, U = np.linalg.eigh(hh)                       # w[0] = lighter
    # the Goldstone-free CP-odd / charged states: eigenvalue not equal to zero
    wa = np.linalg.eigvalsh(Ah); wc = np.linalg.eigvalsh(Hm)
    mA2 = max(wa, key=abs); mC2 = max(wc, key=abs)
    # h eigenvector (u1,u2) = (-s_a, c_a) up to sign; choose c_a >= 0
    u = U[:, 0]; u = u if u[1] >= 0 else -u
    alpha = np.arctan2(-u[0], u[1])
    return dict(mh=np.sqrt(w[0]), mH=np.sqrt(w[1]), mA=np.sqrt(abs(mA2)), mHp=np.sqrt(abs(mC2)), sba=np.sin(b - alpha), cba=np.cos(b - alpha))


def invert(mHp, mH, mA, sba, l6, l7, tb, m122, mh=125.09, cba_sign=+1.0, v=V):
    """solve lambda_1..5 so that the spectrum is reproduced (linear in lambda; fsolve from zero)."""
    def res(x):
        p = physical(list(x) + [l6, l7], tb, m122, v)
        return [p['mh'] - mh, p['mH'] - mH, p['mA'] - mA, p['mHp'] - mHp, p['cba'] - cba_sign*np.sqrt(1 - sba**2)]   # GM2Calc: c_{b-a} >= 0
    x = fsolve(res, np.array([0.3, 0.3, 0.5, 0.0, 0.0]), xtol=1e-13)
    return x


def higgs_basis(l, tb):
    """Z1..Z7 of V = Y1|H1|^2 + Y2|H2|^2 + ... fitted by evaluating the quartic form at random field points (no reliance on closed formulas)."""
    rng = np.random.default_rng(1); b = np.arctan(tb); c, s = np.cos(b), np.sin(b); l1, l2, l3, l4, l5, l6, l7 = l
    def quart_z2(P1, P2):
        a11, a22, a12 = np.vdot(P1, P1).real, np.vdot(P2, P2).real, np.vdot(P1, P2)
        return (0.5*l1*a11**2 + 0.5*l2*a22**2 + l3*a11*a22 + l4*abs(a12)**2 + (0.5*l5*a12**2).real*2 + 2*((l6*a11 + l7*a22)*a12).real)
    def basis(H1, H2):
        a11, a22, a12 = np.vdot(H1, H1).real, np.vdot(H2, H2).real, np.vdot(H1, H2)
        return np.array([0.5*a11**2, 0.5*a22**2, a11*a22, abs(a12)**2, (a12**2).real, 2*(a11*a12).real, 2*(a22*a12).real])
    rows, rhs = [], []
    for _ in range(60):
        H1 = rng.normal(size=2) + 1j*rng.normal(size=2); H2 = rng.normal(size=2) + 1j*rng.normal(size=2)
        P1 = c*H1 - s*H2; P2 = s*H1 + c*H2             # inverse of H1 = c P1 + s P2, H2 = -s P1 + c P2
        rows.append(basis(H1, H2)); rhs.append(quart_z2(P1, P2))
    Z = np.linalg.lstsq(np.array(rows), np.array(rhs), rcond=None)[0]
    return Z       # Z1..Z7 (Z5 = real part: all couplings real)
