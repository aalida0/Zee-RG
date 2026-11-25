"""Two-body scalar decays that HiggsTools has no channel for, and light-quark charged-scalar decays.
All masses in GeV.  Higgs-basis potential (Haber): the CP-even sector (rho1, rho2) with H1^0 = (v + rho1)/sqrt2,
H2^0 = rho2/sqrt2 has the cubic terms
    V3 = Z1 v rho1^3/2 + Z345 v rho1 rho2^2/2 + 3 Z6 v rho1^2 rho2/2 + Z7 v rho2^3/2,   Z345 = Z3+Z4+Z5,
(Z1..Z7 = lam1, lam2, lam3, ... of the code, Z7 = eta[1]).  The mass eigenstates are the eigenvectors of
M^2 = v^2 [[Z1, Z6], [Z6, mA^2/v^2 + Z5]]; this reproduces m_h = 125.5 and m_H of the scalar sector (tests)."""
import numpy as np
from . import V_EW as V

GF = 1/(np.sqrt(2)*V**2)
MQ = dict(c=1.27, s=0.093, b=4.18, u=0.0022, d=0.0047)
VCKM = dict(cs=0.973, cb=0.0411, us=0.225, ud=0.974, ub=0.0038, cd=0.221)


def lam_k(a, b, c):
    return max(a*a + b*b + c*c - 2*a*b - 2*a*c - 2*b*c, 0.0)


def cp_even_basis(sc):
    """Return (mh, mH, uh, uH): eigenvectors in the (rho1, rho2) basis (sign fixed so that uh[0] > 0)."""
    M = np.array([[sc['lam1']*V**2, sc['lam6']*V**2], [sc['lam6']*V**2, sc['mA']**2 + sc['lam5']*V**2]])
    w, U = np.linalg.eigh(M)
    uh, uH = U[:, 0].copy(), U[:, 1].copy()
    if uh[0] < 0: uh = -uh
    if uH[1] > 0: uH = -uH
    return np.sqrt(w[0]), np.sqrt(w[1]), uh, uH


def g_Hhh(sc, Z7=0.0):
    """g = d^3 V / dH dh dh (GeV), L contains -(g/2) H h h."""
    Z1, Z6 = sc['lam1'], sc['lam6']; Z345 = sc['lam3'] + sc['lam4'] + sc['lam5']
    mh, mH, uh, uH = cp_even_basis(sc)
    def V3(r1, r2): return V*(Z1*r1**3/2 + Z345*r1*r2**2/2 + 1.5*Z6*r1**2*r2 + Z7*r2**3/2)
    # exact: V3 is a cubic polynomial in (h, H); extract the h^2 H coefficient by finite differences of a polynomial
    f = lambda h, H: V3(h*uh[0] + H*uH[0], h*uh[1] + H*uH[1])
    # odd-in-H part of a cubic: c_h2H h^2 H + c_H3 H^3 (+ h H^2 even); evaluate and remove the H^3 term
    c_H3 = (f(0, 1) - f(0, -1))/2
    c_h2H = (f(1, 1) + f(-1, 1) - f(1, -1) - f(-1, -1))/4 - c_H3
    return 2*c_h2H


def gamma_Hhh(sc, mhH, mh=125.5, Z7=0.0):
    g = g_Hhh(sc, Z7); mH = mhH
    if mH <= 2*mh: return 0.0
    return g**2/(32*np.pi*mH)*np.sqrt(1 - 4*mh**2/mH**2)


def g_h_h1_h2(sc, trilin_raw):
    """coupling v*[sc(dhh - dHH) + dHh (c^2 - s^2)] of h h1+ h2- (code convention), GeV."""
    c, s = np.cos(sc['phi']), np.sin(sc['phi']); dHH, dhh, dHh = trilin_raw
    return V*(s*c*(dhh - dHH) + dHh*(c*c - s*s))


def gamma_h2_h1_h(sc, trilin_raw, mh=125.5):
    m2, m1 = sc['m2'], sc['m1']
    if m2 <= m1 + mh: return 0.0
    g = g_h_h1_h2(sc, trilin_raw)
    return g**2*np.sqrt(lam_k(m2**2, m1**2, mh**2))/(16*np.pi*m2**3)


def gamma_hp_quarks(m, A):
    """Gamma(h+ -> u d') summed over c s, c b, u s, u d, u b, c d (type-I-like quark sector: coupling A*(m_u, m_d)),
    for masses below the top threshold the t b mode is separate."""
    tot = {}
    for key, (mu, md) in dict(cs=(MQ['c'], MQ['s']), cb=(MQ['c'], MQ['b']), us=(MQ['u'], MQ['s']),
                              ud=(MQ['u'], MQ['d']), ub=(MQ['u'], MQ['b']), cd=(MQ['c'], MQ['d'])).items():
        if m <= mu + md: tot[key] = 0.0; continue
        ph = (1 - (mu + md)**2/m**2)*(1 - (mu - md)**2/m**2)
        tot[key] = 3*GF*m*A**2*VCKM[key]**2*(mu**2 + md**2)/(4*np.sqrt(2)*np.pi)*np.sqrt(max(ph, 0))*(1 - 0.0)
    return tot
