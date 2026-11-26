"""SM limit of the one-loop self-energy engine (report v11 / referee plan, independent validation).

Configuration: the second doublet and the singlet are decoupled from h (all portal quartics, the mixing term mu and rho_t vanish), so that Pi_{phi1 phi1}(s) must equal the SM Higgs self-energy in Landau gauge and MSbar.
The scalar and top parts are compared with closed forms derived here independently of selfenergy.py (hand derivation, conventions: pole at s = m^2 + Pi(s), Pi in units of 1/16 pi^2):
  Pi_top(s)  = N_c y^2 / (8 pi^2) [ -2 A(m_t) + (4 m_t^2 - s) B(s; m_t, m_t) ],      y = y_t/sqrt(2) the h t t coupling,
  Pi_scal(s) = (1/16 pi^2) [ 3 lam/2 A(m_h) - 9 lam^2 v^2/2 B(s; m_h, m_h) - 3/2 lam^2 v^2 B(s; 0, 0) ],   lam = m_h^2/v^2 (this model's lambda_1),
with the h^3 coupling 3 lam v, the h G G coupling lam v (three Goldstone fields) and the quartics 3 lam, lam.  The vector part is covered by the Hessian and lnQ-slope tests of test_selfenergy.py."""
import os, sys, json, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, 'gate1'), os.path.join(ROOT, 'gate2')]
os.environ.setdefault('ZEE_INPUTS', 'modern')
from zee import matching as M, selfenergy as SE

K = 1/(16*np.pi**2)


def _Ab(m2, q2): return 0.0 if m2 <= 0 else m2*(np.log(m2/q2) - 1)


def _Bb(s, m1, m2, q2):
    """MSbar bubble, real part, by direct numerical Feynman-parameter integration (independent of selfenergy.B_)"""
    from scipy import integrate
    def f(x):
        a = x*m1 + (1 - x)*m2 - x*(1 - x)*s
        return np.log(max(abs(a), 1e-300)/q2)
    pts = []
    disc = (m1 - m2 - s)**2 - 4*s*m2          # real roots of the argument (above threshold): integrable logarithmic singularities
    if s > 0 and disc > 0:
        for sg in (1, -1):
            x = ((s + m2 - m1) + sg*np.sqrt(disc))/(2*s) if s > 0 else None
            if x is not None and 0 < x < 1: pts.append(x)
    v = integrate.quad(f, 0, 1, points=sorted(pts) or None, limit=400, epsabs=1e-13, epsrel=1e-12)[0]
    return -v


def _setup(Mx=200., mu0=172.5):
    c0 = np.zeros(11); c0[0] = M.MH**2/M.V**2; c0[1] = 0.3
    sc = dict(MHp=Mx, M33=1.1*Mx, mu=0.0, lam1=c0[0], lam3=0, lam4=0, lam5=0, lam6=0)
    P0 = M.tree_params(c0, sc); T0 = M.mass_tensors(c0, P0); phi0 = np.zeros(10); phi0[2] = M.V
    g = (0.3583, 0.6478); yt = 0.9369
    return c0, SE.Pi(phi0, T0, g, yt, 0.0, mu0), yt, mu0**2


def test_scalar_and_top_parts_match_closed_forms():
    c0, eng, yt, q2 = _setup(); lam = c0[0]; v2 = M.V**2; mh2 = lam*v2; mt2 = yt**2*v2/2
    for s in (0.0, 0.5*mh2, mh2, 0.9*4*mt2, 3.0*mt2, 4*mt2*1.5, 1e6, 4e5):
        top = 3*(yt**2/2)/(8*np.pi**2)*(-2*_Ab(mt2, q2) + (4*mt2 - s)*_Bb(s, mt2, mt2, q2))
        scal = K*(1.5*lam*_Ab(mh2, q2) - 4.5*lam**2*v2*_Bb(s, mh2, mh2, q2) - 1.5*lam**2*v2*_Bb(s, 0.0, 0.0, q2))
        assert abs(eng.at(s, 'F')[2, 2]*K - top) < 1e-6*max(1, abs(top)), ('top', s, eng.at(s, 'F')[2, 2]*K, top)
        if s == 0.0: continue                    # B(0; 0, 0) of the massless Goldstones is infrared divergent at s = 0 (not a test of the finite parts)
        assert abs(eng.at(s, 'S')[2, 2]*K - scal) < 1e-6*max(1, abs(scal)), ('scalar', s, eng.at(s, 'S')[2, 2]*K, scal)


def test_sm_limit_matching_shift_is_sm_like_and_independent_of_decoupled_masses():
    """Delta lambda_SM at mu = m_t with decoupled extras: the full matching gives lambda(m_t) ~ 0.128 against the tree value 0.1299 (literature MSbar lambda(m_t) ~ 0.126 with G_F-defined v; our v is the Landau-gauge
    effective-potential minimum held at 246.22 GeV, which accounts for O(1%) of the difference -- this is a sanity check of size and sign, not a precision validation); the result must not depend on the mass of the decoupled states."""
    from zee import matching_full as MF
    out = []
    for Mx in (100., 300.):
        c0 = np.zeros(11); c0[0] = M.MH**2/M.V**2; c0[1] = 0.3
        sc = dict(MHp=Mx, M33=1.1*Mx, mu=0.0, lam1=c0[0], lam3=0, lam4=0, lam5=0, lam6=0)
        m = MF.match_full(c0, sc, 172.5, (0.3583, 0.6478, 1.1666), 0.9369, 0.0, prescription='sym'); assert m['ok']
        out.append(c0[0]/2 + m['delta'][0]/2)
    assert abs(out[0] - out[1]) < 5e-5 and 0.125 < out[0] < 0.130 and out[0] < c0[0]/2
