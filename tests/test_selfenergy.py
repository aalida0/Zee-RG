"""Momentum-dependent one-loop self-energies (zee/selfenergy.py) and the full matching (zee/matching_full.py).
(i) loop functions against numerical quadrature above and below threshold, B_SV(s=0)=0 in Landau gauge, massless-vector limits;
(ii) Pi(0)/(16 pi^2) = Hessian of the one-loop effective potential at a generic non-tachyonic point, for the scalar, top and gauge parts separately;
(iii) the explicit lnQ dependence of Pi has the s-slope -gamma_ij (Landau-gauge field anomalous dimensions from Yukawa and gauge couplings), which tests the s-dependent fermion and
vector pieces including the SV bubble; (iv) the matching solves its 9 equations, is the identity when Pi is zero, and its quartic shifts are finite."""
import os, sys, json, numpy as np
from scipy import integrate
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, 'gate1'), os.path.join(ROOT, 'gate2')]
os.environ.setdefault('ZEE_INPUTS', 'modern')
from zee import matching as M, selfenergy as SE, matching_full as MF, rgeq

_c = np.array([0.12, 0.05, 0.3, 0.2, 0.1, 0.03, 0.04, 0.2, 0.1, 0.15, 0.1])
_P = dict(m11=250.**2, m22=300.**2, m12=1e4, mh2=280.**2, mu=30.)
_T = M.mass_tensors(_c, _P)
_phi = np.zeros(10); _phi[[2, 6, 0, 8, 3, 4]] = [150, 40, 20, 15, 10, -12]
_g = (0.36, 0.64); _yt = 0.9; _rt = 0.18; _mu0 = 300.; K = 1/(16*np.pi**2)


def test_bubble_function_against_quadrature():
    q2 = 300.**2
    def num(x, y, s):
        f = lambda t: np.log(abs(t*x + (1 - t)*y - t*(1 - t)*s)/q2); pts = []
        disc = (x - y - s)**2 - 4*s*y
        if disc > 0:
            for sg in (1, -1):
                r = (-(x - y - s) + sg*np.sqrt(disc))/(2*s)
                if 0 < r < 1: pts.append(r)
        return -integrate.quad(f, 0, 1, points=pts or None, limit=400)[0]
    for x, y, s in [(1e5, 2e5, 5e4), (1e5, 2e5, 1e6), (0, 1e4, 5e4), (0, 0, 1e5), (4e5, 4e5, 4.2e5)]:
        assert abs(SE.B_(x, y, s, q2) - num(x, y, s)) < 1e-9


def test_landau_gauge_SV_function_vanishes_at_zero_momentum_and_massless_limit_is_continuous():
    q2 = 300.**2
    for x, y in [(1e5, 6e3), (0.0, 8e3), (1e4, 1e4)]: assert abs(SE.B_SV(x, y, 0.0, q2)) < 1e-8
    assert abs(SE.B_SV(1e5, 1e-3, 2e4, q2) - SE.B_SV(1e5, 0.0, 2e4, q2)) < 1e-6*abs(SE.B_SV(1e5, 0.0, 2e4, q2))
    assert abs(SE.B_VV(1e-3, 6.4e3, 2e4, q2) - SE.B_VV(0.0, 6.4e3, 2e4, q2)) < 1e-4


def test_zero_momentum_self_energy_is_the_effective_potential_hessian():
    se = SE.Pi(_phi, _T, _g, _yt, _rt, _mu0)
    for parts, vp in (('S', 'S'), ('F', 'T'), ('V', 'V')):
        _, H = M.derivs(lambda ph: M.V1(ph, _T, _mu0, _g[0], _g[1], _yt, _rt, 1e-6, vp), _phi, h=0.5)
        assert np.abs(se.at(0.0, parts)*K - H).max() < 1e-6*max(1.0, np.abs(H).max())


def test_lnQ_slope_in_s_is_minus_the_landau_gauge_anomalous_dimension():
    th = rgeq._gens(*_g); gamV = -3*sum(t.T@t for t in th); _, y = SE._fermion_sector(_phi, _yt, _rt)
    gamF = 3*np.einsum('IJi,IJj->ij', y, y.conj()).real
    def D(s, parts): return SE.Pi(_phi, _T, _g, _yt, _rt, _mu0*np.exp(0.5)).at(s, parts) - SE.Pi(_phi, _T, _g, _yt, _rt, _mu0*np.exp(-0.5)).at(s, parts)
    for parts, gam in (('F', gamF), ('V', gamV)):
        slope = (D(9e4, parts) - D(3e4, parts))/(6e4)/2                # Delta ln Q^2 = 2
        assert np.abs(slope + gam).max() < 1e-9


def test_full_matching_solves_its_equations_and_is_finite():
    inp = json.load(open(os.path.join(ROOT, 'results', 'benchmarks_input.json'))); import bench
    sc, c = bench.coup(inp['BM1']['x'], inp['BM1']['eta']); r = -sc['tb']; g = (0.36, 0.64, 1.05)
    for pres in ('min', 'hpole-sym'):
        m = MF.match_full(c, sc, 612.0, g, 0.88, r, prescription=pres)
        assert m['ok'] and m['resid'] < 1e-8 and np.abs(m['delta']).max() < 0.2
    PiE, Mt, info = MF.pi_eff(c, sc, 612.0, g, 0.88, 0.88*r, 'min'); assert np.allclose(PiE, PiE.T)
