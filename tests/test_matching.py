"""One-loop effective-potential matching (zee/matching.py): identity without loops, analytic top-only shift of lambda_1, Goldstone zero modes and residual of the full solution."""
import os, sys, json, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, 'gate1'), os.path.join(ROOT, 'gate2')]
os.environ.setdefault('ZEE_INPUTS', 'modern')
import bench
from zee import matching as M

_inp = json.load(open(os.path.join(ROOT, 'results', 'benchmarks_input.json')))
_sc, _c = bench.coup(_inp['BM1']['x'], _inp['BM1']['eta']); _r = -_sc['tb']


def test_five_point_derivatives_of_the_top_loop():
    mu0, yt, V = 600.0, 0.9, M.V; fun = lambda ph: -12*M.K16*M.f_(M.top_mass2(ph, yt, -0.4), 2*np.log(mu0), 1e-6, 1.5)
    phi0 = np.zeros(10); phi0[2] = V; g, H = M.derivs(fun, phi0); x = yt**2*V**2/2; lnx = np.log(x/mu0**2)
    assert abs(H[2, 2] - (-12*M.K16*(2*lnx*yt**4*V**2 + (2*x*lnx - 2*x)*yt**2))) < 1e-6*abs(H[2, 2])
    assert abs(g[2] - (-12*M.K16*(2*x*lnx - 2*x)*yt**2*V)) < 1e-6*abs(g[2])


def test_no_loop_matching_is_the_identity_and_top_only_matches_the_analytic_shift():
    a = M.match(_c, _sc, 600.0, (0.36, 0.64, 1.05), 0.9, _r, no_loop=True)
    assert np.abs(a['delta']).max() < 1e-12
    b = M.match(_c, _sc, 600.0, (0.36, 0.64, 1.05), 0.9, _r, parts='T'); V = M.V; x = 0.9**2*V**2/2; lnx = np.log(x/600.0**2)
    d2 = -12*M.K16*(2*lnx*0.9**4*V**2 + (2*x*lnx - 2*x)*0.9**2); d1 = -12*M.K16*(2*x*lnx - 2*x)*0.9**2*V
    assert abs(b['delta'][0] - (d1/V - d2)/V**2) < 1e-8 and b['resid'] < 1e-8


def test_full_solution_keeps_goldstones_massless_and_is_regulator_insensitive():
    mu0 = 612.0; g = (0.36, 0.64, 1.05); out = {}
    for kap in (0.25, 1.0):
        out[kap] = M.match(_c, _sc, mu0, g, 0.88, _r, kappa=kap)
    b = out[1.0]; assert b['ok'] and b['resid'] < 1e-8
    P, c = M._unpack(np.array([b['P'][k] for k in ('m11', 'm22', 'm12', 'mh2', 'mu')] + [b['c'][0], b['c'][3], b['c'][4], b['c'][5]]), _c); T = M.mass_tensors(c, P)
    phi0 = np.zeros(10); phi0[2] = M.V
    g1, H1 = M.derivs(lambda ph: M.V1(ph, T, mu0, g[0], g[1], 0.88, _r*0.88, M.MH**2), phi0)
    Htot = M.mass2_at(phi0, T) + H1; w = np.sort(np.abs(np.linalg.eigvalsh(Htot)))[:3]
    assert w.max() < 1e-3*M.V**2*1e-3                       # three Goldstone zero modes (gauge invariance of the effective potential)
    assert np.abs(out[0.25]['delta'] - b['delta']).max() < 0.01          # regulator X0 -> m_h^2/4 changes lambda_{1,4,5,6} by < 0.01
