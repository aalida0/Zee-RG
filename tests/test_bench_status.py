import os, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, 'gate1'), os.path.join(ROOT, 'gate2')]
import bench


def test_trajectory_start_failure_claims_no_scale():
    """report v8 P5: a boundary point that already violates a condition must not return its own mass scale as a 'cutoff'"""
    c = np.zeros(11); c[0] = -0.1; c[1] = 0.3                       # lambda1 < 0: unbounded at the boundary
    out = bench.trajectory(c, np.log(600.0), -0.4, np.array([0.36, 0.64, 1.05, 0.88]))
    assert out['fail'].startswith('start') and out['log10Lambda'] == -1.0


def test_trajectory_good_point_has_valid_scale():
    c = np.array([0.26, 0.26, 0.2, 0.1, 0.05, 0.02, 0.0, 0.1, 0.0, 0.0, 0.1])
    out = bench.trajectory(c, np.log(600.0), -0.4, np.array([0.36, 0.64, 1.05, 0.88]))
    assert not str(out['fail']).startswith('start') and out['log10Lambda'] > 2.78


def test_yukawa_perturbativity_cut():
    """Benchmarks have max Yukawa-type coupling ~0.4; scaling the lepton couplings Y2 up by 1000 violates |y| < sqrt(4 pi) and the cut sees it."""
    import json, numpy as np
    from zee import hgmodel as H
    B = json.load(open(ROOT + '/results/benchmarks_input.json')); x = np.array(B['BM1']['x']); sc = H.evaluate(list(x), 'NO', eta=B['BM1']['eta'])['sc']
    assert H.yukawa_max(x, sc) < 1.0
    x2 = x.copy(); x2[2:11] *= 1000
    assert H.yukawa_max(x2, sc) > H.YUK_MAX


def test_rg_status_numerical_stops_are_not_physical_cutoffs():
    """report v9 5.5: 'integration' and 'nonfinite' (finite last-good reach) are numerical stops = unresolved prefix; start failures fail;
    only physical first failures give 'ok'."""
    import benchmarks as B
    ok = dict(a='unitarity', b='perturbativity', c='BFB', d='None')
    assert B.rg_status_of(ok, 11.6) == (False, False, 'ok')
    assert B.rg_status_of(dict(ok, e='nonfinite'), 11.6)[1:] == (True, 'unresolved (numerical stop: demonstrated prefix only)')
    assert B.rg_status_of(dict(ok, e='integration'), 11.6)[1] is True
    assert B.rg_status_of(dict(ok, e='start:BFB'), -1.0)[:1] == (True,) and B.rg_status_of(dict(ok, e='start:BFB'), -1.0)[2] == 'start-fail'
    assert B.rg_status_of(ok, float('nan'))[1] is True


def test_borderline_bfb_is_unresolved_not_physical():
    import benchmarks as B
    assert B.rg_status_of(dict(a='unitarity', b='BFB?'), 11.0) == (False, True, 'unresolved (borderline BFB margin)')


def test_running_top_yukawa_check():
    """a trajectory whose top Yukawa is non-perturbative at the matching scale is a start failure ('yukawa'), tan(beta) large enough makes rho_t = tan(beta) y_t non-perturbative"""
    c = np.array([0.26, 0.26, 0.2, 0.1, 0.05, 0.02, 0.0, 0.1, 0.0, 0.0, 0.1])
    out = bench.trajectory(c, np.log(600.0), -5.0, np.array([0.36, 0.64, 1.05, 0.88]))      # rho_t = 5 * 0.88 = 4.4 > sqrt(4 pi)
    assert out['fail'] == 'start:yukawa' and out['log10Lambda'] == -1.0


def test_dy_above_500_gev_is_outside_domain():
    """report v10: above the last tabulated mass the Drell-Yan screen claims no exclusion; states with appreciable leptonic BR are UNRESOLVED (fail closed), negligible ones pass and are flagged"""
    from zee import dy
    a = dy.screen_state(700.0, 0.8, 1e-3, 2e-3, 1e-3); assert a['status'] == 'PASS' and a['outside_domain']
    b = dy.screen_state(700.0, 0.8, 0.1, 0.2, 0.1); assert b['status'] == 'UNRESOLVED' and b['outside_domain']
