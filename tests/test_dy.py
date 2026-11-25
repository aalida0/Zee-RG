import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from zee import dy
import pytest

def st(m, be, bm, bt, d=0.0): return dy.screen_state(m, d, be, bm, bt)['status']

def test_cross_section_table_endpoints():
    assert dy.sigma_dy(100., 0.) == pytest.approx(96.51); assert dy.sigma_dy(150., 1.) == pytest.approx(63.34)
    assert dy.sigma_dy(1000., 1.) == 0.0
    for a, b in zip([100, 120, 175, 260, 333, 480], [120, 175, 260, 333, 480, 499]): assert dy.sigma_dy(a, .3) > dy.sigma_dy(b, .3)

def test_published_pure_muon_edges():          # Das et al.: B_mu = 1 excluded 93.5 - 450 GeV
    assert st(95, 0, 1, 0) == 'FAIL' and st(449, 0, 1, 0) == 'FAIL' and st(455, 0, 1, 0) == 'PASS' and st(90, 0, 1, 0) == 'PASS'

def test_published_pure_electron_upper_edge():  # B_e = 1 excluded up to 425 GeV
    assert st(424, 1, 0, 0) == 'FAIL' and st(430, 1, 0, 0) == 'PASS'

def test_published_mixed_benchmark():          # (B_e, B_mu, B_tau) = (1/4, 1/4, 1/2): excluded 125-185, allowed below and above
    assert all(st(m, .25, .25, .5) == 'FAIL' for m in (126, 150, 184))
    assert st(110, .25, .25, .5) == 'PASS' and st(190, .25, .25, .5) == 'PASS'
    assert st(124, .25, .25, .5) in ('PASS', 'FAIL')    # edge within the interpolation, not asserted

def test_published_mu_tau_half_half():         # about 120 - 255 GeV (rough in the paper)
    assert all(st(m, 0, .5, .5) == 'FAIL' for m in (125, 200, 250)) and st(262, 0, .5, .5) == 'PASS'

def test_tau_window():
    assert st(200, 0, .05, .95) == 'FAIL' and st(400, 0, .05, .95) == 'PASS' and st(200, 0.0, .05, .6) == 'PASS'

def test_old_benchmarks_fail():
    assert st(136.9, 1.4e-7, .988, .0116, d=0.0215) == 'FAIL' and st(146.6, 1.8e-7, .9868, .0132, d=0.01) == 'FAIL'

def test_unresolved_for_large_doublet_fraction():
    assert st(300, 0.0, .5, .5, d=0.5) == 'UNRESOLVED'
    r = dy.decide([136.9, 758.], [.02, .98], [0, 0], [.988, .001], [.01, .01]); assert r['status'] == 'FAIL'
    r = dy.decide([600., 758.], [.02, .98], [0, 0], [.9, .001], [.1, .01]); assert r['status'] == 'PASS'
