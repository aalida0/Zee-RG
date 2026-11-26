"""PyR@TE 3 generated RGEs (complete SM fermion content): the one-loop functions reproduce the production solver's first-failure scale and the two-loop
functions do not lower it (validation/two_loop_check.py; full results in results/two_loop_check.json)."""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'validation'))
import two_loop_check as T, validate_finalist as VF, cert as CE


def test_pyrate_one_loop_matches_production_and_two_loop_is_not_lower():
    mod = T.load_rges(os.path.join(ROOT, 'validation', 'pyrate_RGEs_2loop_full.py'))
    x, eta = VF._load('BM1')
    m0, o0 = CE.cert(x, eta, ('default', 'qscale_1.1'))
    m1, o1 = T.cert(mod, 1, x, eta, ('default', 'qscale_1.1'))
    m2, o2 = T.cert(mod, 2, x, eta, ('default', 'qscale_1.1'))
    assert abs(m1 - m0) < 0.03 and abs(o1['default']['log10'] - o0['default']['log10']) < 0.03
    assert m2 >= m1 - 0.02
