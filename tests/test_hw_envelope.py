import os, sys, json, numpy as np
_R = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path[:0] = [_R, _R + '/gate2']
os.environ.setdefault('ZEE_INPUTS', 'modern')
import hw_envelope as HE

def test_rates_exactly_quadratic_and_min_at_clipped_vertex():
    d = json.load(open(os.path.join(_R, 'results', 'benchmarks_input.json')))['BM1']
    r = HE.envelope(np.array(d['x'], float), np.array(d['eta'], float))
    for k, v in r.items():
        assert v['quad_check'] < 1e-10                      # |A + sB|^2 is exactly quadratic in s
        assert -1 <= v['s_min'] <= 1 and v['min'] <= min(v['off'], v['nominal'], v['reversed']) * (1 + 1e-12) + 1e-300
        assert v['max'] >= max(v['nominal'], v['reversed']) * (1 - 1e-12)
