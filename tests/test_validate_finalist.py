import os, sys
_R = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path[:0] = [_R, _R + '/gate1', _R + '/gate2']
os.environ.setdefault('ZEE_INPUTS', 'modern')
import validate_finalist as VF

def _m(**k): return dict(dict(n_variants=13, complete=True, n_start_fail=0, n_match_failed=0, min_stress=11.5, bfb_margin_min=0.02), **k)

def test_assess_fail_closed_on_incomplete():
    rec = dict(matching={'a': _m(), 'b': _m(complete=False, n_start_fail=12, min_stress=12.4)}, margins=dict(conservative=0.02, max_quartic=0.7))
    a = VF.assess(rec)
    assert a['robust_scale'] is None and a['n_start_failures_total'] == 12 and a['min_scale_over_prescriptions_defined_only'] == 11.5   # a point with undefined variants has no robust scale
    assert a['plan_targets']['margin_above_0.05'] is False and a['plan_targets']['quartics_below_1'] is True and 'robust_scale_above_1e10' not in a['plan_targets']

def test_assess_complete():
    a = VF.assess(dict(matching={'a': _m(), 'b': _m(min_stress=11.9)}))
    assert a['robust_scale'] == 11.5 and a['plan_targets']['robust_scale_above_1e10'] is True

def test_smoke_light_tasks():
    for k in ('margins', 'hw'):
        kind, arg, res, dt = VF.task((k, 'BM1', None)); assert kind == k and res
    assert abs(VF.task(('margins', 'BM1', None))[2]['vred'] - 0.026) < 1e-3
