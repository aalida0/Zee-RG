import os, sys
_R = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path[:0] = [_R, _R + '/gate1', _R + '/gate2']
os.environ.setdefault('ZEE_INPUTS', 'modern')
import validate_finalist as VF, cert as CE

def _m(**k): return dict(dict(n_variants=13, complete=True, n_start_fail=0, n_match_failed=0, n_numerical=0, min_stress=11.5, bfb_margin_min=0.02), **k)
def _y(**k): return dict(dict(n_variants=13, complete=True, n_start_fail=0, n_match_failed=0, n_numerical=0, min=11.6), **k)
def _all(**per):
    """a record with all seven matching prescriptions; per[mode] overrides the entry of that prescription"""
    return {m: _m(**per.get(m, {})) for m in VF.MODES}

def test_assess_fail_closed_on_incomplete():
    mt = _all(**{'full-max': dict(complete=False, n_start_fail=12, min_stress=12.4)})
    a = VF.assess(dict(matching=mt, margins=dict(conservative=0.02, max_quartic=0.7)))
    assert a['robust_scale'] is None and a['n_start_failures_total'] == 12 and a['min_scale_over_prescriptions_defined_only'] == 11.5   # a point with undefined variants has no robust scale
    assert a['all_prescriptions_complete'] is False and a['n_prescriptions_complete'] == 6
    assert a['ep_clean'] is True and a['classification'].startswith('EP-clean')                    # EP-clean is distinct from all-prescription robust
    assert a['plan_targets']['margin_above_0.05'] is False and a['plan_targets']['quartics_below_1'] is True and 'robust_scale_above_1e10' not in a['plan_targets']

def test_assess_complete_matching_only():
    a = VF.assess(dict(matching=_all(**{'full-sym': dict(min_stress=11.3)})))
    assert a['all_prescriptions_complete'] is True and a['robust_scale'] == 11.3 and a['plan_targets']['robust_scale_above_1e10'] is True
    assert a['robust_scale_matching_only'] == 11.3 and a['robust_scale_combined_yukawa'] is None and 'combined' in a['robust_scale_basis'] and a['classification'].startswith('robust')

def test_partial_mode_set_is_never_complete_or_robust():
    a = VF.assess(dict(matching={'ep': _m()}))                         # --modes ep: one prescription of seven
    assert a['all_prescriptions_complete'] is False and a['robust_scale'] is None and a['ep_clean'] is True
    assert len(a['coverage']['matching_missing']) == 6 and a['classification'].startswith('EP-clean')
    a2 = VF.assess(dict(matching={m: _m() for m in VF.MODES[:-1]}))   # one mode missing
    assert a2['all_prescriptions_complete'] is False and a2['robust_scale'] is None and a2['coverage']['matching_missing'] == ['full-hpole-max']

def test_missing_variants_are_incomplete():
    mt = _all(**{'ep': dict(n_variants=12)})                          # a prescription with 12 of 13 variants
    a = VF.assess(dict(matching=mt))
    assert a['all_prescriptions_complete'] is False and a['robust_scale'] is None and a['coverage']['matching_wrong_variant_count'] == ['ep'] and a['ep_clean'] is False

def test_numerical_stop_is_not_a_complete_scale():
    # a finite prefix with an unresolved failure is counted by n_numerical and makes the prescription incomplete (matched_scales.summarize does this; here the record-level consequence)
    a = VF.assess(dict(matching=_all(**{'ep': dict(complete=False, n_numerical=1)})))
    assert a['robust_scale'] is None and a['n_numerical_stops_total'] == 1 and a['ep_clean'] is False

def test_combined_yukawa_coverage_enters_robust_scale():
    y = {m: _y() for m in VF.MODES}; y['full-max'] = _y(min=11.31)
    a = VF.assess(dict(matching=_all(), yukawa=y))
    assert a['all_combined_yukawa_complete'] is True and a['robust_scale_combined_yukawa'] == 11.31 and a['robust_scale'] == 11.31 and 'combined' in a['robust_scale_basis']
    y2 = {m: _y() for m in VF.HPOLE}                                 # only the three h-pole combined runs (the previous default): not the full set
    a2 = VF.assess(dict(matching=_all(), yukawa=y2))
    assert a2['all_combined_yukawa_complete'] is False and a2['robust_scale_combined_yukawa'] is None and 'not complete' in a2['robust_scale_basis']
    y3 = {m: _y() for m in VF.MODES}; y3['ep'] = _y(complete=False, n_start_fail=2)
    assert VF.assess(dict(matching=_all(), yukawa=y3))['all_combined_yukawa_complete'] is False

def test_variant_status_semantics():
    st = CE.variant_status
    assert st(None) == 'ceiling' and st('None') == 'ceiling' and st('start:BFB') == 'boundary_failure' and st('match_failed') == 'match_failed'
    assert st('integration') == 'numerical_stop' and st('nonfinite') == 'numerical_stop' and st('BFB?') == 'unresolved'
    assert all(st(f) == 'physical_failure' for f in ('BFB', 'unitarity', 'perturbativity', 'yukawa'))
    assert CE.eligible('physical_failure') and CE.eligible('ceiling') and not any(CE.eligible(s) for s in ('boundary_failure', 'match_failed', 'numerical_stop', 'unresolved'))

def test_yukawa_variant_summary_fail_closed():
    import yukawa_run as YR
    out = {'a': dict(status='physical_failure'), 'b': dict(status='ceiling'), 'c': dict(status='boundary_failure'), 'd': dict(status='match_failed'), 'e': dict(status='numerical_stop'), 'f': dict(status='unresolved')}
    S = YR.summarize_variants(out)
    assert S['complete'] is False and S['n_variants'] == 6 and S['n_start_fail'] == 1 and S['n_match_failed'] == 1 and S['n_numerical'] == 2
    assert YR.summarize_variants({'a': dict(status='physical_failure'), 'b': dict(status='ceiling')})['complete'] is True

def test_smoke_light_tasks():
    for k in ('margins', 'hw'):
        kind, arg, res, dt = VF.task((k, 'BM1', None)); assert kind == k and res
    assert abs(VF.task(('margins', 'BM1', None))[2]['vred'] - 0.026) < 1e-3
