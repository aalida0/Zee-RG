"""matched_scales fail-closed semantics and cache key (report v11 4.3)."""
import os, sys, numpy as np
_R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [_R, _R + '/gate1', _R + '/gate2']
os.environ.setdefault('ZEE_INPUTS', 'modern')
import matched_scales as MS

def test_cache_key_distinguishes_inputs():
    x = np.arange(5.); e = np.ones(6); g = np.array([.36, .64, 1.05, .88])
    k = MS._key(x, e, 'ep', 1.0, 200., g)
    assert k == MS._key(x.copy(), e.copy(), 'ep', 1.0, 200., g.copy())
    for k2 in (MS._key(x + 1e-9, e, 'ep', 1.0, 200., g), MS._key(x, e*1.0000001, 'ep', 1.0, 200., g), MS._key(x, e, 'full-sym', 1.0, 200., g),
               MS._key(x, e, 'ep', 2.0, 200., g), MS._key(x, e, 'ep', 1.0, 201., g), MS._key(x, e, 'ep', 1.0, 200., g*1.001)):
        assert k2 != k
    hash(k)   # immutable, hashable

def test_match_valid_rejects():
    good = dict(c=np.ones(11), P={'a': 1.0}, ok=True, resid=1e-12, error=None)
    assert MS.match_valid(good) is None
    assert MS.match_valid(dict(good, ok=False)) is not None
    assert MS.match_valid(dict(good, resid=1e-3)) is not None
    assert MS.match_valid(dict(good, resid=float('nan'))) is not None
    assert MS.match_valid(dict(good, c=np.r_[np.ones(10), np.nan])) is not None
    assert MS.match_valid(dict(good, error='ValueError')) == 'ValueError'

def test_summary_never_turns_failure_into_scale():
    res = {'default': dict(log10Lambda=None, fail='start:BFB', stress=False), 'match_x1': dict(log10Lambda=12.5, fail='unitarity', stress=False),
           'qscale_1.1': dict(log10Lambda=11.0, fail='BFB', stress=True), 'sm_yt_1.02': dict(log10Lambda=None, fail='match_failed', stress=False)}
    S = MS.summarize(res)
    assert S['min_rematched'] == 12.5 and S['min_stress'] == 11.0 and not S['complete'] and S['n_match_failed'] == 1 and S['n_start_fail'] == 1
    assert MS.summarize({'a': dict(log10Lambda=None, fail='match_failed', stress=False)})['min_rematched'] is None

def test_numerical_and_borderline_stops_have_no_scale():
    """report v12 3.3: integration / nonfinite / 'BFB?' stops carry a finite prefix that is NOT a physical reach"""
    res = {'default': dict(log10Lambda=12.0, fail='unitarity', stress=False), 'match_x1': dict(log10Lambda=14.2, fail='integration', stress=False),
           'matchH_x1': dict(log10Lambda=13.0, fail='nonfinite', stress=False), 'qscale_1.1': dict(log10Lambda=9.0, fail='BFB?', stress=True),
           'sm_yt_1.02': dict(log10Lambda=16.0, fail=None, stress=False)}
    S = MS.summarize(res)
    assert not S['complete'] and S['n_numerical'] == 3 and S['n_defined'] == 2 and S['numerical_prefix_min'] == 9.0
    assert S['min_rematched'] == 12.0 and S['min_stress'] == 12.0                 # the 9.0 of the borderline stress variant is not a scale; the ceiling (16.0) is
    ok = MS.summarize({'a': dict(log10Lambda=16.0, fail=None, stress=False), 'b': dict(log10Lambda=11.0, fail='BFB', stress=True)})
    assert ok['complete'] and ok['min_stress'] == 11.0 and ok['n_numerical'] == 0

def test_no_mutable_default_cache():
    import inspect
    assert inspect.signature(MS.piece_matched).parameters['cache'].default is None
