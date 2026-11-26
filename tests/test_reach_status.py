"""Shared reach classifier and campaign status semantics (report v11, P1): a numerical stop with a positive prefix is never a valid reach, a start-failure sentinel never satisfies a baseline
cutoff, borderline BFB is distinct from a negative margin, and true physical crossings and ceilings keep their scale."""
import os, sys, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, 'gate1'), os.path.join(ROOT, 'gate2')]
os.environ.setdefault('ZEE_INPUTS', 'modern')
import cert as CE, bench


def V(*fails): return {str(i): dict(log10=5.0, fail=f) for i, f in enumerate(fails)}


def test_classes():
    assert CE.reach_class(7.0, V('integration', None)) == 'numerical_stop'
    assert CE.reach_class(-1.0, V('start:BFB', None)) == 'boundary_failure'
    assert CE.reach_class(7.0, V('BFB?', 'unitarity')) == 'unresolved'
    assert CE.reach_class(7.0, V('unitarity', 'BFB')) == 'physical_failure'
    assert CE.reach_class(16.0, V(None, None)) == 'ceiling'
    assert CE.reach_class(None, V(None)) == 'numerical_stop' and CE.reach_class(7.0, V('match_failed', None)) == 'match_failed'


def test_status_semantics():
    rc = lambda cs, *f: CE.reach_class(cs, V(*f))
    # numerical stop with a positive last-checked prefix above the requested cutoff must NOT meet it and must not suppress candidates: UNRESOLVED, never PASS
    r = CE.status_fields(9.0, rc(9.0, 'integration'), 8.0, 'PASS'); assert r[2] is False and r[3] == 'UNRESOLVED' and r[0].startswith('unresolved')
    # true physical crossing above / below the cutoff
    assert CE.status_fields(9.0, rc(9.0, 'unitarity'), 8.0, 'PASS')[2:] == (True, 'PASS')
    assert CE.status_fields(7.0, rc(7.0, 'unitarity'), 8.0, 'PASS')[2:] == (False, 'FAIL')
    # ceiling is a valid (right-censored) reach
    assert CE.status_fields(16.0, rc(16.0, None), 15.0, 'PASS')[2:] == (True, 'PASS')
    # baseline logL=-1: the start-failure sentinel -1 does not satisfy it; other classes are not compared with a cutoff
    r = CE.status_fields(-1.0, rc(-1.0, 'start:BFB'), -1.0, 'PASS'); assert r[1] is True and r[2] is None and r[3] == 'FAIL'
    r = CE.status_fields(5.0, rc(5.0, 'unitarity'), -1.0, 'PASS'); assert r[2] is None and r[3] == 'PASS'
    assert CE.status_fields(5.0, rc(5.0, 'integration'), -1.0, 'PASS')[3] == 'UNRESOLVED'
    # no cutoff requested: record status follows the low-energy status
    assert CE.status_fields(5.0, rc(5.0, 'unitarity'), None, 'PASS')[3] == 'PASS' and CE.status_fields(None, None, 8.0, 'FAIL')[3] == 'FAIL'


def test_bfb_borderline_is_distinguished_from_negative_margin():
    from zee import vac
    c = np.zeros(11); c[0] = c[1] = 0.1; c[10] = 0.1
    assert CE._bad(c) is None
    cneg = c.copy(); cneg[10] = -0.1; assert CE._bad(cneg) == 'BFB'
