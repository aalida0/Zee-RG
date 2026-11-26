"""Collector selection (report v11, P2-P4): candidates are identified by the full 25-vector, copies found by several jobs cannot fill the top-K slots, all memberships survive, and the portfolio carries
comfortable-margin points that are not record-breaking."""
import os, sys, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, 'gate1'), os.path.join(ROOT, 'gate2')]
os.environ.setdefault('ZEE_INPUTS', 'modern')
import collect_campaign as C


def pt(x0, e0, rate): return dict(x=[x0]*19, eta=[e0]*6, obs=dict(mu_e_g=rate))


def test_same_x_different_eta_are_distinct():
    a, b = pt(1.0, 0.1, 5.0), pt(1.0, 0.2, 4.0)
    assert C.exact_key(a) != C.exact_key(b)
    out = C.dedup_group([(a, 3, 'j1'), (b, 3, 'j2')], 'mu_e_g'); assert len(out) == 2 and out[0][0] is a


def test_cross_job_duplicates_do_not_fill_topk_and_provenance_is_merged():
    best = pt(1.0, 0.1, 9.0); others = [pt(2.0 + i, 0.1, 5.0 - i) for i in range(3)]
    L = [(dict(best), 3, 'job%d' % j) for j in range(10)] + [(o, 3, 'jobo') for o in others]      # ten maximisers kept the same initial benchmark
    out = C.dedup_group(L, 'mu_e_g'); assert len(out) == 4 and len(out[0][2]) == 10 and [t[0]['obs']['mu_e_g'] for t in out] == [9.0, 5.0, 4.0, 3.0]


def test_portfolio_keeps_a_comfortable_margin_point_that_is_not_a_record():
    good = {}
    for i in range(30):
        good[('p', i)] = dict(obs=dict(mu_e_g=1e-13*(1 + i)), cs=11.0 + 0.01*i, margin=0.002 + 0.0005*i if i != 7 else 0.05, x=[0]*19, eta=[0]*6, lam_h=0.1)
    port = C.portfolio(good, ('mu_e_g',), per_obs=8)['mu_e_g']
    assert ('p', 29) in port and ('p', 7) in port and 2 < len(port) <= 8       # the record-holder and the high-margin point both survive
    assert C.pareto_front([dict(a=1, b=1), dict(a=2, b=0), dict(a=0, b=0)], ('a', 'b')) == [0, 1]


def test_matched_proxy_flags_bm2_not_bm1():
    import json
    inp = json.load(open(os.path.join(ROOT, 'results', 'benchmarks_input.json')))
    b1 = C.matched_proxy(inp['BM1']['x'], inp['BM1']['eta'], modes=('full-max',)); b2 = C.matched_proxy(inp['BM2']['x'], inp['BM2']['eta'], modes=('full-max',))
    assert b1['rmargin'] > 0.02 and b2['rmargin'] < 0         # BM2 loses boundedness at the boundary under the heavy-momentum h-H prescription, BM1 does not
