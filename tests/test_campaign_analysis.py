import os, sys, json, numpy as np
_R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [_R, _R + '/gate1', _R + '/gate2', _R + '/analysis']
os.environ.setdefault('ZEE_INPUTS', 'modern')
import campaign_analysis as CA

def test_spearman_and_partial_known_cases():
    rng = np.random.default_rng(0); z = rng.normal(size=400); a = z + 0.1*rng.normal(size=400); b = z + 0.1*rng.normal(size=400)
    assert CA.spearman(a, b) > 0.95
    assert abs(CA.partial_spearman(a, b, [z])) < 0.25          # the correlation is entirely explained by the control z
    assert abs(CA.spearman(rng.normal(size=400), rng.normal(size=400))) < 0.2

def test_lambda6_study_insufficient_and_ok():
    rng = np.random.default_rng(1)
    def pt(i, seed): return dict(lam6=rng.normal(), h_tau_mu=10**rng.uniform(-6, -4), mH_A_mean=700 + rng.normal()*50, maxY2=0.02, reach=11 + rng.normal(), seed=seed)
    few = [pt(i, i % 4) for i in range(10)]; assert CA.lambda6_study(few, rng)['status'].startswith('insufficient')
    many = []
    for i in range(120):
        p = pt(i, i % 4); p['h_tau_mu'] = 1e-5*np.exp(2*p['lam6']); many.append(p)
    r = CA.lambda6_study(many, rng); assert r['status'] == 'ok' and r['spearman_lam6_logBRhtm']['value'] > 0.9 and r['partial_given_mH_Y2']['value'] > 0.9
    lo, hi = r['spearman_lam6_logBRhtm']['ci']; assert lo > 0.8

def test_start_none_never_pooled():
    rows = [dict(start=None), dict(start=0), dict(start=0), dict(start=1)]
    ps = CA.per_start(rows); assert set(ps) == {0, 1} and len(ps[0]) == 2

def test_smoke_pipeline_on_pilot_summary(tmp_path):
    import subprocess
    out = tmp_path / 'o'
    r = subprocess.run([sys.executable, '-W', 'ignore', os.path.join(_R, 'analysis', 'campaign_analysis.py'), os.path.join(_R, 'results', 'campaign_summary.json'), str(out)], capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-800:]
    for f in ('points.csv', 'report.md', 'fig_rates_vs_reach.pdf', 'fig_spectrum.pdf', 'fig_lambda6.pdf', 'fig_baseline_vs_highscale.pdf', 'analysis.json'): assert (out / f).exists(), f
    import csv
    rows = list(csv.DictReader(open(out / 'points.csv'))); assert len(rows) > 10 and sum(bool(r['reach']) for r in rows) > 5      # the figures must contain points
