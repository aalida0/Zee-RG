import os, sys, json, subprocess
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _jobs(tmp_path, kind):
    subprocess.run([sys.executable, os.path.join(ROOT, 'condor', 'make_campaign_sub.py'), 'T', kind], cwd=tmp_path, check=True, capture_output=True)
    return json.load(open(tmp_path / ('expected_jobs_%s_T.json' % kind))), (tmp_path / 'campaign.sub').read_text()


def test_full_job_list_is_unique_and_allocated_by_pilot_evidence(tmp_path):
    jobs, sub = _jobs(tmp_path, 'full')
    assert 1900 <= len(jobs) <= 2050
    ids = [(j['mode'], j['logL'], j['seed']) for j in jobs]; assert len(ids) == len(set(ids))          # one output file per job
    assert all(j['seed'] // 1000 == j['start'] for j in jobs)
    n = lambda **kw: sum(all(j[k] == v for k, v in kw.items()) for j in jobs)
    assert n(mode='sample', logL=10.0, start=2) <= 3 and n(mode='sample', logL=12.0, start=4) <= 3      # pilot-1 stalled pairs: probes only
    assert not any(j['mode'].startswith('max:') and j['logL'] == 14.0 and j['start'] in (2, 4) for j in jobs)
    assert all(j['margin'] == 0.0 for j in jobs if j['logL'] >= 12.0) and all(j['margin'] == 0.0 for j in jobs if j['mode'] == 'sample')
    assert n(mode='max:mu_e_g', logL=14.0) <= 10 and n(mode='sample', logL=12.0, start=0) >= 50
    assert 'arguments = $(mode) $(logL) $(seed) 2400' in sub and sub.count('\nqueue\n') == len(jobs)


def test_pilot_kinds_stay_small(tmp_path):
    assert len(_jobs(tmp_path, 'pilot')[0]) == 20 and len(_jobs(tmp_path, 'pilot2')[0]) == 11
