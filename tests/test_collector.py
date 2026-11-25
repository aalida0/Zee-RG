import os, sys, json, numpy as np, pytest
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, ROOT + '/gate1', ROOT + '/gate2']; os.environ.setdefault('ZEE_INPUTS', 'modern')
pytest.importorskip('campaign')
import collect_campaign as CC


def _job(tmp, name, mode, logL, seed, start, best, archive=()):
    man = dict(code='c', data='d', run_id='t', commit='x', env=None, collider_data={}, packages={}, dirty=False, hw_scale=1.0)
    j = dict(mode=mode, logL=logL, seed=seed, start=start, manifest=man, wall=1.0, n_prop=10, n_acc=1, complete=True, points=[], trace=[], best=best, best_archive=list(archive), best_unresolved=None,
             rejects={}, higgs=True, n_post=0, n_cert=0, n_enter=0, climb_at_end=False, burn=0.25, nprop_target=10, margin=0.0)
    json.dump(j, open(os.path.join(tmp, name), 'w'))


def _rec(x0, rate):
    x = [x0] + [0.0]*18; return dict(x=x, eta=[0.0]*6, obs=dict(mu_e_g=rate, tau_mu_g=rate, tau_e_g=rate, h_tau_mu=rate))


def run_collect(tmp, monkeypatch, order):
    import accept as AC, cert as CE, campaign as C
    calls = []
    def fake_accept(x, eta, cmin, logL=None, collider=True, n_vac=200):
        calls.append(float(x[0])); return dict(status='PASS', failed=[], chi2=12.0, obs=dict(mu_e_g=float(x[0]), tau_mu_g=float(x[0]), tau_e_g=float(x[0]), h_tau_mu=float(x[0])), collider=dict(BR_h_tau_mu=float(x[0])))
    # reach: x0 = 1 -> physical failure at 12; x0 = 2 -> boundary failure (sentinel -1); x0 = 3 -> numerical stop with a positive prefix 13
    def fake_cert(x, eta, variants=None):
        k = round(x[0])
        if k == 1: return 12.0, {'default': dict(log10=12.0, fail='unitarity')}
        if k == 2: return -1.0, {'default': dict(log10=-1.0, fail='start:BFB')}
        return 13.0, {'default': dict(log10=13.0, fail='nonfinite')}
    monkeypatch.setattr(AC, 'accept', fake_accept); monkeypatch.setattr(CE, 'cert', fake_cert)
    monkeypatch.setattr(C, 'manifest', lambda: dict(code='c', data='d', commit='x', collider_data={}, packages={}, hw_scale=1.0, dirty=False))
    for k in ('ALLOW_MIXED', 'ALLOW_DIRTY', 'ALLOW_NO_COLLIDER'): monkeypatch.setenv(k, '1')
    monkeypatch.setenv('ZEE_OUT', str(tmp)); monkeypatch.setattr(sys, 'argv', ['collect', str(tmp), '5', '0', '4'])
    CC.OUT = None; CC.main()
    return json.load(open(os.path.join(tmp, 'campaign_summary.json'))), json.load(open(os.path.join(tmp, 'campaign_finalists.json')))


@pytest.mark.parametrize('order', [0, 1])
def test_same_input_in_two_groups_keeps_both_memberships_and_sentinel_is_not_a_scale(tmp_path, monkeypatch, order):
    d = tmp_path / ('o%d' % order); d.mkdir()
    A = _rec(1.0, 1e-13); Bsent = _rec(2.0, 5e-13); Cstop = _rec(3.0, 9e-13)
    jobs = [('camp_a.json', 'max:mu_e_g', 10.0, 1, 0, A), ('camp_b.json', 'max:tau_mu_g', 11.0, 2, 0, A), ('camp_c.json', 'max:mu_e_g', 10.0, 3, 0, Bsent), ('camp_d.json', 'max:mu_e_g', 12.0, 4, 0, Cstop)]
    for nm, mode, L, seed, st, rec in (jobs if order == 0 else jobs[::-1]): _job(str(d), nm, mode, L, seed, st, rec)
    summ, fin = run_collect(d, monkeypatch, order)
    mem = [v for v in summ['verified'] if v['x'][0] == 1.0]
    assert sorted(v['group'] for v in mem) == ['max:mu_e_g|10', 'max:tau_mu_g|11']                       # the same input appears in both groups, whatever the file order
    by = {v['group']: v for v in mem}
    assert by['max:mu_e_g|10']['meets_cutoff'] is True and by['max:tau_mu_g|11']['meets_cutoff'] is True       # screening reach 12 >= both 10 and 11
    sent = [v for v in summ['verified'] if v['x'][0] == 2.0][0]
    assert sent['reach_class'] == 'boundary_failure' and sent['meets_cutoff'] is False and sent['screen13'] is None   # -1 sentinel never satisfies a cutoff
    stop = [v for v in summ['verified'] if v['x'][0] == 3.0][0]
    assert stop['reach_class'] == 'numerical_stop' and stop['meets_cutoff'] is False and stop['prefix_screen13'] == 13.0    # positive prefix is reported but is not a cutoff
    assert all(f['x'][0] == 1.0 for f in fin.values())                                                    # only the validly certified point can be a finalist


def test_unwritable_output_directory_stops_early(tmp_path, monkeypatch):
    """report v10 P0: a frozen read-only checkout needs ZEE_OUT; an unwritable output directory is reported before any work"""
    _job(str(tmp_path), 'camp_a.json', 'max:mu_e_g', 10.0, 1, 0, _rec(1.0, 1e-13))
    monkeypatch.setenv('ZEE_OUT', '/proc/zee_out_not_writable'); monkeypatch.setattr(sys, 'argv', ['collect', str(tmp_path)])
    with pytest.raises(SystemExit) as e: CC.main()
    assert 'not writable' in str(e.value)
