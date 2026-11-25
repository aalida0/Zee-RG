import os, sys, pickle, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from zee import hgmodel as H, vac

def test_generic_trilepton_matches_hg_on_slice():
    rng = np.random.default_rng(1); n = 0
    for _ in range(40):
        x = H.LO + rng.random(18) * (H.HI - H.LO); x[:11] = rng.choice([-1, 1], 11) * 10**rng.uniform(-6, -2, 11)
        try: e = H.evaluate(list(x), 'NO')
        except Exception: continue
        f, Y = H.unpack(x); sc = e['sc']
        if not np.isfinite(e['chi2']): continue
        n += 1
        assert e['obs']['tau_3mu_hg'] > 0 and abs(e['obs']['tau_3mu_gen']/e['obs']['tau_3mu_hg'] - 1) < 1e-8   # HG (3.9) vs generic
        assert e['obs']['mu_3e_tree'] == 0.0 or e['obs']['mu_3e_tree'] < 1e-40
    assert n >= 10

def test_unitarity_2hdm():
    c = np.zeros(11); c[:5] = [1.0, 0.5, 0.3, 0.2, 0.1]; l1, l2, l3, l4, l5 = c[:5]
    a = 1.5*(l1 + l2) + np.sqrt(2.25*(l1 - l2)**2 + (2*l3 + l4)**2)
    assert np.isclose(vac.unit_max(c), a)

def test_ew_vacuum_is_stationary():
    sc = dict(MHp=300.0, M33=500.0, mu=100.0); c = np.array([0.26, 1.0, 0.5, 0.2, 0.1, 0.05, 0, 0, 0, 0, 1.0])
    m11 = -c[0]/2; m22 = (300.0**2 - 0.5*246.22**2*c[2])/246.22**2; m12 = c[5]/2; mh2 = 500.0**2/246.22**2
    p0 = np.zeros(10); p0[2] = 1.0; e = np.eye(10)
    g = [(vac.vfull(p0 + 1e-6*e[i], c, m11, m22, m12, mh2, 100/246.22) - vac.vfull(p0 - 1e-6*e[i], c, m11, m22, m12, mh2, 100/246.22))/2e-6 for i in range(10)]
    assert np.allclose(g, 0, atol=1e-6)
