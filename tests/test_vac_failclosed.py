import json, numpy as np
from zee import vac, hgmodel as H

R = json.load(open('results/benchmarks.json'))


def setup(name='BM1'):
    r = R[name]; x = np.array(r['x']); eta = np.array(r['eta'])
    sc = H.evaluate(list(x), 'NO', eta=eta)['sc']
    c = np.zeros(11); c[0], c[2], c[3], c[4], c[5] = sc['lam1'], sc['lam3'], sc['lam4'], sc['lam5'], sc['lam6']; c[1], c[6:] = eta[0], eta[1:]
    return c, sc


def test_gradient_matches_finite_difference():
    c, sc = setup(); V = 246.22
    args = (c, -c[0]/2, (sc['MHp']**2 - 0.5*V**2*c[2])/V**2, c[5]/2, (sc['M33']**2 - 0.5*V**2*c[7])/V**2, sc['mu']/V)
    p = np.random.default_rng(3).normal(size=10)
    v, g = vac.vgrad(p, *args)
    assert abs(v - float(vac.vfull(p, *args))) < 1e-12*max(1, abs(v))
    h = 1e-6; fd = np.array([(vac.vgrad(p + h*e, *args)[0] - vac.vgrad(p - h*e, *args)[0])/(2*h) for e in np.eye(10)])
    assert np.allclose(g, fd, rtol=1e-6, atol=1e-8)


def test_ew_vacuum_is_stationary():
    c, sc = setup()
    r = vac.global_search(c, sc, nstart=0)
    assert r['ew_grad'] < 1e-8


def test_all_failed_is_not_ok(monkeypatch):
    c, sc = setup()
    import scipy.optimize as so
    class Bad: fun = np.nan; x = np.full(10, np.nan); jac = np.full(10, np.nan)
    monkeypatch.setattr(vac, 'minimize', lambda *a, **k: Bad())
    r = vac.global_search(c, sc, nstart=5)
    assert r['status'] == 'unresolved' and not vac.global_ok(c, sc, nstart=5)[0]
