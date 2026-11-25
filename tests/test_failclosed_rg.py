"""C08: solver failure, empty output and the endpoint never count as 'valid to 1e16'."""
import os, sys, json, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path[:0] = [ROOT, os.path.join(ROOT, 'gate1'), os.path.join(ROOT, 'gate2')]
import lmax2 as L, bench
from zee import hgmodel as H

R = json.load(open(os.path.join(ROOT, 'results', 'benchmarks_input.json')))


def setup():
    x = np.array(R['BM2']['x']); eta = np.array(R['BM2']['eta'])
    sc, c = bench.coup(x, eta); return sc, c


class Fake:
    def __init__(self, status, t, y):
        self.status = status; self.t = np.array(t); self.y = np.array(y); self.t_events = [np.array([])]


def test_solver_failure_is_not_valid(monkeypatch):
    sc, c = setup(); t0 = np.log(min(sc['m1'], sc['mA'], sc['mH']))
    good, why = L.lam_valid(c, t0, -sc['tb'], status=True)
    assert good > 10 and why in ('none', 'perturbativity', 'unitarity')   # BM2 is limited by unitarity near 1e12-1e13
    monkeypatch.setattr(L.rgfast2, 'run', lambda *a, **k: Fake(-1, [], np.zeros((15, 0))))
    v, why = L.lam_valid(c, t0, -sc['tb'], status=True)
    assert why == 'integration' and v < np.log10(np.exp(t0))


def test_nonfinite_is_not_valid(monkeypatch):
    sc, c = setup(); t0 = np.log(min(sc['m1'], sc['mA'], sc['mH']))
    y = np.full((15, 1), np.nan)
    monkeypatch.setattr(L.rgfast2, 'run', lambda *a, **k: Fake(0, [t0 + 1.0], y))
    v, why = L.lam_valid(c, t0, -sc['tb'], status=True)
    assert why == 'nonfinite' and v < 16


def test_bench_trajectory_reaches_endpoint():
    sc, c = setup(); t0 = np.log(min(sc['m1'], sc['mA'], sc['mH']))
    r = bench.trajectory(c, t0, -sc['tb'], np.array([0.36, 0.64, 1.05, 0.88]), n_per_dec=5)
    if r['fail'] is None: assert r['solver_status'] == 0
    assert (r['fail'] is None and abs(r['log10Lambda'] - 16) < 1e-9) or (r['fail'] is not None and r['log10Lambda'] < 16)
