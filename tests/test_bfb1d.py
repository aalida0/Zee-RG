import os, sys, numpy as np, json
R = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path[:0] = [R]
from zee import bfb1d, bfb2


def test_1d_reduction_agrees_with_grid_on_random_real_quartics():
    rng = np.random.default_rng(3); n_sign = n = 0; worst = 0.0
    for _ in range(300):
        c = np.concatenate([rng.uniform(0.1, 1.5, 2), rng.uniform(-1, 2, 5), rng.uniform(-0.6, 1.0, 3), [rng.uniform(0.01, 1.0)]])
        v1 = bfb1d.min_vred_1d(c); v2 = bfb2.min_vred(c)
        n += 1; n_sign += (v1 >= 0) == (v2 >= 0)
        worst = max(worst, abs(v1 - v2) - 2e-3)                  # grid + pattern search vs exact-in-u, 1-d grid in a
        assert v1 <= v2 + 1e-4                                   # the exact-in-u minimum can only be lower than the grid minimum
    assert n_sign >= n - 3                                       # same verdict except within a few marginal cases
    assert worst < 5e-2


def test_known_unbounded_and_bounded_cases():
    ok = [0.26, 1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.1]
    assert bfb1d.min_vred_1d(ok) > 0
    bad = list(ok); bad[3] = -2.0                                # l4 << 0 with l5 = 0: V2 < 0 along z
    assert bfb1d.min_vred_1d(bad) < 0 and bfb2.min_vred(bad) < 0
    lhz = list(ok); lhz[10] = 0.0; assert bfb1d.min_vred_1d(lhz) > 0
    lhz[7] = -0.1; lhz[8] = 0.2; assert bfb1d.min_vred_1d(lhz) < 0   # B < 0 at a = 1 with lh = 0: unbounded along the singlet
    neg = list(ok); neg[10] = -0.01; assert bfb1d.min_vred_1d(neg) == -np.inf


def test_benchmarks_bounded_in_both_implementations():
    sys.path[:0] = [R + '/gate2']; import bench
    d = json.load(open(R + '/results/benchmarks_input.json'))
    for k in d:
        sc, c = bench.coup(np.array(d[k]['x']), np.array(d[k]['eta']))
        assert bfb1d.min_vred_1d(c) > 0 and bfb2.min_vred(c) > 0
        assert abs(bfb1d.min_vred_1d(c) - bfb2.min_vred(c)) < 5e-3
