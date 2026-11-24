import numpy as np
from zee import bfb, bfb2
def test_counterexample_to_hg_condition():
    c = np.zeros(11); c[0] = .26; c[1] = 3.54; c[5] = .8; c[10] = 1.0
    assert 2*abs(c[5]+c[6]) < (c[0]+c[1])/2 + c[2]+c[3]+c[4]          # HG Eq. (3.3) satisfied
    assert bfb2.min_vred(c) < -0.3 and bfb.min_v4(c) < -0.3            # but unbounded
def test_exact_reduction_matches_minimiser():
    rng = np.random.default_rng(11)
    for _ in range(25):
        c = np.zeros(11); c[0] = rng.uniform(.2, .6); c[1] = rng.uniform(.05, 3.5); c[2:5] = rng.uniform(-1, 1.5, 3); c[5] = rng.uniform(-.9, .9)
        c[6:10] = rng.uniform(-1.5, 1.5, 4); c[10] = rng.uniform(.05, 3)
        assert (bfb2.min_vred(c) < -1e-4) == (bfb.min_v4(c, n=4000) < -1e-4)
