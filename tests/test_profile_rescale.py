import os, sys, json, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, ROOT + '/gate1', ROOT + '/gate2']; os.environ.setdefault('ZEE_INPUTS', 'modern')
from zee import hgmodel as H, neutrino as N


def mnu(x):
    sc = H.scalar_sector(x); f, Y = H.unpack(x)
    return N.mnu_hg_leading(f, H.M_LEP, Y, sc['tb'], sc['m1'], sc['m2'], sc['mu']), sc


def test_rescale_keeps_full_complex_neutrino_matrix_and_scalars():
    """The leading M_nu (the full complex 3x3 matrix, not only the mass-squared differences) and the scalar sector are invariant under
    gate2/profile_tb's continuation step; every nonzero f and Y2 entry is rescaled (regression for the wrong-index bug that scaled Y2^{e tau})."""
    pts = [np.array(s['x'], float) for s in json.load(open(ROOT + '/results/campaign_starts.json'))] + [np.array(json.load(open(ROOT + '/results/refit_hw.json'))['best']['x'], float)]
    for x in pts:
        M0, sc0 = mnu(x)
        for tbv in (0.2, 0.38, 0.45, 1.0, 3.0):
            x1 = H.rescale_fixed_nu(x, tbv); M1, sc1 = mnu(x1)
            assert np.abs(M1 - M0).max() <= 1e-12*np.abs(M0).max()
            assert sc1['m1'] == sc0['m1'] and sc1['m2'] == sc0['m2'] and sc1['mu'] == sc0['mu'] and sc1['tb'] == np.exp(x1[11]) == tbv or abs(sc1['tb'] - tbv) < 1e-12
            nz = np.nonzero(x[:11])[0]; assert len(nz) >= 5
            sb = lambda t: t/np.sqrt(1 + t*t); tb0 = np.exp(x[11])
            for i in nz: assert x1[i]/x[i] == (tb0/tbv if i < 2 else sb(tbv)/sb(tb0)) or abs(x1[i]/x[i] - (tb0/tbv if i < 2 else sb(tbv)/sb(tb0))) < 1e-12
