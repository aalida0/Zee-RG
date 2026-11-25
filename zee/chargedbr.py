"""Total widths and branching ratios of the two charged scalars without HiggsTools (so that the Drell-Yan screen can run
anywhere).  The channels are the ones of gate2/hcoll.py: l nu, t b, light quarks, W h/H/A, h1+ Z (h2+ only) and h2+ -> h1+ h.
tests/test_chargedbr.py compares with the HiggsTools branching ratios when the Higgs module is installed."""
import numpy as np
from . import hgmodel as H, widths as WD, dipfull as DF, V_EW as V

MW, MZ, MT, MB = 80.379, 91.1876, 172.5, 4.18
G2 = 2*MW/V; CW = MW/MZ; GF = 1/(np.sqrt(2)*V**2)


def _svw(m1, m2, mv, G):
    if m1 <= m2 + mv: return 0.0
    return G**2*WD.lam_k(m1**2, m2**2, mv**2)**1.5/(16*np.pi*m1**3*mv**2)


def widths_and_brs(x, eta):
    """-> dict(m=[m1,m2], d=[cos^2 phi, sin^2 phi], width=[...], BR={'enu','munu','taunu','tb','qq','Wneutral','Zh1','h1h'} per state)"""
    x = np.asarray(x, float); eta = np.zeros(6) if eta is None else np.asarray(eta, float); e = H.evaluate(list(x), "NO", eta=eta); sc = e["sc"]
    f, Y = H.unpack(x); Lam, Dd = H.lepton_couplings(Y, sc); GL = H.charged_lepton_widths(f, Dd, sc)
    tb, c, s = sc['tb'], np.cos(sc['phi']), np.sin(sc['phi']); sba, cba = sc['sba'], sc['cba']
    mh, mH, mA = 125.5, sc['mH'], sc['mA']; gZ = G2/(2*CW)
    l7 = float(eta[1]); l8 = float(eta[2]); l10 = float(eta[4])
    out = dict(m=[sc['m1'], sc['m2']], d=[c*c, s*s], width=[], BR=[], sc=sc)
    for k, (m, w) in enumerate(((sc['m1'], c), (sc['m2'], s))):
        A = tb*w; part = {}
        for i, nm in enumerate(('enu', 'munu', 'taunu')): part[nm] = float(GL[k, i])
        part['tb'] = 3*GF*m*MT**2*A**2/(4*np.sqrt(2)*np.pi)*(1 - MT**2/m**2)**2 if m > MT + MB else 0.0
        part['qq'] = float(sum(WD.gamma_hp_quarks(m, A).values()))
        part['Wneutral'] = sum(_svw(m, mm, MW, G2/2*w*cp) for mm, cp in ((mh, cba), (mH, sba), (mA, 1.0)))
        part['Zh1'] = _svw(m, sc['m1'], MZ, gZ*c*s) if k == 1 else 0.0
        part['h1h'] = WD.gamma_h2_h1_h(sc, DF.charged_trilinears(sc, l7, l8, l10)['raw_h']) if k == 1 else 0.0
        tot = sum(part.values()); out['width'].append(tot); out['BR'].append({n: v/tot for n, v in part.items()})
    return out
