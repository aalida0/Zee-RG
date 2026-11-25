import sys, pickle, numpy as np
_R = __import__('os').path.dirname(__import__('os').path.dirname(__import__('os').path.abspath(__file__)))
sys.path.insert(0, _R)
import Higgs.predictions as HP, Higgs.bounds as HB, Higgs.signals as HS
from zee import hgmodel as H
HBD, HSD = '/tmp/hbdataset', '/tmp/hsdataset'
_b = HB.Bounds(HBD); _s = HS.Signals(HSD)
def run(x, ordering='NO', with_bounds=True, extra=None):
    e = H.evaluate(list(x), ordering); sc = e['sc']
    f, Y = H.unpack(np.array(x)); Lam, Dd = H.lepton_couplings(Y, sc)
    tb, sba, cba = sc['tb'], sc['sba'], sc['cba']
    kV = sba; kt = sba - tb * cba
    k = lambda p, i: Lam[p][i, i].real * H.V / H.M_LEP[i]
    p = HP.Predictions()
    h = p.addParticle(HP.NeutralScalar('h', 'even')); h.setMass(125.5)
    c = HP.NeutralEffectiveCouplings(); c.tt = kt; c.bb = kt; c.cc = kt; c.ss = kt; c.ZZ = kV; c.WW = kV
    c.tautau = k('h', 2); c.mumu = k('h', 1); c.gg = kt; c.gamgam = float(np.sqrt(e['rgg'])); c.Zgam = kV
    HP.effectiveCouplingInput(h, c, reference=HP.ReferenceModel.SMHiggsEW)
    G = 4.1e-3
    h.setDecayWidth('mutau', e['obs']['h_tau_mu'] * G / 2); h.setDecayWidth('etau', e['obs']['h_tau_e'] * G / 2)
    allowed = None
    if with_bounds:
        for name, par, mass in (('H', 'even', sc['mH']), ('A', 'odd', sc['mA'])):
            X = p.addParticle(HP.NeutralScalar(name, par)); X.setMass(mass); cc = HP.NeutralEffectiveCouplings()
            if name == 'H':
                kHt = cba + tb * sba; cc.tt = kHt; cc.bb = kHt; cc.cc = kHt; cc.ss = kHt; cc.ZZ = cba; cc.WW = cba; cc.tautau = k('H', 2); cc.mumu = k('H', 1)
            else:
                cc.tt = 1j * tb; cc.bb = 1j * tb; cc.cc = 1j * tb; cc.ss = 1j * tb; cc.tautau = 1j * abs(k('A', 2)); cc.mumu = 1j * abs(k('A', 1))
            HP.effectiveCouplingInput(X, cc, reference=HP.ReferenceModel.SMHiggs)
        ex = extra(p) if extra else None
        r = _b(p); allowed = bool(r.allowed)
    if extra: return float(_s(p)), allowed, e, ex
    return float(_s(p)), allowed, e
if __name__ == '__main__':
    pts = pickle.load(open('/tmp/mbest_NO.pkl', 'rb'))[:1]
    p0 = HP.Predictions(); h0 = p0.addParticle(HP.NeutralScalar('h', 'even')); h0.setMass(125.5); HP.effectiveCouplingInput(h0, HP.smLikeEffCouplings, reference='SMHiggsEW')
    print('SM chisq', _s(p0), 'nobs', _s.observableCount())
    for c0, x in pts:
        chi, ok, e = run(x); print('HS chi2', chi, 'HB allowed', ok, 'old higgs chi2', e['chi_higgs'], 'kappas', e['mus'])
