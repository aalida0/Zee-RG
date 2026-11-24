"""HiggsSignals chi2 (and HiggsBounds check) for fit points. UNTESTED here: HiggsTools is not installable in the sandbox
(not on PyPI for py3.13; GitLab blocked). Install on a machine with access:
  pip install higgstools ; git clone https://gitlab.com/higgsbounds/hbdataset ; git clone https://gitlab.com/higgsbounds/hsdataset
Usage: ZEE_INPUTS=modern python gate1/higgstools_chi2.py SAMPLES.pkl HBDATASET HSDATASET OUT.pkl
Check the API names against the HiggsTools docs before trusting results."""
import sys, pickle, numpy as np
sys.path.insert(0, '.')
from zee import hgmodel as H
import Higgs.predictions as HP, Higgs.bounds as HB, Higgs.signals as HS
S = pickle.load(open(sys.argv[1], 'rb')); hb = HB.Bounds(sys.argv[2]); hs = HS.Signals(sys.argv[3])
out = []
for c, x in S:
    e = H.evaluate(list(x), 'NO'); sc = e['sc']; tb = sc['tb']
    kV = sc['sba']; kt = sc['sba'] - tb * sc['cba']; ktau = e['mus']['tt'] ** 0.5 if False else None
    p = HP.Predictions(); h = p.addParticle(HP.NeutralScalar('h', 'even')); h.setMass(125.5)
    cp = HP.NeutralEffectiveCouplings(); cp.tt = kt; cp.bb = kt; cp.cc = kt; cp.ZZ = kV; cp.WW = kV
    cp.tautau = H.evaluate(list(x), 'NO')['mus']['tt'] ** 0.5; cp.gg = kt; cp.gamgam = e['rgg'] ** 0.5; cp.Zgam = 1.0   # approximations: refine with loop-level kappas
    HP.effectiveCouplingInput(h, cp, reference=HP.ReferenceModel.SMHiggsEW)
    out.append((c, hs(p), hb(p).allowed))
pickle.dump(out, open(sys.argv[4], 'wb'))
