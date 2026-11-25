"""HiggsTools predictions for one parameter point and one choice of the free quartics, with
 - the h->gamma gamma coupling recomputed for those quartics (same Lagrangian as the RG analysis),
 - lepton-flavour-violating decays of h, H, A and the model total width of h,
 - charged scalars h1+, h2+ with l nu, t b, W h/H/A and (h2+) h1+ Z decays, pp -> t b h+ and t -> b h+,
 - A -> Z h, H -> Z A, A -> Z H, H/A -> h+- W-+ where open.
Decays into three scalars (H -> hh, h2+ -> h1+ h) are not included."""
import sys, numpy as np
sys.path.insert(0, '/home/claude/zee_fit'); sys.path.insert(0, '/home/claude/zee_fit/gate1')
import Higgs.predictions as HP, Higgs.bounds as HB, Higgs.signals as HS
from zee import hgmodel as H, neutrino as N
import m_filter_l4w as M
_b = HB.Bounds('/tmp/hbdataset'); _s = HS.Signals('/tmp/hsdataset')
V = H.V; MW, MZ, MT, MB = 80.379, 91.1876, 172.5, 4.18; G2 = 2*MW/V; CW = MW/MZ; GF = 1/(np.sqrt(2)*V**2)
lam = lambda a, b, c: max(a*a + b*b + c*c - 2*a*b - 2*a*c - 2*b*c, 0.0)
def svw(m1, m2, mv, G):   # Gamma(S1 -> S2 V) for vertex G (p1+p2)^mu
    if m1 <= m2 + mv: return 0.0
    return G**2*lam(m1**2, m2**2, mv**2)**1.5/(16*np.pi*m1**3*mv**2)
def run(x, eta=None):
    x = np.asarray(x, float); e = H.evaluate(list(x), 'NO'); sc = e['sc']; f, Y = H.unpack(x); Lam, Dd = H.lepton_couplings(Y, sc)
    tb, sba, cba = sc['tb'], sc['sba'], sc['cba']; c, s = np.cos(sc['phi']), np.sin(sc['phi'])
    rgg = e['rgg'] if eta is None else e['rgg']*M.rgg_gen(sc, eta[1], eta[2], eta[4])/M.rgg_gen(sc, 0.0, 0.0, 0.0)
    k = lambda p, i: Lam[p][i, i].real*V/H.M_LEP[i]
    lfv = lambda p, i, j: lambda m: m/(8*np.pi)*(abs(Lam[p][i, j])**2 + abs(Lam[p][j, i])**2)
    p = HP.Predictions(); out = {}
    h = p.addParticle(HP.NeutralScalar('h', 'even')); h.setMass(125.5)
    cc = HP.NeutralEffectiveCouplings(); kt = sba - tb*cba
    cc.tt = kt; cc.bb = kt; cc.cc = kt; cc.ss = kt; cc.ZZ = sba; cc.WW = sba; cc.tautau = k('h', 2); cc.mumu = k('h', 1)
    cc.gg = kt; cc.gamgam = float(np.sqrt(rgg)); cc.Zgam = sba
    HP.effectiveCouplingInput(h, cc, reference=HP.ReferenceModel.SMHiggsEW)
    for d, (i, j) in ((HP.Decay.mutau, (1, 2)), (HP.Decay.etau, (0, 2)), (HP.Decay.emu, (0, 1))): h.setDecayWidth(d, lfv('h', i, j)(125.5))
    out['Gamma_h'] = h.totalWidth(); out['BR_h_tau_mu'] = h.br(HP.Decay.mutau); out['BR_h_tau_e'] = h.br(HP.Decay.etau)
    ms = dict(h=125.5, H=sc['mH'], A=sc['mA'], Hp1=sc['m1'], Hp2=sc['m2'])
    for name, par in (('H', 'even'), ('A', 'odd')):
        X = p.addParticle(HP.NeutralScalar(name, par)); X.setMass(ms[name]); q = HP.NeutralEffectiveCouplings()
        if name == 'H':
            kH = cba + tb*sba; q.tt = kH; q.bb = kH; q.cc = kH; q.ss = kH; q.ZZ = cba; q.WW = cba; q.tautau = k('H', 2); q.mumu = k('H', 1)
        else:
            q.tt = 1j*tb; q.bb = 1j*tb; q.cc = 1j*tb; q.ss = 1j*tb; q.tautau = 1j*abs(k('A', 2)); q.mumu = 1j*abs(k('A', 1))
        HP.effectiveCouplingInput(X, q, reference=HP.ReferenceModel.SMHiggs)
        for d, (i, j) in ((HP.Decay.mutau, (1, 2)), (HP.Decay.etau, (0, 2)), (HP.Decay.emu, (0, 1))): X.setDecayWidth(d, lfv(name, i, j)(ms[name]))
    gZ = G2/(2*CW)
    p.particle('A').setDecayWidth(HP.ChainDecay.Z, 'h', svw(ms['A'], 125.5, MZ, gZ*cba))
    p.particle('A').setDecayWidth(HP.ChainDecay.Z, 'H', svw(ms['A'], ms['H'], MZ, gZ*sba))
    p.particle('H').setDecayWidth(HP.ChainDecay.Z, 'A', svw(ms['H'], ms['A'], MZ, gZ*sba))
    D = N.d_matrix(H.M_LEP, Y, tb); y2 = 2*f
    for name, w, ws in (('Hp1', c, s), ('Hp2', s, c)):
        m = ms[name]; A = tb*w; X = p.addParticle(HP.ChargedScalar(name)); X.setMass(m)
        for dec, i in zip((HP.Decay.enu, HP.Decay.munu, HP.Decay.taunu), range(3)):
            X.setDecayWidth(dec, m/(16*np.pi)*(w**2*np.sum(abs(D[:, i])**2) + ws**2*np.sum(abs(y2[:, i])**2)))
        X.setDecayWidth(HP.Decay.tb, 3*GF*m*MT**2*A**2/(4*np.sqrt(2)*np.pi)*(1 - MT**2/m**2)**2 if m > MT + MB else 0.0)
        for nn, cpl in (('h', cba), ('H', sba), ('A', 1.0)): X.setDecayWidth(HP.ChainDecay.W, nn, svw(m, ms[nn], MW, G2/2*w*cpl))
        if name == 'Hp2': X.setDecayWidth(HP.ChainDecay.Z, 'Hp1', svw(m, ms['Hp1'], MZ, gZ*c*s))
        brt = A**2*(1 - m**2/MT**2)**2/((1 - MW**2/MT**2)**2*(1 + 2*MW**2/MT**2)) if m < MT - MB else 0.0; brt = brt/(1 + brt)
        X.setCxn(HP.Collider.LHC13, HP.Production.Hpmtb, HP.EffectiveCouplingCxns.ppHpmtb(HP.Collider.LHC13, m, A, A, brt))
        if brt > 0: X.setCxn(HP.Collider.LHC13, HP.Production.brtHpb, brt)
        out[name + '_BRtaunu'] = X.br(HP.Decay.taunu); out[name + '_width'] = X.totalWidth()
    for name in ('H', 'A'):
        for hp, w in (('Hp1', c), ('Hp2', s)):
            cpl = (sba if name == 'H' else 1.0)*w
            p.particle(name).setDecayWidth(HP.ChainDecay.W, hp, 2*svw(ms[name], ms[hp], MW, G2/2*cpl))
    r = _b(p); out['HB_allowed'] = bool(r.allowed); out['HS_chi2'] = float(_s(p)); out['rgg'] = rgg
    try: out['HB_most_sensitive'] = str(r.selectedLimits) [:200]
    except Exception: pass
    return out, e
