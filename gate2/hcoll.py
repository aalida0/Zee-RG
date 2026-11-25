"""HiggsTools predictions for one parameter point and one choice of the free quartics, with
 - the h->gamma gamma coupling recomputed for those quartics (same Lagrangian as the RG analysis),
 - lepton-flavour-violating decays of h, H, A and the model total width of h,
 - charged scalars h1+, h2+ with l nu, t b, W h/H/A and (h2+) h1+ Z decays, pp -> t b h+ and t -> b h+,
 - A -> Z h, H -> Z A, A -> Z H, H/A -> h+- W-+ where open.
pp -> h+ h- (Drell-Yan) is confronted with slepton searches via zee/dy.py.
Decays into three scalars (H -> hh, h2+ -> h1+ h) are not included."""
import sys, os, numpy as np
_R = __import__('os').path.dirname(__import__('os').path.dirname(__import__('os').path.abspath(__file__)))
sys.path.insert(0, _R); sys.path.insert(0, _R + '/gate1')
import Higgs.predictions as HP, Higgs.bounds as HB, Higgs.signals as HS
from zee import hgmodel as H, neutrino as N, dy as DY
import m_filter_l4w as M
_b = HB.Bounds(os.environ.get('ZEE_HBDATA', '/tmp/hbdataset')); _s = HS.Signals(os.environ.get('ZEE_HSDATA', '/tmp/hsdataset'))
V = H.V; MW, MZ, MT, MB = 80.379, 91.1876, 172.5, 4.18; G2 = 2*MW/V; CW = MW/MZ; GF = 1/(np.sqrt(2)*V**2)
lam = lambda a, b, c: max(a*a + b*b + c*c - 2*a*b - 2*a*c - 2*b*c, 0.0)
def svw(m1, m2, mv, G):   # Gamma(S1 -> S2 V) for vertex G (p1+p2)^mu
    if m1 <= m2 + mv: return 0.0
    return G**2*lam(m1**2, m2**2, mv**2)**1.5/(16*np.pi*m1**3*mv**2)
def run(x, eta=None):
    x = np.asarray(x, float); e = H.evaluate(list(x), 'NO', eta=eta); sc = e['sc']; f, Y = H.unpack(x); Lam, Dd = H.lepton_couplings(Y, sc)
    tb, sba, cba = sc['tb'], sc['sba'], sc['cba']; c, s = np.cos(sc['phi']), np.sin(sc['phi'])
    rgg = e['rgg']   # computed in evaluate with the free quartics eta (same statistic as the fit)
    # diagonal effective coupling relative to the SM: -L = phi ebar_L Lam e_R + h.c. = phi ebar (Re Lam + i Im Lam g5) e,
    # so Re -> CP-even and Im -> CP-odd part (HiggsTools complex kappa). For A = i Dd/sqrt2 the CP-odd part is Re Dd/sqrt2.
    k = lambda p, i: complex(Lam[p][i, i])*V/H.M_LEP[i]
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
            q.tt = 1j*tb; q.bb = 1j*tb; q.cc = 1j*tb; q.ss = 1j*tb; q.tautau = k('A', 2); q.mumu = k('A', 1)
        HP.effectiveCouplingInput(X, q, reference=HP.ReferenceModel.SMHiggs)
        for d, (i, j) in ((HP.Decay.mutau, (1, 2)), (HP.Decay.etau, (0, 2)), (HP.Decay.emu, (0, 1))): X.setDecayWidth(d, lfv(name, i, j)(ms[name]))
    gZ = G2/(2*CW)
    p.particle('A').setDecayWidth(HP.ChainDecay.Z, 'h', svw(ms['A'], 125.5, MZ, gZ*cba))
    p.particle('A').setDecayWidth(HP.ChainDecay.Z, 'H', svw(ms['A'], ms['H'], MZ, gZ*sba))
    p.particle('H').setDecayWidth(HP.ChainDecay.Z, 'A', svw(ms['H'], ms['A'], MZ, gZ*sba))
    # h_k+ -> l_j nu: -L = nubar U^dag Dd e_R (c h1+ - s h2+) + nu^T C (2f) e_L (s h1+ + c h2+)  (HG 3.26-3.27,
    # Aristizabal Sierra-Restrepo hep-ph/0604012): width into l_j = column-j norm of Dd (resp. 2f), Dd = Y2^dag/cb - sqrt2 tb m/v
    GL = H.charged_lepton_widths(f, Dd, sc)
    for kk, (name, w, ws) in enumerate((('Hp1', c, s), ('Hp2', s, c))):
        m = ms[name]; A = tb*w; X = p.addParticle(HP.ChargedScalar(name)); X.setMass(m)
        for dec, i in zip((HP.Decay.enu, HP.Decay.munu, HP.Decay.taunu), range(3)): X.setDecayWidth(dec, GL[kk, i])
        X.setDecayWidth(HP.Decay.tb, 3*GF*m*MT**2*A**2/(4*np.sqrt(2)*np.pi)*(1 - MT**2/m**2)**2 if m > MT + MB else 0.0)
        for nn, cpl in (('h', cba), ('H', sba), ('A', 1.0)): X.setDecayWidth(HP.ChainDecay.W, nn, svw(m, ms[nn], MW, G2/2*w*cpl))
        if name == 'Hp2': X.setDecayWidth(HP.ChainDecay.Z, 'Hp1', svw(m, ms['Hp1'], MZ, gZ*c*s))
        brt = A**2*(1 - m**2/MT**2)**2/((1 - MW**2/MT**2)**2*(1 + 2*MW**2/MT**2)) if m < MT - MB else 0.0; brt = brt/(1 + brt)
        X.setCxn(HP.Collider.LHC13, HP.Production.Hpmtb, HP.EffectiveCouplingCxns.ppHpmtb(HP.Collider.LHC13, m, A, A, brt))
        if brt > 0: X.setCxn(HP.Collider.LHC13, HP.Production.brtHpb, brt)
        out[name + '_BRenu'] = X.br(HP.Decay.enu); out[name + '_BRmunu'] = X.br(HP.Decay.munu)
        out[name + '_BRtaunu'] = X.br(HP.Decay.taunu); out[name + '_width'] = X.totalWidth()
    for name in ('H', 'A'):
        for hp, w in (('Hp1', c), ('Hp2', s)):
            cpl = (sba if name == 'H' else 1.0)*w
            p.particle(name).setDecayWidth(HP.ChainDecay.W, hp, 2*svw(ms[name], ms[hp], MW, G2/2*cpl))
    # Drell-Yan h+ h- vs LHC slepton searches (zee/dy.py): doublet fractions cos^2(phi) for h1+, sin^2(phi) for h2+
    out['DY_allowed'], out['DY_S_fb'] = DY.dy_allowed([ms['Hp1'], ms['Hp2']], [c*c, s*s], [out['Hp1_BRenu'], out['Hp2_BRenu']], [out['Hp1_BRmunu'], out['Hp2_BRmunu']])
    r = _b(p); out['HB_allowed'] = bool(r.allowed); out['HS_chi2'] = float(_s(p)); out['rgg'] = rgg
    try: out['HB_most_sensitive'] = str(r.selectedLimits) [:200]
    except Exception: pass
    return out, e
