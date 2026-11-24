import sys, pickle, numpy as np
sys.path.insert(0, '/home/claude/zee_fit')
from zee import hgmodel as H, rge, neutrino as N
V = H.V; ML = H.M_LEP
G0 = np.array([0.36, 0.64, 1.05]); YT0 = 0.88
def HB(x):
    sc = H.scalar_sector(x); f, Y = H.unpack(x); cb = 1/np.sqrt(1+sc['tb']**2)
    Y2 = -np.sqrt(2)*sc['tb']*np.diag(ML)/V + Y/cb
    return sc, f, Y, Y2
def back(sc, Y2):
    cb = 1/np.sqrt(1+sc['tb']**2); return cb*(Y2 + np.sqrt(2)*sc['tb']*np.diag(ML)/V)
def obs(sc, f, Y, ordering):
    M = N.mnu_hg_leading(f, ML, Y, sc['tb'], sc['m1'], sc['m2'], sc['mu'])
    o = N.oscillation_observables(M, ordering); t = H.OSC[ordering]
    dd = (o['delta_deg']-t['d'][0]+180.) % 360.-180.
    r = np.array([(o['s12sq']-t['s12'][0])/t['s12'][1], (o['s13sq']-t['s13'][0])/t['s13'][1], (o['s23sq']-t['s23'][0])/t['s23'][1],
                  (o['dm21_eV2']-t['dm21'][0])/t['dm21'][1], (o['dm3l_eV2']-t['dm3'][0])/t['dm3'][1], dd/t['d'][1]])
    Lam, Dd = H.lepton_couplings(Y, sc); mass = {'h': H.MH, 'H': sc['mH'], 'A': sc['mA']}
    out = {'chi_osc': float((r**2).sum())}
    for nm, (i, j) in {'mu_e_g': (0, 1), 'tau_mu_g': (1, 2)}.items():
        cl, cr = H.dipoles(i, j, Lam, mass, np.zeros((3, 3), complex)); clc, crc = H.charged_c(i, j, f, Dd, sc)
        out[nm] = (abs(cl+clc)**2+abs(cr+crc)**2)*(1.0 if nm == 'mu_e_g' else H.BR_TAU_LNUNU[i])*48*np.pi**3*H.ALPHA/H.GF**2
    return out
def landau(Y1, Y2, f, t0, ymax):
    s = rge.run(Y1, Y2, f, YT0, G0, t0, np.log(1e19), ymax)
    return float(np.exp(s.t[-1])) if s.status == 1 else 1e19
if __name__ == '__main__':
    ordering = sys.argv[1]; pks = sys.argv[2].split(','); dchi = float(sys.argv[3])
    pts = []
    for p in pks: pts += pickle.load(open(p, 'rb'))
    cmin = min(c for c, _ in pts); pts = [(c, x) for c, x in pts if c <= cmin+dchi]
    # dedupe near-identical
    uniq = []
    for c, x in pts:
        if all(np.abs(x-u).max() > 1e-4 for _, u in uniq): uniq.append((c, x))
    print('chi2min', round(cmin, 2), 'points', len(uniq), flush=True)
    rows = []
    for c, x in uniq:
        sc, f, Y, Y2 = HB(x); mu0 = min(sc['m1'], sc['mA'], sc['mH']); t0 = np.log(mu0)
        Y1 = np.diag(np.sqrt(2)*ML/V).astype(complex)
        row = dict(chi2=c, mu0=mu0, ymax=max(np.abs(Y2).max(), np.abs(2*f).max()))
        row['LP_sqrt4pi'] = landau(Y1, Y2, f, t0, np.sqrt(4*np.pi)); row['LP_4pi'] = landau(Y1, Y2, f, t0, 4*np.pi)
        base = obs(sc, f, Y, ordering); row['base'] = base
        for Lam in (1e4, 1e8, 1e16):
            if row['LP_4pi'] < Lam: row[Lam] = None; continue
            s = rge.run(Y1, Y2, f, YT0, G0, t0, np.log(Lam))
            Y1u, Y2u, fu, ytu, gu = rge.unpack(s.y[:, -1])
            for (i, j) in ((0, 0), (0, 1), (1, 0), (1, 1)): Y2u[i, j] = -np.sqrt(2)*sc['tb']*ML[i]/V if i == j else 0
            fu[0, 1] = fu[1, 0] = 0
            s2 = rge.run(Y1u, Y2u, fu, ytu, gu, np.log(Lam), t0)
            Y1d, Y2d, fd, _, _ = rge.unpack(s2.y[:, -1])
            Yd = back(sc, Y2d); new = obs(sc, fd, Yd, ordering)
            row[Lam] = dict(reg=float(max(abs(Yd[0, 0]), abs(Yd[0, 1]), abs(Yd[1, 0]), abs(Yd[1, 1]))), regf=float(abs(fd[0, 1])),
                            dY=float(np.abs(Y2d-Y2).max()), df=float(np.abs(fd-f).max()), dchi_osc=new['chi_osc']-base['chi_osc'],
                            mueg_ratio=new['mu_e_g']/max(base['mu_e_g'], 1e-40), taumug_ratio=new['tau_mu_g']/max(base['tau_mu_g'], 1e-40))
        rows.append(row)
    pickle.dump(rows, open(sys.argv[4], 'wb'))
    for k in ('LP_sqrt4pi', 'LP_4pi'):
        v = np.array([r[k] for r in rows]); print(k, 'min/median/max scale', f'{v.min():.2e} {np.median(v):.2e} {v.max():.2e}',
              'fraction pert. up to 1e4,1e8,1e16:', [round(float((v > L).mean()), 2) for L in (1e4, 1e8, 1e16)])
    for Lam in (1e4, 1e8, 1e16):
        rr = [r[Lam] for r in rows if r[Lam]]
        if rr:
            print('Lambda', f'{Lam:.0e}', 'n', len(rr), 'max regenerated Y2(ee,emu,mue,mumu):', f"{max(r['reg'] for r in rr):.2e}", 'f_emu:', f"{max(r['regf'] for r in rr):.2e}",
                  'max |dchi_osc|', f"{max(abs(r['dchi_osc']) for r in rr):.2e}", 'mu->egamma ratio range', f"{min(r['mueg_ratio'] for r in rr):.2g}-{max(r['mueg_ratio'] for r in rr):.2g}",
                  'tau->mugamma ratio', f"{min(r['taumug_ratio'] for r in rr):.3g}-{max(r['taumug_ratio'] for r in rr):.3g}")
