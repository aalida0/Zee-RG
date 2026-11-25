"""Benchmark records (YAML-like JSON) with all observables, couplings, and amplitude decompositions."""
import sys, json, pickle, numpy as np
sys.path.insert(0, '/home/claude/zee_fit'); sys.path.insert(0, '/home/claude/zee_fit/gate1'); sys.path.insert(0, '/home/claude/zee_fit/gate2')
import hcoll, postfit, bench
from zee import hgmodel as H, neutrino as N
BM = {'BM1': '/tmp/loc/loc_mu_e_g_12_0.json', 'BM2': '/tmp/loc2/loc_mu_e_g_16_1.json', 'BM3': '/tmp/envres2/out2/env_h_tau_mu_12_23.json'}
def decomp(i, j, x):
    e = H.evaluate(list(x), 'NO'); sc = e['sc']; f, Y = H.unpack(np.asarray(x)); Lam, Dd = H.lepton_couplings(Y, sc)
    mass = dict(h=H.MH, H=sc['mH'], A=sc['mA']); bz0 = H.bz_matrix(Lam, mass, sc); z = np.zeros_like(bz0)
    n1 = H.dipoles(i, j, Lam, mass, z); bz = tuple(a - b for a, b in zip(H.dipoles(i, j, Lam, mass, bz0), n1)); ch = H.charged_c(i, j, f, Dd, sc)
    tot = [n1[k] + bz[k] + ch[k] for k in range(2)]; T = abs(tot[0])**2 + abs(tot[1])**2
    return {'neutral_1loop': float((abs(n1[0])**2 + abs(n1[1])**2)/T), 'barr_zee': float((abs(bz[0])**2 + abs(bz[1])**2)/T), 'charged': float((abs(ch[0])**2 + abs(ch[1])**2)/T)}
out = {}
for name, fn in BM.items():
    b = json.load(open(fn))['best']; x = np.array(b['x']); eta = np.array(b['eta'])
    col, e = hcoll.run(x, eta); sc = e['sc']; o = e['obs']; f, Y = H.unpack(x); _, c = bench.coup(x, eta)
    M = N.mnu_hg_leading(f, H.M_LEP, Y, sc['tb'], sc['m1'], sc['m2'], sc['mu']); osc = N.oscillation_observables(M, 'NO')
    rob = {p: pickle.load(open(f'/tmp/bench/{name}_{p}.pkl', 'rb')) for p in ['default', 'match_x0.5', 'match_x2', 'rtol_1e-6', 'rtol_1e-10', 'grid_300', 'vacuum']}
    out[name] = dict(source=fn, x=x.tolist(), eta=eta.tolist(), chi2=float(e['chi2']),
        masses=dict(h=125.5, H=sc['mH'], A=sc['mA'], h1p=sc['m1'], h2p=sc['m2']), tan_beta=sc['tb'], c_beta_alpha=sc['cba'], phi=sc['phi'], mu_over_v=sc['mu']/H.V,
        quartics={k: float(v) for k, v in zip(['l1', 'l2', 'l3', 'l4', 'l5', 'l6', 'l7', 'l8', 'l9', 'l10', 'lh'], c)},
        f={'e_tau': float(f[0, 2].real), 'mu_tau': float(f[1, 2].real)}, Y2_abs_max=float(np.abs(Y).max()), Y2={f'{a}{b}': [float(Y[i, j].real), float(Y[i, j].imag)] for i, a in enumerate('emt') for j, b in enumerate('emt') if abs(Y[i, j]) > 0},
        neutrino=dict(m_light=float(osc['masses_eV'].min()), sum_m=float(osc['masses_eV'].sum()), s23sq=osc['s23sq'], delta=osc['delta_deg'] % 360, mbb=float(abs(M[0, 0])/M.__abs__().max()*0) if False else None),
        clfv={k: float(o[k]) for k in ['mu_e_g', 'tau_mu_g', 'tau_e_g', 'mu_3e', 'Cr_Al', 'Cr_Au', 'tau_3mu', 'tau_mumue', 'd_e', 'd_mu', 'das_mu', 'da_e']},
        higgs=dict(BR_h_tau_mu=col['BR_h_tau_mu'], BR_h_tau_e=col['BR_h_tau_e'], Gamma_h_MeV=col['Gamma_h']*1e3, HS_chi2=col['HS_chi2'], HB_allowed=col['HB_allowed'], rgg=col['rgg']),
        bsgamma_e4=float(postfit.bsg(sc)), dipole_decomposition=dict(mu_e_g=decomp(0, 1, x), tau_mu_g=decomp(1, 2, x)),
        robustness={p: ({k: (float(v) if isinstance(v, (float, np.floating)) else v) for k, v in r.items()}) for p, r in rob.items()})
json.dump(out, open('/tmp/benchmarks.json', 'w'), indent=1, default=float)
for n, r in out.items():
    print(n, 'chi2 %.2f' % r['chi2'], {k: '%.3g' % v for k, v in r['masses'].items()}, 'tb %.2f cba %.3f mu/v %.2f phi %.2f l6 %.2f' % (r['tan_beta'], r['c_beta_alpha'], r['mu_over_v'], r['phi'], r['quartics']['l6']))
    print('   clfv', {k: '%.2g' % v for k, v in r['clfv'].items()}, 'h', '%.2g' % r['higgs']['BR_h_tau_mu'], 'bsg %.2f' % r['bsgamma_e4'])
    print('   decomp', {k: {a: round(b, 2) for a, b in v.items()} for k, v in r['dipole_decomposition'].items()}, 'Ymax %.2g' % r['Y2_abs_max'], 'f', r['f'])
    print('   L', {p: round(v.get('log10Lambda', np.nan), 2) for p, v in r['robustness'].items() if p != 'vacuum'}, 'nu', {k: (round(v, 4) if v is not None else None) for k, v in r['neutrino'].items()})
    print('   quartics', {k: round(v, 2) for k, v in r['quartics'].items()})
