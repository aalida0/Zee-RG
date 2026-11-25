"""Figures and numbers for the revised draft. Usage: ZEE_INPUTS=modern python gate2/figs2.py DRAFT/figures"""
import sys, json, pickle, numpy as np, pandas as pd
sys.path.insert(0, '/home/claude/zee_fit'); sys.path.insert(0, '/home/claude/zee_fit/gate2')
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import bench
from scipy.integrate import solve_ivp
from zee import rgfast2 as R, bfb2, vac
out = sys.argv[1]; plt.rcParams.update({'font.size': 9})
d = pickle.load(open('/tmp/sample2_df.pkl', 'rb')); LIM = 9.963 + 3.84
ok = lambda t: (t.chi2 <= LIM) & (t.HS_chi2 - 151.65 <= 3.84) & t.HB_allowed & t.hard & t.glob
s = d[ok(d) & d.HSok0].copy(); s['set'] = 'sample'
A = pickle.load(open('/tmp/allvalid.pkl', 'rb')); A = A[ok(A) & (A.chi2 <= LIM)].copy(); A['set'] = 'search'
V = pd.DataFrame([r for r in pickle.load(open('/tmp/loc_valid.pkl', 'rb')) if 'err' not in r]); V = V[ok(V)].copy(); V['set'] = 'search'
allp = pd.concat([s, A[A.src.map(lambda z: z[0] != 'sample')], V], ignore_index=True)
cols = ['mu_e_g', 'tau_mu_g', 'BR_h_tau_mu', 'Cr_Al', 'mu_3e']
s[['chi2', 'Lf_strict'] + cols + ['tb', 'mA', 'm1', 'cba']].to_csv(out + '/final_sample.csv', index=False)
allp[['set', 'chi2', 'Lf'] + cols + ['tb', 'mA', 'm1', 'cba', 'lam6', 'mu']].to_csv(out + '/validated_points.csv', index=False)
print('sample', len(s), 'search points', (allp.set == 'search').sum())
for k in cols + ['tb', 'mA', 'm1']: print(k, '%.2g %.2g | stable1e4: %.2g %.2g' % (s[k].min(), s[k].max(), s[s.Lf_strict >= 4][k].min(), s[s.Lf_strict >= 4][k].max()))
for n, col, r in [('MEG', 'mu_e_g', 6e-14), ('Mu3e', 'mu_3e', 1e-16), ('Mu2eI', 'Cr_Al', 6.2e-16), ('Mu2e', 'Cr_Al', 6.1e-17), ('Mu2eII', 'Cr_Al', 6e-18), ('tmg', 'tau_mu_g', 1e-9), ('h', 'BR_h_tau_mu', 1e-4)]:
    print(n, (s[col] > r).sum(), (s[s.Lf_strict >= 4][col] > r).sum(), 'of', len(s), (s.Lf_strict >= 4).sum())
print('stable counts', {L: int((s.Lf_strict >= L).sum()) for L in [3, 4, 6, 8, 10, 12, 16]})
# Fig 1: highest demonstrated cutoff vs |c_{beta-alpha}| coloured by Br(h->tau mu)
fig, ax = plt.subplots(figsize=(3.7, 3.2))
for st, mk in (('sample', 'o'), ('search', '^')):
    t = allp[allp.set == st]; scv = ax.scatter(np.abs(t.cba), t.Lf, c=np.log10(t.BR_h_tau_mu), s=9 if st == 'sample' else 12, marker=mk, cmap='viridis', vmin=-7, vmax=-3, label=st, rasterized=True)
B = json.load(open('/tmp/benchmarks.json'))
for n, r in B.items(): ax.annotate(n, (abs(r['c_beta_alpha']), r['robustness']['default']['log10Lambda']), fontsize=7, xytext=(3, 3), textcoords='offset points')
ax.set_xscale('log'); ax.set_xlabel(r'$|c_{\beta-\alpha}|$'); ax.set_ylabel(r'$\log_{10}(\Lambda/{\rm GeV})$ demonstrated'); ax.legend(fontsize=6, loc='lower left')
fig.colorbar(scv, label=r'$\log_{10}{\rm Br}(h\to\tau\mu)$'); fig.tight_layout(); fig.savefig(out + '/fig_cutoff_cba.pdf', dpi=300)
# Fig 2: RG trajectories (max|lambda|, unitarity eigenvalue/8pi, BFB margin) for BM1, BM3 and a low-cutoff sample point
fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.6), sharex=True)
low = s[(s.Lf_strict > 3.5) & (s.Lf_strict < 5)].iloc[0]
cases = [(n, np.array(B[n]['x']), np.array(B[n]['eta'])) for n in ('BM1', 'BM3')] + [('low-cutoff point', np.array(low.x), np.array(low.eta))]
for lab, x, eta in cases:
    sc, c = bench.coup(x, eta); t0 = np.log(min(sc['m1'], sc['mA'], sc['mH'])); g = np.array([0.36, 0.64, 1.05, 0.88])
    sol = solve_ivp(R.rhs_r(-sc['tb']), (t0, np.log(1e16)), np.concatenate([g, c]), dense_output=True, rtol=1e-8, atol=1e-10)
    ts = np.linspace(t0, sol.t[-1], 120); mx, un, bf = [], [], []
    for t in ts:
        cc = sol.sol(t)[4:]; mx.append(np.abs(cc).max()); un.append(vac.unit_max(cc)/(8*np.pi)); bf.append(bfb2.min_vred(cc, refine=1))
        if mx[-1] > 4*np.pi or un[-1] > 1.2: break
    L10 = ts[:len(mx)]/np.log(10)
    axs[0].plot(L10, mx, label=lab); axs[1].plot(L10, un); axs[2].plot(L10, bf)
axs[0].set_ylabel(r'$\max_i|\lambda_i|$'); axs[1].set_ylabel(r'$\max|\Lambda_{2\to2}|/8\pi$'); axs[2].set_ylabel(r'$\min V_{\rm red}$')
axs[1].axhline(1, c='k', lw=.7, ls='--'); axs[2].axhline(0, c='k', lw=.7, ls='--'); axs[0].legend(fontsize=6)
for a in axs: a.set_xlabel(r'$\log_{10}(\mu/{\rm GeV})$')
fig.tight_layout(); fig.savefig(out + '/fig_trajectories.pdf', dpi=300)
# Fig 3: scatter in Br(mu->e gamma) vs Br(tau->mu gamma), sample vs validated search points coloured by cutoff
fig, ax = plt.subplots(figsize=(3.7, 3.2))
for st, mk in (('sample', 'o'), ('search', '^')):
    t = allp[allp.set == st]; scv = ax.scatter(t.tau_mu_g, t.mu_e_g, c=t.Lf, s=9 if st == 'sample' else 12, marker=mk, cmap='plasma', vmin=2, vmax=16, rasterized=True, label=st)
for v, l in ((4.2e-8, '-'), (1e-9, '--')): ax.axvline(v, c='k', lw=.7, ls=l)
for v, l in ((1.5e-13, '-'), (6e-14, '--')): ax.axhline(v, c='k', lw=.7, ls=l)
for n, r in B.items(): ax.annotate(n, (r['clfv']['tau_mu_g'], r['clfv']['mu_e_g']), fontsize=7, xytext=(3, 3), textcoords='offset points')
ax.set_xscale('log'); ax.set_yscale('log'); ax.set_ylim(1e-20, 1e-12); ax.set_xlabel(r'${\rm Br}(\tau\to\mu\gamma)$'); ax.set_ylabel(r'${\rm Br}(\mu\to e\gamma)$'); ax.legend(fontsize=6)
fig.colorbar(scv, label=r'$\log_{10}(\Lambda/{\rm GeV})$'); fig.tight_layout(); fig.savefig(out + '/fig_lfv_cutoff.pdf', dpi=300)
