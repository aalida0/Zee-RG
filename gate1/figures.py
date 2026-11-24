"""Final-sample figures for the draft (needs HiggsTools datasets in /tmp, ZEE_INPUTS=modern). Usage: python gate1/figures.py OUTDIR"""
import sys, pickle, numpy as np
sys.path.insert(0, '/home/claude/zee_fit'); sys.path.insert(0, '/home/claude/zee_fit/gate1')
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import ht
out = sys.argv[1]
d = pickle.load(open('/tmp/mfilt4w_NO.pkl', 'rb')); res = d['res']; S = d['S']
LM = {t[0]: t for t in pickle.load(open('/tmp/lmax_final.pkl', 'rb'))}
rows = []
for i, r in res:
    x = list(S[i][1]); chi, ok, e = ht.run(x); o = e['obs']; sc = e['sc']
    rows.append((chi, ok, LM[i][2] >= 4, LM[i][2] >= 8, o['tau_mu_g'], o['mu_e_g'], o['Cr_Al'], o['mu_3e'], o['h_tau_mu'], sc['mA'], sc['tb'], sc['lam4'], sc['m1'], LM[i][1], LM[i][2]))
A = np.array(rows, float); keep = ((A[:, 0] - A[:, 0].min()) <= 3.84) & (A[:, 1] == 1); A = A[keep]
np.savetxt(out + '/final_sample.csv', A, delimiter=',', header='HSchi2,HBok,stable1e4_opt,stable1e8_opt,Br_tau_mu_gamma,Br_mu_e_gamma,CR_Al,Br_mu3e,Br_h_tau_mu,mA,tanb,lam4,m_h1+,logLmax_random,logLmax_opt', comments='')
st = A[:, 2] > 0; s8 = A[:, 3] > 0
plt.rcParams.update({'font.size': 9, 'axes.linewidth': 0.8})
def base(ax, x, y):
    ax.scatter(A[~st, x], A[~st, y], s=5, c='0.75', label=r'unstable below $10^4$ GeV', rasterized=True)
    ax.scatter(A[st, x], A[st, y], s=5, c='C0', label=r'stable to $10^4$ GeV', rasterized=True)
    ax.scatter(A[s8, x], A[s8, y], s=9, c='C3', marker='s', label=r'stable to $10^8$ GeV', rasterized=True)
    ax.set_xscale('log'); ax.set_yscale('log')
def line(ax, v, ax_='x', ls='-', lab=None):
    (ax.axvline if ax_ == 'x' else ax.axhline)(v, c='k', ls=ls, lw=0.8)
fig, ax = plt.subplots(figsize=(3.6, 3.3)); base(ax, 4, 8)
line(ax, 4.2e-8, 'x'); line(ax, 1e-9, 'x', '--'); line(ax, 1.8e-3, 'y'); line(ax, 1e-4, 'y', '--')
ax.set_xlabel(r'${\rm Br}(\tau\to\mu\gamma)$'); ax.set_ylabel(r'${\rm Br}(h\to\tau\mu)$'); ax.set_ylim(1e-9, 1e-2); ax.legend(fontsize=6, loc='lower right'); fig.tight_layout(); fig.savefig(out + '/fig_tau_h.pdf', dpi=300)
fig, ax = plt.subplots(figsize=(3.6, 3.3)); base(ax, 5, 6)
line(ax, 1.5e-13, 'x'); line(ax, 6e-14, 'x', '--'); line(ax, 6.1e-17, "y", "--")
ax.set_xlabel(r'${\rm Br}(\mu\to e\gamma)$'); ax.set_ylabel(r'${\rm CR}(\mu\,{\rm Al}\to e\,{\rm Al})$'); ax.set_xlim(1e-20, 1e-12); ax.set_ylim(1e-20, 1e-14); fig.tight_layout(); fig.savefig(out + '/fig_mue.pdf', dpi=300)
fig, ax = plt.subplots(figsize=(3.6, 3.3)); ax.scatter(A[~st, 9], A[~st, 10], s=5, c='0.75', rasterized=True); ax.scatter(A[st, 9], A[st, 10], s=5, c='C0', rasterized=True); ax.scatter(A[s8, 9], A[s8, 10], s=9, c='C3', marker='s', rasterized=True)
ax.set_xscale('log'); ax.set_xlabel(r'$m_A$ [GeV]'); ax.set_ylabel(r'$\tan\beta$'); fig.tight_layout(); fig.savefig(out + '/fig_mass.pdf', dpi=300)
fig, ax = plt.subplots(figsize=(3.6, 3.0)); sc = [3, 4, 8, 12, 16]; lab = [r'$10^3$', r'$10^4$', r'$10^8$', r'$10^{12}$', r'$10^{16}$']
ax.plot(sc, [100 * np.mean(A[:, 14] >= x) for x in sc], 'o-', label='targeted search'); ax.plot(sc, [100 * np.mean(A[:, 13] >= x) for x in sc], 's--', label='100 random draws')
ax.set_xticks(sc); ax.set_xticklabels(lab); ax.set_yscale('symlog', linthresh=0.1); ax.set_ylim(0, 110); ax.set_xlabel(r'cutoff $\Lambda$ [GeV]'); ax.set_ylabel(r'points stable to $\Lambda$ [\%]'); ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(out + '/fig_stability.pdf', dpi=300)
for x in sc: print(x, 100 * np.mean(A[:, 14] >= x), 100 * np.mean(A[:, 13] >= x))
for nm, m in (('all', A[:, 0] > -1), ('stable1e4', st)): print(nm, m.sum(), A[m, 10].min(), A[m, 10].max(), A[m, 9].min(), A[m, 9].max(), A[m, 4].min(), A[m, 4].max(), A[m, 5].max(), A[m, 6].max(), A[m, 7].max(), A[m, 8].max())
print('N kept', len(A), 'stable1e4', st.sum(), 'stable1e8', s8.sum())
