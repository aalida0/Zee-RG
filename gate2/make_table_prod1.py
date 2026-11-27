"""LaTeX rows of the benchmark table for the reference points (BM1, BM2; results/benchmarks.json) and the campaign benchmarks (results/prod1/benchmarks_prod1.json).
The rows of the one-loop-matching / all-Yukawa / momentum-dependent matching minima of the campaign points are read from their validation records
(results/prod1/validate_<finalist>.json, written by gate2/validate_finalist.py); those of BM1/BM2 from results/matching and results/yukawa as in gate2/make_table.py.
Usage: python3 gate2/make_table_prod1.py [BP1 BP3 ...] > /tmp/bm_rows.tex      (a point with an undefined variant in a prescription gets a dagger count, never a scale for the undefined variant)"""
import json, os, sys, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path[:0] = [ROOT, os.path.dirname(os.path.abspath(__file__))]
from accept import HSREF, DCHI2
P1 = os.path.join(ROOT, 'results', 'prod1')
B0 = json.load(open(os.path.join(ROOT, 'results', 'benchmarks.json'))); B1 = json.load(open(os.path.join(P1, 'benchmarks_prod1.json')))
SRC = json.load(open(os.path.join(P1, 'benchmarks_input_prod1.json')))          # origin string carries the finalist id
sel = sys.argv[1:] or ['BP1', 'BP3']
B = {k: v for k, v in B0.items() if isinstance(v, dict) and v.get('accepted')}; B.update({k: B1[k] for k in sel})
names = list(B)
def sci(v):
    if v == 0: return '0'
    e = int(np.floor(np.log10(abs(v)))); m = v/10**e; return '$%.1f\\times10^{%d}$' % (m, e)
def row(label, f): print(label + ' & ' + ' & '.join(f(n, B[n]) for n in names) + '\\\\')
def val(n):
    cid = SRC[n]['origin'].split('finalist ')[1].split(' ')[0]; return json.load(open(os.path.join(P1, 'validate_%s.json' % cid)))
print('% columns:', ' '.join(names))
row('$m_H,\\ m_A$ [GeV]', lambda n, r: '%.0f, %.0f' % (r['masses']['H'], r['masses']['A']))
row('$m_{h_1^+},\\ m_{h_2^+}$ [GeV]', lambda n, r: '%.0f, %.0f' % (r['masses']['h1p'], r['masses']['h2p']))
row('$\\tan\\beta$, $c_{\\beta-\\alpha}$', lambda n, r: '%.2f, %.4f' % (r['tan_beta'], r['c_beta_alpha']))
row('$\\cos\\varphi$, $\\mu/v$', lambda n, r: '%.2f, %.2f' % (np.cos(r['phi']), r['mu_over_v']))
q = lambda r, ks: ', '.join('%.2f' % r['quartics'][k] for k in ks)
row('$\\lambda_1,\\lambda_2,\\lambda_3$', lambda n, r: q(r, ['l1', 'l2', 'l3']))
row('$\\lambda_4,\\lambda_5,\\lambda_6$', lambda n, r: q(r, ['l4', 'l5', 'l6']))
row('$\\lambda_7,\\lambda_8,\\lambda_9$', lambda n, r: q(r, ['l7', 'l8', 'l9']))
row('$\\lambda_{10}$, $\\lambda_h$', lambda n, r: '%.2f, %.4f' % (r['quartics']['l10'], r['quartics']['lh']))
row('$\\max|Y_2|$, $|f_{\\mu\\tau}|$', lambda n, r: '%.3f, %s' % (max(max(abs(a), abs(b)) for a, b in r['Y2'].values()), sci(abs(r['f']['mu_tau']))))
print('\\midrule')
for lab, k in (('$\\Br(\\mu\\to e\\gamma)$', 'mu_e_g'), ('$\\Br(\\tau\\to\\mu\\gamma)$', 'tau_mu_g'), ('$\\Br(\\tau\\to e\\gamma)$', 'tau_e_g'), ('$\\Br(\\mu\\to3e)$ (dipole)', 'mu_3e'), ('$|d_e|$ [$e$\\,cm]', 'd_e')):
    row(lab, lambda n, r, k=k: sci(r['observables'][k]))
row('$\\Br(h\\to\\tau\\mu)$', lambda n, r: sci(r['collider']['BR_h_tau_mu']) if r.get('collider') else 'n/a')
row('$B\\to X_s\\gamma$ [$10^{-4}$]', lambda n, r: '%.2f' % r['bsgamma_e4'])
row('$\\chi^2$ ($\\chi^2_{\\min}=%.2f$, limit $%.2f$)' % (B[names[0]]['chi2_min'], B[names[0]]['chi2_min'] + DCHI2), lambda n, r: '%.2f' % r['chi2'])
row('$\\Delta\\chi^2_{\\rm HS}=\\chi^2_{\\rm HS}-%.2f$, HiggsBounds' % HSREF, lambda n, r: '%+.2f, %s' % (r['collider']['HS_chi2'] - HSREF, 'allowed' if r['collider']['HB_allowed'] else 'excluded'))
row('$\\sum m_\\nu$ [eV], $\\sin^2\\theta_{23}$', lambda n, r: '%.4f, %.3f' % (r['neutrino']['sum_m'], r['neutrino']['s23sq']))
row('$h_1^+$: BR($\\mu\\nu$), BR($\\tau\\nu$), BR($tb$)', lambda n, r: '%.3f, %.3f, %.3f' % (r['dy']['BR'][0]['munu'], r['dy']['BR'][0]['taunu'], r['dy']['BR'][0]['tb']))
row('Drell--Yan screen', lambda n, r: r['dy']['status'])
print('\\midrule')
rb = lambda r, p: r['robustness'][p]['log10Lambda']
row('$\\log_{10}(\\Lambda/\\mathrm{GeV})$, default inputs', lambda n, r: '%.2f' % rb(r, 'default'))
row('\\quad boundary at $\\mu_0=m_{\\rm light}/2,\\,2m_{\\rm light}$', lambda n, r: '%.1f, %.1f' % (rb(r, 'match_x0.5'), rb(r, 'match_x2')))
row('\\quad boundary at $\\mu_0=m_{\\rm heavy}/2,\\,2m_{\\rm heavy}$', lambda n, r: '%.1f, %.1f' % (rb(r, 'matchH_x0.5'), rb(r, 'matchH_x2')))
row('\\quad quartics $\\times1.1,\\,\\times0.9$', lambda n, r: '%.1f, %.1f' % (rb(r, 'qscale_1.1'), rb(r, 'qscale_0.9')))
row('\\quad $y_t\\pm2\\%$, $g_s\\pm3\\%$ (min)', lambda n, r: '%.1f, %.1f' % (min(rb(r, 'sm_yt_1.02'), rb(r, 'sm_yt_0.98')), min(rb(r, 'sm_gs_1.03'), rb(r, 'sm_gs_0.97'))))
row('minimum over the 13 prescribed variants', lambda n, r: '%.1f' % r['certified_log10_scale'])
import validate_finalist as VF
def vrec(n): return json.load(open(os.path.join(ROOT, 'results', 'validation', 'validate_%s.json' % n))) if n in B0 else val(n)       # ALL columns from the same kind of validation record
ASS = {n: VF.assess(vrec(n)) for n in names}                                                                                          # one assessment logic for every column (exact 7-prescription coverage)
rng = lambda v: '%.1f' % min(v) if '%.1f' % min(v) == '%.1f' % max(v) else '%.1f--%.1f' % (min(v), max(v))
def fmt_modes(n, modes, key):
    """range over the prescriptions `modes` of the minimum scale: key='min_rematched' (re-matched variants only) or 'min_stress' (also the qscale stress variants), defined variants only;
    the number of unbounded-at-start variants, summed over the prescriptions, is appended as (n) (it is the same for both keys, so it is printed with min_rematched only)"""
    mt = vrec(n)['matching']; got = [mt[m] for m in modes]; sc = [g[key] for g in got if g[key] is not None]; nf = sum(g['n_start_fail'] for g in got)
    return (rng(sc) if sc else 'undefined') + ('$^{(%d)}$' % nf if nf and key == 'min_rematched' else '')
SIX = ['full-sym', 'full-min', 'full-max', 'full-hpole-sym', 'full-hpole-min', 'full-hpole-max']; HP = ['full-hpole-sym', 'full-hpole-min', 'full-hpole-max']
yk = lambda n, ks: [v for k, v in vrec(n)['assessment']['yukawa_min_scale'].items() if k in ks]
row('\\quad with all Yukawa couplings evolved (tree boundary)', lambda n, r: '%.1f' % yk(n, ['tree'])[0])
for lab, modes in (('one-loop (effective-potential) matching', ['ep']), ('momentum-dependent matching, $h$ pole (3 prescriptions)', HP), ('momentum-dependent matching, all 6 prescriptions', SIX)):
    row('\\quad %s: re-matched variants' % lab, lambda n, r, m=modes: fmt_modes(n, m, 'min_rematched'))
    row('\\quad \\qquad including the stress variants', lambda n, r, m=modes: fmt_modes(n, m, 'min_stress'))
row('\\quad $h$-pole matching and all Yukawa couplings', lambda n, r: rng(yk(n, HP)))
row('complete prescriptions (of 7), status', lambda n, r: '%d, %s' % (ASS[n]['n_prescriptions_complete'], {'robust over all seven prescriptions': 'robust', 'EP-clean only (not all prescriptions complete or run)': 'EP-clean only'}.get(ASS[n]['classification'], 'not clean')))
row('conservative tree margin $\\min(V_{\\rm red},\\lambda_h)$, min. matched $V_{\\rm red}$ margin', lambda n, r: '%.3f, %.3f' % (ASS[n]['conservative_tree_margin'], ASS[n]['min_matched_bfb_margin']))
row('first failure', lambda n, r: r['robustness']['default']['fail'])
row('no deeper minimum found (1527 starts)', lambda n, r: 'EW' if r['robustness']['vacuum']['global_ok'] else 'not established')
