"""LaTeX rows of the benchmark table for the reference points (BM1, BM2; results/benchmarks.json) and the campaign benchmarks (results/prod1/benchmarks_prod1.json).
The rows of the one-loop-matching / all-Yukawa / momentum-dependent matching minima of the campaign points are read from their validation records
(results/prod1/validate_<finalist>.json, written by gate2/validate_finalist.py); those of BM1/BM2 from results/matching and results/yukawa as in gate2/make_table.py.
Usage: python3 gate2/make_table_prod1.py [BP1 BP2 ...] > /tmp/bm_rows.tex      (a point with an undefined variant in a prescription gets a dagger count, never a scale for the undefined variant)"""
import json, os, sys, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path[:0] = [ROOT, os.path.dirname(os.path.abspath(__file__))]
from accept import HSREF, DCHI2
P1 = os.path.join(ROOT, 'results', 'prod1')
B0 = json.load(open(os.path.join(ROOT, 'results', 'benchmarks.json'))); B1 = json.load(open(os.path.join(P1, 'benchmarks_prod1.json')))
SRC = json.load(open(os.path.join(P1, 'benchmarks_input_prod1.json')))          # origin string carries the finalist id
sel = sys.argv[1:] or ['BP1', 'BP2', 'BP3']
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
row('$\\lambda_{1\\ldots5}$', lambda n, r: q(r, ['l1', 'l2', 'l3', 'l4', 'l5']))
row('$\\lambda_{6\\ldots10},\\lambda_h$', lambda n, r: q(r, ['l6', 'l7', 'l8', 'l9', 'l10', 'lh']))
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
row('Drell--Yan screen (outside recast domain)', lambda n, r: r['dy']['status'])
print('\\midrule')
rb = lambda r, p: r['robustness'][p]['log10Lambda']
row('$\\log_{10}(\\Lambda/\\mathrm{GeV})$, default inputs', lambda n, r: '%.2f' % rb(r, 'default'))
row('\\quad boundary at $\\mu_0=m_{\\rm light}/2,\\,2m_{\\rm light}$', lambda n, r: '%.1f, %.1f' % (rb(r, 'match_x0.5'), rb(r, 'match_x2')))
row('\\quad boundary at $\\mu_0=m_{\\rm heavy}/2,\\,2m_{\\rm heavy}$', lambda n, r: '%.1f, %.1f' % (rb(r, 'matchH_x0.5'), rb(r, 'matchH_x2')))
row('\\quad quartics $\\times1.1,\\,\\times0.9$', lambda n, r: '%.1f, %.1f' % (rb(r, 'qscale_1.1'), rb(r, 'qscale_0.9')))
row('\\quad $y_t\\pm2\\%$, $g_s\\pm3\\%$ (min)', lambda n, r: '%.1f, %.1f' % (min(rb(r, 'sm_yt_1.02'), rb(r, 'sm_yt_0.98')), min(rb(r, 'sm_gs_1.03'), rb(r, 'sm_gs_0.97'))))
row('minimum over the 13 prescribed variants', lambda n, r: '%.1f' % r['certified_log10_scale'])
def _old(kind, n, key):
    fn = os.path.join(ROOT, 'results', kind, ('matching_%s_k1.json' % n) if kind == 'matching' else ('yukawa_%s_all.json' % n))
    try: return '%.1f' % json.load(open(fn))[key]
    except (OSError, KeyError): return 'n/a'
def _oldfull(n):
    try: m = [json.load(open(os.path.join(ROOT, 'results', 'matching', 'matching_%s_full-hpole-%s.json' % (n, k))))['min'] for k in ('sym', 'min', 'max')]
    except (OSError, KeyError): return 'n/a'
    return '%.1f--%.1f' % (min(m), max(m))
def _new_mode(n, mode):
    m = val(n)['matching'][mode]; return m['min_rematched'], m['n_start_fail'], m['n_variants']
rng = lambda v: '%.1f' % min(v) if '%.1f' % min(v) == '%.1f' % max(v) else '%.1f--%.1f' % (min(v), max(v))
def fmt_modes(n, modes):
    """range of the minimum scale over the prescriptions `modes` (defined variants only); the number of start-failed variants is appended as a dagger count"""
    got = [_new_mode(n, m) for m in modes]; sc = [g[0] for g in got if g[0] is not None]; nf = sum(g[1] for g in got)
    s = rng(sc) if sc else 'undefined'
    return s + ('$^{(%d)}$' % nf if nf else '')
isold = lambda n: n in B0
yk = lambda n, ks: [v for k, v in val(n)['assessment']['yukawa_min_scale'].items() if k in ks]
row('\\quad with all Yukawa couplings evolved', lambda n, r: _old('yukawa', n, 'min') if isold(n) else '%.1f' % yk(n, ['tree'])[0])
row('\\quad with one-loop (effective-potential) matching', lambda n, r: ('11.5' if n == 'BM1' else '12.2') if isold(n) else fmt_modes(n, ['ep']))
row('\\quad with momentum-dependent matching ($h$ pole)', lambda n, r: ('11.4--11.6' if n == 'BM1' else '11.5--12.0') if isold(n) else fmt_modes(n, ['full-hpole-sym', 'full-hpole-min', 'full-hpole-max']))
row('\\quad all six mixing prescriptions (defined variants)', lambda n, r: ('11.4--12.0' if n == 'BM1' else '11.5--12.5$^\\dagger$') if isold(n) else fmt_modes(n, ['full-sym', 'full-min', 'full-max', 'full-hpole-sym', 'full-hpole-min', 'full-hpole-max']))
row('\\quad $h$-pole matching and all Yukawa couplings', lambda n, r: ('11.4--11.6' if n == 'BM1' else '11.5--12.0') if isold(n) else rng(yk(n, ['full-hpole-sym', 'full-hpole-min', 'full-hpole-max'])))
row('first failure', lambda n, r: r['robustness']['default']['fail'])
row('no deeper minimum found (1500 starts)', lambda n, r: 'EW' if r['robustness']['vacuum']['global_ok'] else 'not established')
