"""LaTeX rows of the benchmark table from results/benchmarks.json (no manual transcription).
Usage: python3 gate2/make_table.py > /tmp/bm_rows.tex"""
import json, os, sys, numpy as np
sys.path[:0] = [os.path.dirname(os.path.dirname(os.path.abspath(__file__))), os.path.dirname(os.path.abspath(__file__))]
from accept import HSREF, DCHI2      # reference HiggsSignals chi2 and acceptance margin of the gate
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
B = json.load(open(os.path.join(ROOT, 'results', 'benchmarks.json'))); names = [k for k, v in B.items() if v.get('accepted')]
def sci(v):
    if v == 0: return '0'
    e = int(np.floor(np.log10(abs(v)))); m = v/10**e; return '$%.1f\\times10^{%d}$' % (m, e)
def row(label, f): print(label + ' & ' + ' & '.join(f(B[n]) for n in names) + '\\\\')
print('% columns:', ' '.join(names))
row('$m_H,\\ m_A$ [GeV]', lambda r: '%.0f, %.0f' % (r['masses']['H'], r['masses']['A']))
row('$m_{h_1^+},\\ m_{h_2^+}$ [GeV]', lambda r: '%.0f, %.0f' % (r['masses']['h1p'], r['masses']['h2p']))
row('$\\tan\\beta$, $c_{\\beta-\\alpha}$', lambda r: '%.2f, %.4f' % (r['tan_beta'], r['c_beta_alpha']))
row('$\\cos\\varphi$, $\\mu/v$', lambda r: '%.2f, %.2f' % (np.cos(r['phi']), r['mu_over_v']))
q = lambda r, ks: ', '.join('%.2f' % r['quartics'][k] for k in ks)
row('$\\lambda_{1\\ldots5}$', lambda r: q(r, ['l1', 'l2', 'l3', 'l4', 'l5']))
row('$\\lambda_{6\\ldots10},\\lambda_h$', lambda r: q(r, ['l6', 'l7', 'l8', 'l9', 'l10', 'lh']))
row('$\\max|Y_2|$, $|f_{\\mu\\tau}|$', lambda r: '%.3f, %s' % (max(max(abs(a), abs(b)) for a, b in r['Y2'].values()), sci(abs(r['f']['mu_tau']))))
print('\\midrule')
for lab, k in (('$\\Br(\\mu\\to e\\gamma)$', 'mu_e_g'), ('$\\Br(\\tau\\to\\mu\\gamma)$', 'tau_mu_g'), ('$\\Br(\\tau\\to e\\gamma)$', 'tau_e_g'), ('$\\Br(\\mu\\to3e)$ (dipole)', 'mu_3e'), ('$|d_e|$ [$e$\\,cm]', 'd_e')):
    row(lab, lambda r, k=k: sci(r['observables'][k]))
row('$\\Br(h\\to\\tau\\mu)$', lambda r: sci(r['collider']['BR_h_tau_mu']) if r.get('collider') else 'n/a')
row('$B\\to X_s\\gamma$ [$10^{-4}$]', lambda r: '%.2f' % r['bsgamma_e4'])
row('$\\chi^2$ ($\\chi^2_{\\min}=%.2f$, limit $%.2f$)' % (B[names[0]]['chi2_min'], B[names[0]]['chi2_min'] + DCHI2), lambda r: '%.2f' % r['chi2'])
row('$\\Delta\\chi^2_{\\rm HS}=\\chi^2_{\\rm HS}-%.2f$, HiggsBounds' % HSREF, lambda r: '%+.2f, %s' % (r['collider']['HS_chi2'] - HSREF, 'allowed' if r['collider']['HB_allowed'] else 'excluded'))
row('$\\sum m_\\nu$ [eV], $\\sin^2\\theta_{23}$', lambda r: '%.4f, %.3f' % (r['neutrino']['sum_m'], r['neutrino']['s23sq']))
row('$h_1^+$: BR($\\mu\\nu$), BR($\\tau\\nu$), BR($tb$)', lambda r: '%.3f, %.3f, %.3f' % (r['dy']['BR'][0]['munu'], r['dy']['BR'][0]['taunu'], r['dy']['BR'][0]['tb']))
row('Drell--Yan screen (outside recast domain)', lambda r: r['dy']['status'])
print('\\midrule')
rb = lambda r, p: r['robustness'][p]['log10Lambda']
row('$\\log_{10}(\\Lambda/\\mathrm{GeV})$, default inputs', lambda r: '%.2f' % rb(r, 'default'))
row('\\quad boundary at $\\mu_0=m_{\\rm light}/2,\\,2m_{\\rm light}$', lambda r: '%.1f, %.1f' % (rb(r, 'match_x0.5'), rb(r, 'match_x2')))
row('\\quad boundary at $\\mu_0=m_{\\rm heavy}/2,\\,2m_{\\rm heavy}$', lambda r: '%.1f, %.1f' % (rb(r, 'matchH_x0.5'), rb(r, 'matchH_x2')))
row('\\quad quartics $\\times1.1,\\,\\times0.9$', lambda r: '%.1f, %.1f' % (rb(r, 'qscale_1.1'), rb(r, 'qscale_0.9')))
row('\\quad $y_t\\pm2\\%$, $g_s\\pm3\\%$ (min)', lambda r: '%.1f, %.1f' % (min(rb(r, 'sm_yt_1.02'), rb(r, 'sm_yt_0.98')), min(rb(r, 'sm_gs_1.03'), rb(r, 'sm_gs_0.97'))))
row('minimum over the 13 prescribed variants', lambda r: '%.1f' % r['certified_log10_scale'])
def _aux(kind, r, key):
    """minimum over the 13 variants from the one-loop matching (results/matching) or the all-Yukawa run (results/yukawa), if present"""
    names = {id(v): k for k, v in B.items() if isinstance(v, dict) and 'robustness' in v}
    nm = names[id(r)]
    fn = os.path.join(ROOT, 'results', kind, ('matching_%s_k1.json' % nm) if kind == 'matching' else ('yukawa_%s_all.json' % nm))
    try: return '%.1f' % json.load(open(fn))['min']
    except (OSError, KeyError): return 'n/a'
row('\\quad with all Yukawa couplings evolved', lambda r: _aux('yukawa', r, 'min'))
row('\\quad with one-loop (effective-potential) matching', lambda r: _aux('matching', r, 'min'))
def _full(r):
    """range of the minimum scale over the three charged-pair prescriptions of the full (momentum-dependent) matching with the h-pole prescription for the CP-even mixing"""
    names = {id(v): k for k, v in B.items() if isinstance(v, dict) and 'robustness' in v}; nm = names[id(r)]
    try: m = [json.load(open(os.path.join(ROOT, 'results', 'matching', 'matching_%s_full-hpole-%s.json' % (nm, k))))['min'] for k in ('sym', 'min', 'max')]
    except (OSError, KeyError): return 'n/a'
    return '%.1f--%.1f' % (min(m), max(m))
row('\\quad with momentum-dependent matching ($h$ pole)', _full)
row('first failure', lambda r: r['robustness']['default']['fail'])
row('no deeper minimum found (1500 starts)', lambda r: 'EW' if r['robustness']['vacuum']['global_ok'] else 'not established')
