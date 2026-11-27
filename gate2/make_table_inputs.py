"""LaTeX rows of the two reference tables of the draft (corrected oblique-T statistic): Table `tab:inp` (input variations, results/input_sensitivity.json) and Table `tab:ref`
(free-quartic box, refit_hw_cT_eta*.json).  Usage: python3 gate2/make_table_inputs.py > /tmp/inputs_rows.tex"""
import json, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); R = lambda f: json.load(open(os.path.join(ROOT, 'results', f)))
I = R('input_sensitivity.json'); P = ['BM1', 'BM2', 'BP1', 'BP3c']
lab = {'primary': 'Primary: RPP~2026 oblique fit, exact $z_{0.975}$, DESI-DR2', 'oblique2016': 'Oblique fit of Ref.~\\cite{HerreroGarcia:2017xdq} (2016) instead', 'z95_1.949': '$z=1.949$ instead of $1.95996$',
       'desi2024': '$\\sum m_\\nu<0.072$~eV (DESI 2024) instead of DR2', 'nocosmo': 'Without the cosmological term', 'frozen': 'All three frozen inputs (2016 oblique fit, $z=1.949$, DESI 2024)'}
print('% tab:inp')
for k in lab: print('%s & $%.2f$ & %s\\\\' % (lab[k], I[k]['chi2_min'], ' & '.join('$%.2f$' % I[k]['margin'][p] for p in P)))
print('% tab:ref')
base = I['primary']['chi2']
for lab2, fn in (('$|\\lambda_{7,8,10}|\\le3$ (fit, box edge)', 'refit_hw_cT_all.json'), ('$|\\lambda_{7,8,10}|\\le1$', 'refit_hw_cT_eta1.0.json'), ('$|\\lambda_{7,8,10}|\\le0.75$', 'refit_hw_cT_eta0.75.json')):
    m = R(fn)['best']['chi2']; print('%s & $%.2f$ & %s\\\\' % (lab2, m, ' & '.join('$%.2f$' % (m + 3.84 - base[p]) for p in P)))
