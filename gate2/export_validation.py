"""Write the per-prescription matching and all-Yukawa files that gate2/make_table.py reads (results/matching, results/yukawa) from a validate_<NAME>.json.
Usage: python3 gate2/export_validation.py validate_NAME.json"""
import sys, os, json
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
v = json.load(open(sys.argv[1])); nm = v['name']
os.makedirs(os.path.join(ROOT, 'results', 'matching'), exist_ok=True); os.makedirs(os.path.join(ROOT, 'results', 'yukawa'), exist_ok=True)
for mode, m in v['matching'].items():
    fn = 'matching_%s_k1.json' % nm if mode == 'ep' else 'matching_%s_%s.json' % (nm, mode)
    json.dump(dict(name=nm, mode=mode, kappa=1.0, **{k: m[k] for k in m if k != 'variants'}, variants=m['variants']), open(os.path.join(ROOT, 'results', 'matching', fn), 'w'), indent=1)
for key, y in v['yukawa'].items():
    fn = 'yukawa_%s_all.json' % nm if key == 'tree' else 'yukawa_%s_all_%s.json' % (nm, key)
    json.dump(dict(name=nm, mode='all', match=y['match'], min=y['min'], variants=y['variants']), open(os.path.join(ROOT, 'results', 'yukawa', fn), 'w'), indent=1)
os.makedirs(os.path.join(ROOT, 'results', 'validation'), exist_ok=True)
json.dump(v, open(os.path.join(ROOT, 'results', 'validation', os.path.basename(sys.argv[1])), 'w'), indent=1)
print('exported', nm, sorted(v['matching']), sorted(v['yukawa']))
