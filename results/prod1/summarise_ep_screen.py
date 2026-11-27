import json,glob,os
S=json.load(open('/home/claude/prod1b/collect_prod1b/campaign_summary.json')); V=S['verified']
F=json.load(open('/home/claude/prod1b/collect_prod1b/campaign_finalists.json'))
import numpy as np
key=lambda x:tuple(np.round(x,12)); idx={key(v['x']):v for v in V}
rows=[]
for f in sorted(glob.glob('/home/claude/prod1b/val_ep/validate_C*.json'),key=lambda s:int(os.path.basename(s)[10:-5])):
    n=os.path.basename(f)[9:-5]; d=json.load(open(f)); m=d['matching']['ep']; v=idx[key(F[n]['x'])]
    fails=[(k,(round(x['log10Lambda'],1) if x['log10Lambda'] is not None else None),x['fail']) for k,x in m['variants'].items() if x['fail']=='BFB' or x['log10Lambda'] is None or x['log10Lambda']<12]
    rows.append((n,d['tree']['min'],m['min_rematched'],m['min_stress'],d['margins']['vred'],v['obs'],(m['n_start_fail'],m['complete'],[(k,l,f) for k,l,f in fails if f!='start:BFB'])))
print('%-4s %6s %7s %7s %6s | %9s %9s %9s | failing variants (<12 or BFB)'%('id','tree','ep_min','stress','vred','mu->eg','tau->mug','h->tm'))
f2=lambda x:'%7.2f'%x if x is not None else '   None'
for n,t,a,b,vr,o,fl in rows: print('%-4s %6.2f %s %s %6.3f | %9.2e %9.2e %9.2e | startfail=%d complete=%s other=%s'%(n,t,f2(a),f2(b),vr,o['mu_e_g'],o['tau_mu_g'],o['h_tau_mu'],fl[0],fl[1],[(k,l) for k,l,f in fl[2] if f!='unitarity'] or '-'))
print(len(rows),'done')
