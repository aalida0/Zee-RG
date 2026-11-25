_R = __import__('os').path.dirname(__import__('os').path.dirname(__import__('os').path.abspath(__file__)))
import sys,pickle; sys.path.insert(0,_R + '/gate1'); sys.path.insert(0,_R)
import numpy as np, gate1 as G
from zee import observables as O, dipoles as P
rng=np.random.default_rng(23)
def draw():
    x=G.LO+(G.HI-G.LO)*rng.random(len(G.LO))
    for k in range(5):
        r=10**rng.uniform(-4,-1); th=rng.uniform(0,2*np.pi); x[2+2*k]=r*np.cos(th); x[3+2*k]=r*np.sin(th)
    x[0:2]=rng.choice([-1,1],2)*10**rng.uniform(-4,-1,2); return x
b=G.run("NO",starts=[draw() for _ in range(1500)])
rows=[]
for a,c,x in b:
    if c>8: break
    br,phi=G.bound_res(x,True)
    stu=br[-3:]; chi=a+float(np.sum(br[:-3]**2))*0+float(np.sum(stu**2))
    f,Y2,tb,m1,m2,mu,mH,mA,cba=G.unpack(x)
    rows.append(dict(chi2=a+float(np.sum(stu**2)),m1=m1,m2=m2,mu=mu,mH=mH,mA=mA,cba=cba,tb=tb,phi=phi,
      tau_mu_g=P.br(2,1,f,Y2,tb,cba,mH,mA,m1,m2,phi),tau_e_g=P.br(2,0,f,Y2,tb,cba,mH,mA,m1,m2,phi),mu_e_g=P.br(1,0,f,Y2,tb,cba,mH,mA,m1,m2,phi),
      h_tau_mu=O.higgs_lfv_br(1,2,Y2,tb,cba),h_tau_e=O.higgs_lfv_br(0,2,Y2,tb,cba),S=None,x=x))
pickle.dump(rows,open('/tmp/gate1_NO_rows.pkl','wb'))
print("NO points chi2<8:",len(rows),flush=True)
