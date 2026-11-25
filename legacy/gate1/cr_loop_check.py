"""Numerical check of the photon-penguin loop function behind the mu-e conversion monopole term.
Feynman-parameter form factor of the q^2 gamma^mu coefficient (neutral scalar M, charged fermion m, massless externals):
2*Int dx dy [xy/D + r xy/(2 D^2)], D=(x+y) r+(1-x-y), r=m^2/M^2, compared with -(4+3 ln r)/9. Ratio -> 1 as r -> 0."""
import numpy as np
from scipy import integrate
for r in [1e-2, 1e-3, 1e-4, (1.777/2000)**2]:
    f = lambda y, x: x*y/((x+y)*r+1-x-y) + r*x*y/(2*((x+y)*r+1-x-y)**2)
    v, _ = integrate.dblquad(f, 0, 1, lambda x: 0, lambda x: 1-x, epsabs=1e-12, epsrel=1e-10)
    print(r, 2*v/(-(4+3*np.log(r))/9))
