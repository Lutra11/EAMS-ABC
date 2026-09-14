"""Two-objective minimization indicators and corrected paired inference."""
import numpy as np
from scipy.stats import wilcoxon,rankdata,friedmanchisquare

def nondominated(F):
    F=np.unique(np.asarray(F,float),axis=0)
    F=F[np.lexsort((F[:,1],F[:,0]))]; out=[]; best=float('inf')
    for a,b in F:
        if b<best-1e-10: out.append([a,b]); best=b
    return np.array(out)

def hv(F,reference=(1.1,1.1)):
    F=np.asarray(F); F=F[np.all(F<reference,axis=1)]
    if len(F)==0:return 0.
    F=nondominated(F); value=0.; y=reference[1]
    for a,b in F:
        value+=(reference[0]-a)*(y-b); y=b
    return float(value)

def igd_plus(F,R):
    distances=np.sqrt(np.sum(np.maximum(0.,F[None,:,:]-R[:,None,:])**2,axis=2))
    return float(np.mean(np.min(distances,axis=1)))

def holm(pvalues):
    order=np.argsort(pvalues); adjusted=np.empty(len(order)); running=0
    for k,i in enumerate(order):
        running=max(running,(len(order)-k)*pvalues[i]); adjusted[i]=min(1.,running)
    return adjusted

def paired(a,b):
    d=np.array(a)-np.array(b)
    if np.allclose(d,0): return 1.,0.
    p=float(wilcoxon(d,zero_method='wilcox',alternative='two-sided',method='auto').pvalue)
    nz=d[np.abs(d)>1e-12]; r=rankdata(np.abs(nz)); effect=float(np.sum(r*np.sign(nz))/r.sum())
    return p,effect
