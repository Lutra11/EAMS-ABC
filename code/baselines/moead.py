"""MOEA/D with weighted Tchebycheff decomposition and discrete variation."""
import numpy as np
from genetic import crossover

def solve(ctx):
    pop=ctx.initial(); rng=ctx.rng; n=len(pop)
    w=np.column_stack([np.linspace(0,1,n),np.linspace(1,0,n)]); w=np.maximum(w,.001)
    neighbors=np.argsort(np.sum((w[:,None,:]-w[None,:,:])**2,axis=2),axis=1)[:,:min(10,n)]
    F=np.array([s.f for s in pop]); ideal=F.min(axis=0); scale=np.maximum(F.max(axis=0)-ideal,1.)
    while ctx.count<ctx.cfg.evaluations:
        for i in rng.permutation(n):
            if ctx.count>=ctx.cfg.evaluations: break
            pool=neighbors[i] if rng.random()<.9 else np.arange(n)
            a,b=rng.choice(pool,size=2,replace=False)
            s,_=ctx.evaluate(*crossover(ctx.p,pop[a],pop[b],rng)); ideal=np.minimum(ideal,s.f)
            replaced=0
            for j in rng.permutation(pool):
                old=np.max(w[j]*np.abs(pop[j].f-ideal)/scale); new=np.max(w[j]*np.abs(s.f-ideal)/scale)
                if new<=old: pop[j]=s; replaced+=1
                if replaced==2: break
    return dict(neighborhood=min(10,n),replacement_limit=2,normalization='fixed initial range')
