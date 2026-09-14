"""NSGA-II with scheduling-specific variation; MA variant is our transparent control.

MA-NSGA-II is NOT attributed to any published recent algorithm.
"""
import numpy as np
from eams_abc.pareto import ranks,crowding,environmental,dominates
from eams_abc.operators import move
from genetic import crossover

def solve(ctx,memetic=False):
    pop=ctx.initial(); rng=ctx.rng; local_calls=0
    while ctx.count<ctx.cfg.evaluations:
        F=np.array([s.f for s in pop]); rank=ranks(F); cd=np.zeros(len(pop))
        for level in np.unique(rank):
            ids=np.flatnonzero(rank==level); cd[ids]=crowding(F[ids])
        def tournament():
            a,b=rng.integers(len(pop),size=2)
            return pop[a if (rank[a],-cd[a])<(rank[b],-cd[b]) else b]
        children=[]
        while len(children)<len(pop) and ctx.count<ctx.cfg.evaluations:
            s,_=ctx.evaluate(*crossover(ctx.p,tournament(),tournament(),rng))
            if memetic and ctx.count<ctx.cfg.evaluations:
                # One additional move for every child, fully counted in the common budget.
                t,_=ctx.evaluate(*move(ctx.p,s,rng,int(rng.integers(4)),True,ctx.cfg.alpha,True)); local_calls+=1
                if dominates(t.f,s.f) or (not dominates(s.f,t.f) and rng.random()<.5): s=t
            children.append(s)
        pop=environmental(pop+children,ctx.cfg.population)
    return dict(local_calls=local_calls)
