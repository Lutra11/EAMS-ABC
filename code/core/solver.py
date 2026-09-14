from dataclasses import dataclass,asdict
import time
import numpy as np
from .decoder import decode,validate
from .initialization import initialize
from .operators import move,reconstruct
from .pareto import Archive,dominates,ranks,environmental

@dataclass
class Config:
    evaluations:int=6000
    population:int=40
    archive:int=80
    limit:int=30
    alpha:float=.5
    rho:float=.2
    pmin:float=.05
    reconstruction:float=.2
    guided:bool=True
    adaptive:bool=True
    rebuild:bool=True
    gap:bool=True
    uniform_selection:bool=False
    elite_exchange:bool=True

class RunContext:
    def __init__(self,p,seed,cfg):
        self.p=p; self.rng=np.random.default_rng(seed); self.cfg=cfg; self.archive=Archive(cfg.archive); self.count=0; self.history=[]
        self.checkpoints=set(np.linspace(cfg.population,cfg.evaluations,21,dtype=int).tolist())
    def evaluate(self,os,ms):
        if self.count>=self.cfg.evaluations: return None,False
        s=decode(self.p,os,ms); self.count+=1; entered=self.archive.add(s)
        if self.count in self.checkpoints: self.history.append(dict(evaluations=self.count,front=self.archive.F.tolist()))
        return s,entered
    def initial(self): return [self.evaluate(*initialize(self.p,self.rng,i))[0] for i in range(self.cfg.population)]

def abc(ctx):
    p=ctx.p; rng=ctx.rng; cfg=ctx.cfg; pop=ctx.initial(); trials=np.zeros(len(pop),int)
    Q=np.ones(4)*.1; uses=np.zeros(4,int); gains=np.zeros(4); scout=0; generation=0; op_history=[]; exchange_calls=0
    while ctx.count<cfg.evaluations:
        r=ranks(np.array([s.f for s in pop])); weights=1/(1+r); weights=weights/weights.sum()
        indices=list(range(len(pop)))+rng.choice(len(pop),len(pop),p=weights).tolist()
        candidates=[]
        for position,i in enumerate(indices):
            if ctx.count>=cfg.evaluations: break
            old=pop[i]
            if trials[i]>=cfg.limit:
                genes=reconstruct(p,old,rng,cfg.reconstruction,cfg.alpha) if cfg.rebuild else initialize(p,rng,0)
                s,_=ctx.evaluate(*genes); pop[i]=s; candidates.append(s); trials[i]=0; scout+=1; continue
            if cfg.elite_exchange and position>=len(pop) and rng.random()<.75:
                from .variation import crossover
                peer=ctx.archive.items[int(rng.integers(len(ctx.archive.items)))]
                s,_=ctx.evaluate(*crossover(p,old,peer,rng)); candidates.append(s); exchange_calls+=1
                continue
            probs=cfg.pmin+(1-4*cfg.pmin)*(Q+1e-9)/(Q.sum()+4e-9) if cfg.adaptive else np.ones(4)/4
            op=int(rng.choice(4,p=probs)); uses[op]+=1
            s,entered=ctx.evaluate(*move(p,old,rng,op,cfg.guided,cfg.alpha,cfg.gap,cfg.uniform_selection))
            candidates.append(s)
            dom=dominates(s.f,old.f)
            # No reward for a dominated move that improves only one objective.
            improvement=np.maximum(0.,(old.f-s.f)/(old.f+1e-12))
            reward=float(.25*improvement.sum()+.5*(1 if dom else .5)) if (dom or entered) else 0.
            Q[op]=(1-cfg.rho)*Q[op]+cfg.rho*reward; gains[op]+=reward
            accept=dom
            if not dom and not dominates(old.f,s.f):
                # Random scalar preference only resolves incomparable incumbent moves.
                w=rng.random(); accept=w*s.f[0]/old.f[0]+(1-w)*s.f[1]/old.f[1]<1
            if accept: pop[i]=s; trials[i]=0
            else: trials[i]+=1
        generation+=1
        if cfg.elite_exchange:
            # Elitist population feedback closes the loop from archive to bee search.
            previous={id(s):int(t) for s,t in zip(pop,trials)}
            unique={}
            for s in pop+candidates: unique[(s.os.tobytes(),s.ms.tobytes())]=s
            pool=list(unique.values())
            pop=environmental(pool,len(pop)) if len(pool)>=len(pop) else environmental(pop+candidates,len(pop))
            trials=np.array([previous.get(id(s),0) for s in pop],int)
        if generation%10==0: op_history.append([ctx.count,*probs.tolist()])
    return dict(operator_uses=uses.tolist(),operator_reward=gains.tolist(),scouts=scout,operator_history=op_history,exchange_calls=exchange_calls)

def run(p,method,seed,cfg=None):
    cfg=cfg or Config(); ctx=RunContext(p,seed,cfg)
    # Warm-up is excluded from wall time, identically for every method.
    warm=decode(p,*initialize(p,np.random.default_rng(0))); ranks(np.array([warm.f,warm.f]))
    from .criticality import criticality
    criticality(p,warm)
    start=time.perf_counter()
    if method in ['NSGA-II','MA-NSGA-II']:
        from nsga2 import solve
        info=solve(ctx,memetic=method=='MA-NSGA-II')
    elif method=='MOEA-D':
        from moead import solve
        info=solve(ctx)
    else: info=abc(ctx)
    elapsed=time.perf_counter()-start
    for s in ctx.archive.items: validate(p,s)
    return dict(method=method,seed=seed,instance=p.name,config=asdict(cfg),evaluations=ctx.count,seconds=elapsed,front=ctx.archive.F.tolist(),history=ctx.history,diagnostics=info,solutions=[dict(os=s.os.tolist(),ms=s.ms.tolist(),f=s.f.tolist()) for s in ctx.archive.items])

