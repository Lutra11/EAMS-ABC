import bootstrap
from pathlib import Path
import json,itertools,time
import numpy as np
from eams_abc.problem import read_problem
from eams_abc.decoder import decode,validate
from eams_abc.initialization import initialize
from eams_abc.operators import move,reconstruct
from eams_abc.pareto import Archive,ranks
from eams_abc.criticality import criticality

ROOT=Path(__file__).resolve().parents[2]
def main():
    records=[]; rng=np.random.default_rng(9861)
    for path in sorted((ROOT/'dataset'/'raw').rglob('*.txt')):
        p=read_problem(path); checks=0
        for i in range(20):
            s=decode(p,*initialize(p,rng,i)); validate(p,s); checks+=1
            probs=criticality(p,s); assert np.isclose(probs.sum(),1)
            for op in range(4):
                t=decode(p,*move(p,s,rng,op)); validate(p,t); checks+=1
            t=decode(p,*reconstruct(p,s,rng)); validate(p,t); checks+=1
        records.append(dict(instance=p.name,operations=p.size,jobs=p.n,machines=p.m,checks=checks))
    # Independent small exhaustive schedule enumeration, not the shared decoder.
    tiny=ROOT/'dataset'/'tiny_validation.txt'
    tiny.write_text('2 2\n2 2 0 2 1 3 1 1 2\n2 1 1 1 2 0 2 1 3\n')
    p=read_problem(tiny); front=[]; feasible=0
    for ms in itertools.product(*p.eligible):
        durations=p.times[np.arange(4),ms]
        for starts in itertools.product(range(10),repeat=4):
            end=np.array(starts)+durations
            if starts[1]<end[0] or starts[3]<end[2] or max(end)>9: continue
            if any(ms[a]==ms[b] and starts[a]<end[b] and starts[b]<end[a] for a in range(4) for b in range(a+1,4)): continue
            feasible+=1; energy=0.
            for k in range(2):
                ids=[o for o in range(4) if ms[o]==k]
                if ids:
                    work=sum(durations[o] for o in ids)
                    energy+=p.proc[k]*work+p.idle[k]*(max(end[o] for o in ids)-min(starts[o] for o in ids)-work)
            front.append((max(end),energy))
    F=np.unique(front,axis=0); exact=F[ranks(F)==0]
    encoded=[]
    for os in set(itertools.permutations([0,0,1,1])):
        for ms in itertools.product(*p.eligible):
            s=decode(p,np.array(os),np.array(ms));validate(p,s);encoded.append(s.f)
    G=np.unique(encoded,axis=0); decoded_front=G[ranks(G)==0]
    assert np.allclose(exact,decoded_front),(exact,decoded_front)
    # Hand-calculated energy: J1 on M1 then M2; J2 starts on M2 then M1.
    hand=decode(p,np.array([0,1,0,1]),np.array([0,1,1,0]))
    validate(p,hand)
    assert hand.f[0]==4 and np.isclose(hand.f[1],4*p.proc[0]+3*p.proc[1])
    report=dict(status='passed',benchmark_checks=records,total_checks=sum(x['checks'] for x in records),tiny_feasible_schedules=feasible,exact_front=exact.tolist(),encoded_front=decoded_front.tolist(),hand_f=hand.f.tolist())
    (ROOT/'code'/'output'/'correctness.json').write_text(json.dumps(report,indent=2),encoding='utf-8'); print(json.dumps(report,indent=2))
if __name__=='__main__': main()
