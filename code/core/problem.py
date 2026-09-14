from dataclasses import dataclass
from pathlib import Path
import numpy as np

@dataclass
class Problem:
    name: str
    n: int
    m: int
    counts: np.ndarray
    offsets: np.ndarray
    jobs: np.ndarray
    times: np.ndarray
    eligible: list
    proc: np.ndarray
    idle: np.ndarray
    @property
    def size(self): return len(self.jobs)

def read_problem(path, scenario='independent', idle_ratio=0.15):
    lines=Path(path).read_text().strip().splitlines()
    header=lines[0].split(); n,m=map(int,header[:2])
    counts=[]; rows=[]; jobs=[]
    tokens=list(map(int,' '.join(lines[1:]).split())); pos=0
    for j in range(n):
        count=tokens[pos]; pos+=1; counts.append(count)
        for _ in range(count):
            k=tokens[pos]; pos+=1; row=np.zeros(m,dtype=np.float64)
            for _ in range(k):
                machine,p=tokens[pos:pos+2]; pos+=2
                if machine<0 or machine>=m or p<=0 or row[machine]!=0: raise ValueError('Invalid instance')
                row[machine]=p
            rows.append(row); jobs.append(j)
    if pos!=len(tokens): raise ValueError('Unconsumed instance tokens')
    times=np.array(rows); eligible=[np.flatnonzero(r).astype(np.int64) for r in times]
    # Deterministic dimensionless power; scenario has no stochastic dependence on runs.
    if scenario=='independent': proc=2.0+((37*np.arange(1,m+1)+11)%17)/4.0
    elif scenario=='speed':
        means=np.array([np.mean(times[times[:,k]>0,k]) if np.any(times[:,k]>0) else 1 for k in range(m)])
        speed=1/means; spread=np.ptp(speed)
        proc=2.0+4.0*(speed-speed.min())/(spread if spread else 1)
    else: raise ValueError(scenario)
    return Problem(Path(path).stem,n,m,np.array(counts,dtype=np.int64),np.r_[0,np.cumsum(counts)[:-1]].astype(np.int64),np.array(jobs,dtype=np.int64),times,eligible,proc,idle_ratio*proc)
