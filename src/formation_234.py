"""Exact probability-based 2x3x4 trifecta-formation selector for KEIRIN EDGE v15.
No odds/outcomes are read at selection time. Returns 2 first-place candidates,
3 second-place candidates, 4 third-place candidates, and all legal trifectas.
"""
from itertools import combinations
from functools import lru_cache
import numpy as np

@lru_cache(None)
def subsets(n,k): return list(combinations(range(n),k))

def select_234(joint,cars=None,maximize_ticket_count=False):
    p=np.asarray(joint,dtype=float)
    if p.ndim!=3 or len(set(p.shape))!=1:raise ValueError("Expected n x n x n tensor")
    n=len(p)
    if not 4<=n<=9:raise ValueError("4 to 9 riders required")
    if np.any(~np.isfinite(p)) or np.any(p<0) or not np.isclose(p.sum(),1,atol=1e-6):
        raise ValueError("Invalid/unnormalized probability tensor")
    for i in range(n):
        if p[i,i,:].sum()>1e-8 or p[i,:,i].sum()>1e-8 or p[:,i,i].sum()>1e-8:
            raise ValueError("Repeated car has nonzero probability")
    if cars is None:cars=list(range(1,n+1))
    if len(cars)!=n or len(set(cars))!=n:raise ValueError("Bad car mapping")
    max_tickets={4:10,5:15,6:18,7:20,8:22,9:24}[n]
    best=((-np.inf,-1),None)
    Csets=subsets(n,4)
    for A in subsets(n,2):
        for B in subsets(n,3):
            v=p[np.ix_(A,B,range(n))].sum(axis=(0,1))
            if not maximize_ticket_count:
                C=tuple(sorted(np.argsort(-v,kind="stable")[:4].tolist()))
                cnt=sum(len({a,b,c})==3 for a in A for b in B for c in C)
                candidate=(float(v[list(C)].sum()),cnt)
                if candidate>best[0]:best=(candidate,(A,B,C))
            else:
                subset_indices=np.asarray(Csets,dtype=int)
                ccounts=np.asarray([sum(len({a,b,c})==3 for a in A for b in B) for c in range(n)])
                eligibility=np.flatnonzero(ccounts[subset_indices].sum(axis=1)==max_tickets)
                if len(eligibility)==0:continue
                values=v[subset_indices[eligibility]].sum(axis=1)
                k=int(eligibility[int(np.argmax(values))]);C=Csets[k]
                candidate=(float(values.max()),max_tickets)
                if candidate>best[0]:best=(candidate,(A,B,C))
    A,B,C=best[1]
    tickets=sorted(((cars[a],cars[b],cars[c],float(p[a,b,c])) for a in A for b in B for c in C
        if len({a,b,c})==3),key=lambda row:(-row[3],row[:3]))
    return {"first":[cars[a] for a in A],"second":[cars[b] for b in B],
       "third":[cars[c] for c in C],"tickets":tickets,
       "ticket_count":len(tickets),"max_legal_tickets":max_tickets,
       "predicted_coverage":sum(t[3] for t in tickets),
       "objective":"maximize_coverage_with_maximum_ticket_count" if maximize_ticket_count else "maximize_coverage"}
