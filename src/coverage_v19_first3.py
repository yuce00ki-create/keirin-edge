"""Coverage-first formation selection for KEIRIN EDGE v18/v19.

Use v18 calibrated joint probabilities; select first-place candidate sets of
three, then the best 24 legal triples without enforcing 3x3x3 (preferred).
A strict 3x3x3 mode is provided for comparison but has lower observed hit rate.
"""
from itertools import combinations
import numpy as np

def first_three_free24(joint,cars=None,k=24):
    p=np.asarray(joint,dtype=float)
    n=len(p)
    if p.shape!=(n,n,n) or not 5<=n<=9 or np.any(~np.isfinite(p)) or np.any(p<0) or not np.isclose(p.sum(),1):
        raise ValueError("Invalid joint tensor")
    if cars is None: cars=list(range(1,n+1))
    if len(cars)!=n or len(set(cars))!=n:raise ValueError("Bad car list")
    best=(-1,None,None)
    for A in combinations(range(n),3):
        cand=sorted(((a,b,c,float(p[a,b,c])) for a in A for b in range(n)
                     for c in range(n) if len({a,b,c})==3),key=lambda x:-x[3])[:k]
        mass=sum(t[3] for t in cand)
        if mass>best[0]:best=(mass,A,cand)
    mass,A,tickets=best
    return {"first":[cars[a] for a in A],
            "tickets":[(cars[a],cars[b],cars[c],pr) for a,b,c,pr in tickets],
            "ticket_count":len(tickets),"predicted_coverage":mass}

def strict_three_three_three(joint,cars=None):
    p=np.asarray(joint,dtype=float);n=len(p)
    if p.shape!=(n,n,n) or not 5<=n<=9 or np.any(~np.isfinite(p)) or np.any(p<0) or not np.isclose(p.sum(),1):
        raise ValueError("Invalid joint tensor")
    if cars is None:cars=list(range(1,n+1))
    sets=np.array(list(combinations(range(n),3)),dtype=int)
    best=(-1,None)
    for A in sets:
        for B in sets:
            third_mass=p[np.ix_(A,B,range(n))].sum(axis=(0,1))
            C=tuple(np.argsort(-third_mass,kind="stable")[:3])
            mass=float(third_mass[list(C)].sum())
            if mass>best[0]:best=(mass,(A,B,C))
    _,(A,B,C)=best
    tickets=sorted(((cars[a],cars[b],cars[c],float(p[a,b,c])) for a in A for b in B
                   for c in C if len({a,b,c})==3),key=lambda x:-x[3])
    return {"first":[cars[a] for a in A],"second":[cars[b] for b in B],
            "third":[cars[c] for c in C],"tickets":tickets,
            "ticket_count":len(tickets),"predicted_coverage":sum(t[3] for t in tickets)}

if __name__=="__main__":
    import itertools
    for n in (5,6,7,8,9):
        p=np.zeros((n,n,n))
        for a,b,c in itertools.permutations(range(n),3):p[a,b,c]=1/(n*(n-1)*(n-2))
        r=strict_three_three_three(p)
        assert r["ticket_count"]=={5:15,6:18,7:21,8:24,9:27}[n]
        assert first_three_free24(p)["ticket_count"]==24
    print("Formation selector smoke tests passed")
