"""Exact trifecta probability optimization for 2x3x4 position boxes.
Input is v12/v11 3D joint probability tensor. No actual results or odds are used.
"""
import itertools
import numpy as np

def optimize_box(joint, counts=(2,3,4)):
    p=np.asarray(joint,dtype=float)
    if p.ndim!=3 or len(set(p.shape))!=1:
        raise ValueError("Input must be square 3D joint probability tensor")
    n=len(p)
    if n<max(counts) or n>9 or np.any(~np.isfinite(p)) or np.any(p<0):
        raise ValueError("Invalid runners or probabilities")
    if not np.isclose(p.sum(),1,atol=1e-6):
        raise ValueError("Probabilities must sum to one")
    for i in range(n):
        if p[i,i,:].sum()>1e-8 or p[i,:,i].sum()>1e-8 or p[:,i,i].sum()>1e-8:
            raise ValueError("Duplicate finishing positions must have zero probability")
    best=(-1,None)
    for A in itertools.combinations(range(n),counts[0]):
        for B in itertools.combinations(range(n),counts[1]):
            by_third=p[np.ix_(A,B,range(n))].sum(axis=(0,1))
            C=tuple(sorted(np.argsort(-by_third,kind="stable")[:counts[2]].tolist()))
            mass=float(by_third[list(C)].sum())
            if mass>best[0]+1e-14:best=(mass,(A,B,C))
    A,B,C=best[1]
    bets=sorted([(a,b,c,float(p[a,b,c])) for a in A for b in B for c in C if len({a,b,c})==3],key=lambda t:-t[3])
    return {"first":A,"second":B,"third":C,"tickets":bets,"tickets_n":len(bets),
            "coverage":float(sum(t[3] for t in bets))}

def optimize_first_two_24(joint,k=24):
    p=np.asarray(joint,dtype=float);n=len(p); best=(-1,None,None)
    for A in itertools.combinations(range(n),2):
        tickets=sorted([(a,b,c,float(p[a,b,c])) for a in A for b in range(n)
            for c in range(n) if len({a,b,c})==3],key=lambda x:-x[3])[:k]
        mass=sum(t[3] for t in tickets)
        if mass>best[0]:best=(mass,A,tickets)
    return {"first":best[1],"tickets":best[2],"tickets_n":len(best[2]),"coverage":best[0]}
