"""KEIRIN EDGE v19: k-ticket coverage with first-place candidate constraint.

Optimizes the total calibrated trifecta probability from an existing joint tensor.
Only layout/selection changes. Never leaks actual outcomes or payout odds.
"""
import itertools
import numpy as np

def optimize_k(joint, cars=None, k=15, first_candidates=3):
    p=np.asarray(joint,dtype=float); n=len(p)
    if p.shape!=(n,n,n) or not (5<=n<=9): raise ValueError("Invalid joint shape")
    if np.any(~np.isfinite(p)) or np.any(p<0) or not np.isclose(p.sum(),1):
        raise ValueError("Joint probability must be nonnegative and sum to 1")
    if not (1<=first_candidates<=n) or not (1<=k<=n*(n-1)*(n-2)):
        raise ValueError("Invalid selection size")
    if cars is None: cars=list(range(1,n+1))
    if len(cars)!=n or len(set(cars))!=n: raise ValueError("Bad car numbers")
    best=(-1,None,None)
    for A in itertools.combinations(range(n),first_candidates):
        ts=sorted(((a,b,c,float(p[a,b,c])) for a in A for b in range(n)
                for c in range(n) if len({a,b,c})==3),key=lambda t:-t[3])[:k]
        mass=sum(t[3] for t in ts)
        if mass>best[0]: best=(mass,A,ts)
    prob, first, ts=best
    return dict(first=[cars[a] for a in first],
        tickets=[(cars[a],cars[b],cars[c],v) for a,b,c,v in ts],
        n_tickets=len(ts),estimated_hit_probability=prob)

def compare_15_vs_24(joint,cars=None):
    small=optimize_k(joint,cars,15,3)
    full=optimize_k(joint,cars,24,3)
    return dict(tickets15=small,tickets24=full,
        incremental_probability=full["estimated_hit_probability"]-small["estimated_hit_probability"])