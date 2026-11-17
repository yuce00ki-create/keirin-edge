"""KEIRIN EDGE v18 24-ticket coverage inference utilities.
Keep 2-winner, free-24-ticket strategy; do not force strict 2x3x4.
The training and experiment bundle is in the attached ZIP.
"""
import itertools
import numpy as np
from scipy.special import softmax

def blend(joint, winner_ranker_logits, alpha=1.0):
    p=np.asarray(joint,dtype=np.float64)
    if p.ndim!=3 or len(set(p.shape))!=1 or np.any(~np.isfinite(p)) or np.any(p<0) or not np.isclose(p.sum(),1):
        raise ValueError("Invalid normalized joint probability tensor")
    if not 0<=alpha<=1:raise ValueError("Bad alpha")
    winner=p.sum(axis=(1,2))
    if np.any(winner<=0):raise ValueError("Invalid winner marginal")
    rank=softmax(np.asarray(winner_ranker_logits,dtype=float))
    q=p*(((1-alpha)*winner+alpha*rank)/winner)[:,None,None]
    return q/q.sum()

def pick_two_first_24(joint,cars=None,k=24):
    p=np.asarray(joint,dtype=float);n=len(p)
    if p.shape!=(n,n,n) or not np.isclose(p.sum(),1) or np.any(p<0):
        raise ValueError("Invalid trifecta tensor")
    if cars is None: cars=list(range(1,n+1))
    if len(cars)!=n or len(set(cars))!=n:raise ValueError("Invalid car numbers")
    best=(-1,None,None)
    for a,b in itertools.combinations(range(n),2):
        candidates=sorted(((i,j,m,float(p[i,j,m])) for i in (a,b) for j in range(n)
            for m in range(n) if len({i,j,m})==3), key=lambda x:-x[3])[:k]
        mass=sum(row[3] for row in candidates)
        if mass>best[0]:best=(mass,(a,b),candidates)
    mass,A,tickets=best
    tickets=[(cars[i],cars[j],cars[m],s) for i,j,m,s in tickets]
    return {"first":[cars[x] for x in A],"tickets":tickets,"ticket_count":len(tickets),
            "predicted_coverage":mass}
