"""KEIRIN EDGE v22: ticket coverage without forced outsiders.

This changes ticket selection, NOT the trained model. Unrestricted mode
maximizes total model probability over exactly k valid trifecta tickets.
"""
from itertools import combinations, permutations
import numpy as np

def pick_tickets(joint, k=20, first_limit=3):
    q=np.asarray(joint,float)
    n=len(q)
    if q.shape!=(n,n,n) or not 5<=n<=9:
        raise ValueError("Expected 5-to-9-car trifecta probability tensor")
    if not np.isfinite(q).all() or (q<0).any() or not np.isclose(q.sum(),1,atol=1e-5):
        raise ValueError("Invalid probability tensor")
    if not 1<=k<=n*(n-1)*(n-2):
        raise ValueError("Invalid ticket count")
    ordered=sorted(((a,b,c,float(q[a,b,c])) for a,b,c in permutations(range(n),3)),key=lambda t:(-t[3],t[:3]))
    if first_limit is None:
        selected=ordered[:k]
    else:
        if not 1<=first_limit<=n: raise ValueError("Invalid first-place count")
        best=(-1,None)
        for first in combinations(range(n),first_limit):
            subset=[t for t in ordered if t[0] in first][:k]
            mass=sum(t[3] for t in subset)
            if mass>best[0]:best=(mass,subset)
        selected=best[1]
    return {"tickets":[tuple(t[:3]) for t in selected],
            "ticket_probabilities":[t[3] for t in selected],
            "estimated_coverage":float(sum(t[3] for t in selected)),
            "first_places":sorted({t[0] for t in selected})}

def compare(joint,k=20):
    return {"v19_first3":pick_tickets(joint,k,3),
            "unrestricted":pick_tickets(joint,k,None)}
