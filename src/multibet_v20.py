"""Multi-bet probabilities derived from v19 trifecta joint probability tensor.
Keep odds outside the predictor. Detailed executable tests are in chat ZIP.
"""
from itertools import permutations,combinations
from collections import defaultdict
import numpy as np

KINDS={"exacta":"2車単","quinella":"2車複","wide":"ワイド","trio":"3連複","trifecta":"3連単"}

def marginals(joint,cars=None):
    p=np.asarray(joint,float)
    if p.ndim!=3 or len(set(p.shape))!=1 or np.any(~np.isfinite(p)) or np.any(p<0) or not np.isclose(p.sum(),1):
        raise ValueError("Invalid joint probabilities")
    n=len(p);cars=list(range(1,n+1)) if cars is None else list(cars)
    if len(cars)!=n or len(set(cars))!=n:raise ValueError("Invalid cars")
    d={x:defaultdict(float) for x in KINDS}
    for a,b,c in permutations(range(n),3):
        v=float(p[a,b,c])
        if not v:continue
        x,y,z=cars[a],cars[b],cars[c]
        d["exacta"][(x,y)]+=v
        d["quinella"][tuple(sorted((x,y)))]+=v
        d["trio"][tuple(sorted((x,y,z)))]+=v
        d["trifecta"][(x,y,z)]+=v
        for pair in combinations(sorted((x,y,z)),2):d["wide"][pair]+=v
    return {kind:dict(vals) for kind,vals in d.items()}

def expected_value_candidates(joint,odds, min_ev=1.15, cars=None):
    """odds: dict keyed by (english_kind,tuple_cars), values payout multipliers.
    EV screen is NOT a proof of profitability; calibrate from untouched future races.
    """
    d=marginals(joint,cars)
    results=[]
    for (kind,key),o in odds.items():
        key=tuple(key)
        if kind not in d or key not in d[kind]:continue
        if not np.isfinite(o) or o<=1:continue
        prob=d[kind][key]; ev=prob*o
        if ev>=min_ev:
            results.append({"kind":KINDS[kind],"ticket":key,"prob":prob,
                            "odds":o,"estimated_return_multiple":ev})
    return sorted(results,key=lambda row:-row["estimated_return_multiple"])
