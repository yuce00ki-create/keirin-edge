"""v21: confidence-based screening for 3連複 top-k prediction.

This is research-only screening, not a profit/ROI promise. Use a threshold
fitted on PREVIOUS races only. The full joint q must be a pre-race forecast.
"""
from itertools import combinations, permutations
import numpy as np

def trio_topk(q,cars=None,k=4):
    q=np.asarray(q,float)
    if q.ndim!=3 or len(set(q.shape))!=1 or not (5<=len(q)<=9):
        raise ValueError("Need 5-9 participant trifecta probability tensor")
    if np.any(q<0) or not np.all(np.isfinite(q)) or not np.isclose(q.sum(),1,atol=1e-5):
        raise ValueError("q must be normalized and nonnegative")
    n=len(q)
    cars=list(range(1,n+1)) if cars is None else list(cars)
    if len(cars)!=n or len(set(cars))!=n:raise ValueError("invalid cars")
    trios=[]
    for trio in combinations(range(n),3):
        prob=sum(float(q[t]) for t in permutations(trio))
        trios.append((prob,tuple(sorted(cars[t] for t in trio))))
    trios.sort(key=lambda v:(-v[0],v[1]))
    selected=trios[:k]
    return {"k":k,"tickets":[{"cars":list(t),"prob":p} for p,t in selected],
            "estimated_coverage":sum(p for p,_ in selected)}

def fit_threshold(historical_coverages,quantile=.8):
    """Only pass data from an earlier calibration period, not the evaluation races."""
    scores=np.asarray(historical_coverages,float)
    if scores.ndim!=1 or len(scores)<50 or np.any(~np.isfinite(scores)):
        raise ValueError("insufficient or invalid earlier confidence scores")
    if not 0<quantile<1:raise ValueError("quantile must be between 0 and 1")
    return float(np.quantile(scores,quantile))

def screen(q,threshold,cars=None,k=4):
    result=trio_topk(q,cars,k)
    result["threshold"]=float(threshold)
    result["high_confidence"]=bool(result["estimated_coverage"]>=threshold)
    result["warning"]="Selective hit rate is not positive expected return."
    return result
