"""KEIRIN EDGE v15 coverage-oriented 24-ticket ensemble.

Blend the winner-marginal distributions of two separately trained models:
(1) v12 conditional trifecta LightGBM, (2) winner top-two LambdaRank.
Keep v12 2nd/3rd conditional probabilities; select the best winner pair
and top-24 valid triples by joint probability. The trained ranker, 2025
alpha tuning and reproducible experiments are in the conversation ZIP.
"""
import itertools
import numpy as np
from scipy.special import softmax

def mix_winner_head(v12_joint, ranker_winner_scores, alpha=0.75):
    p=np.asarray(v12_joint,dtype=float)
    if p.ndim!=3 or len(set(p.shape))!=1 or not np.isclose(p.sum(),1):
        raise ValueError("Expected normalized n x n x n joint trifecta probabilities")
    if not 0<=alpha<=1:raise ValueError("alpha must be between 0 and 1")
    base=p.sum(axis=(1,2))
    if np.any(base<=0):raise ValueError("Every runner must have positive winner probability")
    extra=softmax(np.asarray(ranker_winner_scores,dtype=float))
    out=p*((1-alpha)*base+alpha*extra)[:,None,None]/base[:,None,None]
    return out/out.sum()

def optimize_two_winners_top24(joint,k=24):
    p=np.asarray(joint,dtype=float);n=len(p)
    if k<1 or k>n*(n-1)*(n-2):raise ValueError("Invalid ticket count")
    best=(-1.,None,None)
    for first_pair in itertools.combinations(range(n),2):
        tickets=sorted(((a,b,c,float(p[a,b,c])) for a in first_pair
            for b in range(n) for c in range(n) if len({a,b,c})==3),
            key=lambda row:-row[3])[:k]
        mass=sum(t[3] for t in tickets)
        if mass>best[0]:best=(mass,first_pair,tickets)
    return {"coverage_probability":best[0],"first_pair":best[1],
            "tickets":best[2],"tickets_n":len(best[2])}

def rerank(v12_joint, ranker_winner_scores, alpha=0.75, k=24):
    return optimize_two_winners_top24(mix_winner_head(v12_joint,ranker_winner_scores,alpha),k)
