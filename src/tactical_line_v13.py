"""KEIRIN EDGE v13: verified lineup, triple interactions, calibrated correction.
Full runnable source and tests: keirin_edge_v13_lineaware_module.zip in ChatGPT artifact.
This module makes no claim of training without genuine pre-start line snapshots.
"""
from datetime import datetime
import csv, itertools
import numpy as np
from scipy.special import logsumexp
from scipy.optimize import minimize

def load_lines(path):
    required={"race_id","banum","line_id","line_pos","source_url","captured_at","start_at"}
    by_race={}
    with open(path,encoding="utf-8-sig",newline="") as f:
        reader=csv.DictReader(f)
        if required-set(reader.fieldnames or []):raise ValueError("Missing official-line columns")
        for row in reader:
            t=datetime.fromisoformat(row["captured_at"].replace("Z","+00:00"))
            start=datetime.fromisoformat(row["start_at"].replace("Z","+00:00"))
            if not t.tzinfo or not start.tzinfo or t>=start:raise ValueError("Post-start or unknown timestamp")
            if not row["source_url"].startswith(("https://","http://")):raise ValueError("Missing source")
            key=row["race_id"];car=int(row["banum"]);lid=row["line_id"];pos=int(row["line_pos"])
            race=by_race.setdefault(key,{})
            if car in race:raise ValueError("Duplicate car")
            race[car]=(lid,pos)
    for rid,race in by_race.items():
        if len(race)<5 or len(race)>9:raise ValueError("Incomplete lineup")
        lines={}
        for car,(lid,pos) in race.items():lines.setdefault(lid,[]).append(pos)
        if any(sorted(pos)!=list(range(1,len(pos)+1)) for pos in lines.values()):
            raise ValueError("Inconsistent line ordering")
        by_race[rid]={car:(lid,pos,len(lines[lid])) for car,(lid,pos) in race.items()}
    return by_race

def features(a,b,c,lines,scores):
    la,pa,na=lines[a];lb,pb,nb=lines[b];lc,pc,nc=lines[c]
    ab=la==lb;ac=la==lc;bc=lb==lc
    return np.array([ab,ac,bc,ab and pa<pb,ab and pb<pa,bc and pb<pc,
        ac and pa<pc,ab and bc,ab and bc and (pa,pb,pc)==(1,2,3),
        pa==1,pb==2,pc==3,na,nb,nc,scores[a]-scores[b],
        scores[b]-scores[c]],dtype=np.float64)

def triples(cars,lines,scores):
    if set(cars)!=set(lines):raise ValueError("Missing line coverage")
    tickets=list(itertools.permutations(cars,3))
    return tickets,np.stack([features(*t,lines,scores) for t in tickets])

def probabilities(cars,lines,scores,baseline_joint,weights):
    tickets,X=triples(cars,lines,scores)
    idx={car:i for i,car in enumerate(cars)}
    prior=np.array([baseline_joint[idx[a],idx[b],idx[c]] for a,b,c in tickets])
    if np.any(prior<=0):raise ValueError("Non-positive baseline probabilities")
    z=np.log(prior)+X@weights
    return list(zip(tickets,np.exp(z-logsumexp(z)).tolist()))

def train(residual_races,penalty=20):
    """Requires >500 real line-race bundles and truly out-of-fold baseline probs.
    Inputs: cars,lines,scores,baseline_joint,actual (3 car numbers).
    Caller must enforce chronological splits and verified outcome provenance.
    """
    if len(residual_races)<500:raise ValueError("Insufficient genuine historical line data")
    prepared=[]
    for r in residual_races:
        cars=r["cars"];tickets,X=triples(cars,r["lines"],r["scores"])
        answer=tickets.index(tuple(r["actual"]))
        idx={car:i for i,car in enumerate(cars)}
        p=np.array([r["baseline_joint"][idx[a],idx[b],idx[c]] for a,b,c in tickets])
        if np.any(p<=0) or not np.isclose(p.sum(),1):raise ValueError("Bad OOF probabilities")
        prepared.append((X,np.log(p),answer))
    def objective(w):
        loss=.5*penalty*np.dot(w,w)
        grad=penalty*w.copy()
        for X,base,y in prepared:
            z=base+X@w
            prob=np.exp(z-logsumexp(z))
            loss+=logsumexp(z)-z[y]
            grad+=prob@X-X[y]
        return loss,grad
    fit=minimize(objective,np.zeros(17),jac=True,method="L-BFGS-B")
    if not fit.success:raise RuntimeError(fit.message)
    return fit.x
