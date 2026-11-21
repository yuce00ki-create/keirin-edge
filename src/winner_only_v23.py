"""KEIRIN EDGE v23: first-place-only model.

LightGBM winner ranking head trained with one winner per race. Outputs only
1st-place probabilities; no trifecta/second/third predictions.
"""
import numpy as np
from scipy.special import softmax
from scipy.optimize import minimize_scalar

def fit_winner_ranker(training_races, n_estimators=80):
    import lightgbm as lgb
    x, y, groups = [], [], []
    for _, features, ranks, _ in training_races:
        x.append(np.asarray(features, float))
        y.extend((np.asarray(ranks)==1).astype(int).tolist())
        groups.append(len(ranks))
    model=lgb.LGBMRanker(objective="lambdarank",label_gain=[0,1],
        n_estimators=n_estimators,learning_rate=.05,max_depth=5,
        num_leaves=15,min_child_samples=50,reg_lambda=6.,
        verbosity=-1,n_jobs=4,random_state=810)
    model.fit(np.vstack(x),np.asarray(y),group=groups,eval_at=(1,2,3))
    return model

def fit_temperature(model,validation_races):
    samples=[]
    for _, features, ranks, _ in validation_races:
        scores=np.asarray(model.predict(features,raw_score=True),float)
        actual=int(np.flatnonzero(np.asarray(ranks)==1)[0])
        samples.append((scores,actual))
    def nll(log_t):
        t=np.exp(log_t)
        return -sum(np.log(max(softmax(scores/t)[a],1e-15))
                    for scores,a in samples)/len(samples)
    result=minimize_scalar(nll,bounds=(np.log(.2),np.log(10)),method="bounded")
    return float(np.exp(result.x))

def predict_winner(model,features,cars,temperature=1.,top_k=3):
    cars=list(cars)
    if not 1<=top_k<=len(cars) or len(set(cars))!=len(cars):
        raise ValueError("Invalid cars or top_k")
    scores=np.asarray(model.predict(features,raw_score=True),float)
    if scores.shape!=(len(cars),) or not np.isfinite(scores).all():
        raise ValueError("Invalid winner scores")
    if not np.isfinite(temperature) or temperature<=0:
        raise ValueError("Invalid temperature")
    probs=softmax(scores/temperature)
    order=np.argsort(-probs,kind="stable")
    return {"winner_pick":cars[int(order[0])],
        "top_k":[cars[int(i)] for i in order[:top_k]],
        "top_k_probability":float(probs[order[:top_k]].sum()),
        "first_place_probabilities":[
            {"car":cars[int(i)],"probability":float(probs[int(i)])}
            for i in order]}

def evaluate(model,races,temp):
    counts={1:0,2:0,3:0};loss=0.;brier=0.
    for _,x,ranks,cars in races:
        result=predict_winner(model,x,cars,temp,top_k=min(3,len(cars)))
        actual=cars[int(np.flatnonzero(np.asarray(ranks)==1)[0])]
        order=[item["car"] for item in result["first_place_probabilities"]]
        for k in counts:counts[k]+=int(actual in order[:k])
        probabilities=np.array([item["probability"] for item in result["first_place_probabilities"]])
        idx=order.index(actual);loss-=np.log(max(probabilities[idx],1e-15))
        target=np.zeros(len(probabilities));target[idx]=1
        brier+=np.sum((probabilities-target)**2)
    n=len(races)
    return {"n":n,"accuracy":{f"top_{k}":counts[k]/n for k in counts},
        "hits":{f"top_{k}":counts[k] for k in counts},
        "winner_nll":loss/n,"winner_brier":brier/n}
