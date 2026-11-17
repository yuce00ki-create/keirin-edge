"""KEIRIN EDGE v16: optional, auditable pre-race lineup interaction correction.
The default weights are ZERO until genuine historical data can train them.
Never use actual race results to tune a past-race prediction.
"""
from dataclasses import dataclass
from itertools import permutations
import numpy as np

FEATURE_NAMES=("same_line_1_2","same_line_2_3","same_line_1_3",
 "line_order_1_2","line_order_2_3","line_order_1_3",
 "consecutive_line_2_3","line_triple_1_2_3","head_first","head_second",
 "chaser_second","chaser_third","solo_first","crossline_first_secondthird")

@dataclass(frozen=True)
class Line:
    group: str
    position: int
    length: int

def validate_lines(lines,cars):
    if set(lines)!=set(cars):raise ValueError("missing riders or line positions")
    groups={}
    for rider,l in lines.items():
        if l.position<1 or l.length<1:raise ValueError("bad line metadata")
        groups.setdefault(l.group,[]).append(l.position)
    for name,pos in groups.items():
        if sorted(pos)!=list(range(1,len(pos)+1)):raise ValueError("invalid line ordering")
        if any(l.length!=len(pos) for l in lines.values() if l.group==name):
            raise ValueError("line length mismatch")

def features(a,b,c,lines):
    A,B,C=[lines[x] for x in (a,b,c)]
    ab=A.group==B.group;bc=B.group==C.group;ac=A.group==C.group
    return np.array([
        ab,bc,ac,ab and A.position<B.position,bc and B.position<C.position,
        ac and A.position<C.position,bc and B.position+1==C.position,
        ab and bc and (A.position,B.position,C.position)==(1,2,3),
        A.position==1 and A.length>1,B.position==1 and B.length>1,
        B.position==2,C.position==2,A.length==1,
        (not ab) and bc and B.position<C.position],dtype=float)

def rerank(joint,cars,lines,weights=None):
    """Rerank a valid n*n*n v15 joint probability tensor with line features."""
    cars=list(cars);validate_lines(lines,cars)
    p=np.asarray(joint,dtype=float);n=len(cars)
    if p.shape!=(n,n,n) or np.any(p<0) or not np.isclose(p.sum(),1):
        raise ValueError("invalid joint trifecta probabilities")
    w=np.zeros(len(FEATURE_NAMES)) if weights is None else np.asarray(weights,float)
    if w.shape!=(len(FEATURE_NAMES),) or not np.all(np.isfinite(w)):
        raise ValueError("bad line coefficients")
    out=np.zeros_like(p)
    tuples=list(permutations(range(n),3))
    X=np.stack([features(cars[a],cars[b],cars[c],lines) for a,b,c in tuples])
    z=X@w;z-=z.max()
    for (a,b,c),val in zip(tuples,z):out[a,b,c]=p[a,b,c]*np.exp(val)
    total=out.sum()
    if total<=0:raise ValueError("zero probability")
    return out/total

if __name__=="__main__":
    L={1:Line("A",1,2),7:Line("A",2,2),2:Line("B",1,2),
       5:Line("B",2,2),3:Line("C",1,2),6:Line("C",2,2),
       4:Line("D",1,1)}
    cars=list(range(1,8));p=np.zeros((7,7,7))
    for t in permutations(range(7),3):p[t]=1/210
    assert np.allclose(rerank(p,cars,L),p)
    assert features(1,2,5,L)[13]==1
    print("Line feature checks passed")
