"""Strict top-5 validation and leakage guard (stdlib only)."""
import csv, json, math, re, sys
from collections import Counter
from datetime import datetime
BANNED={'rank','agari','finish_type','margin','win_lose','ni_sha_tan','san_ren_tan','san_ren_fuku','wide','payout','dividend','result','finish_rank'}
def gate_features(names,source_asof,start_at,strict=True):
    bad=sorted(set(names)&BANNED)
    if bad:raise ValueError('post-race/target fields: '+','.join(bad))
    if not source_asof:
        if strict:raise ValueError('missing source_asof; pre-race availability unproven')
        return False
    a=datetime.fromisoformat(source_asof.replace('Z','+00:00'))
    b=datetime.fromisoformat(start_at.replace('Z','+00:00'))
    if a.tzinfo is None or b.tzinfo is None:raise ValueError('timezone required')
    if a>=b:raise ValueError('snapshot at/after start')
    return True
def audit_csv(path):
    rows=list(csv.DictReader(open(path,encoding='utf-8-sig',newline='')))
    if not rows:raise ValueError('empty')
    if len({r['race_id'] for r in rows})!=len(rows):raise ValueError('duplicate races')
    counts=Counter()
    for r in rows:
        a=r['v8_top5'].split(' / ');b=r['v9_top5'].split(' / ')
        if len(a)!=5 or len(b)!=5 or len(set(a))!=5 or len(set(b))!=5:raise ValueError('invalid top5')
        x=int(r['actual'] in a);y=int(r['actual'] in b)
        if (x,y)!=(int(r['v8_hit']),int(r['v9_hit'])):raise ValueError('hit mismatch')
        counts[(x,y)]+=1
    n=len(rows);x=counts[(1,0)];y=counts[(0,1)];m=x+y
    p=min(1.,2*sum(math.comb(m,i) for i in range(min(x,y)+1))/2**m) if m else 1.
    years=Counter(re.search(r'20\d{2}',r['race_id']).group() if re.search(r'20\d{2}',r['race_id']) else 'unknown' for r in rows)
    return dict(n=n,v8_hits=x+counts[(1,1)],v9_hits=y+counts[(1,1)],
                delta_pp=100*(y-x)/n,v8_only=x,v9_only=y,both_hit=counts[(1,1)],
                neither=counts[(0,0)],mcnemar_exact_p=p,year_counts=dict(years),
                verified_point_in_time=False,verified_line_order=False)
if __name__=='__main__':print(json.dumps(audit_csv(sys.argv[1]),ensure_ascii=False,indent=2))
