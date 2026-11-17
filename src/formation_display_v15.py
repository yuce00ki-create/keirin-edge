"""Lossless formation display for v15 free-24 trifecta tickets."""
from collections import defaultdict
from itertools import product

def render_formations(tickets):
    triples = [tuple(int(x) for x in ticket[:3]) for ticket in tickets]
    if len(triples) != len(set(triples)) or any(len(set(t)) != 3 for t in triples):
        raise ValueError("Duplicate or invalid tickets")
    grouped = defaultdict(set)
    for a, b, c in triples:
        grouped[(a,b)].add(c)
    forms = [{"first":[a], "second":[b], "third":sorted(cs)} for (a,b),cs in sorted(grouped.items())]
    assert_exact_coverage(triples, forms)
    return forms

def assert_exact_coverage(tickets, formations):
    expected = {tuple(t[:3]) for t in tickets}
    expanded = []
    for f in formations:
        expanded.extend(t for t in product(f["first"], f["second"], f["third"]) if len(set(t)) == 3)
    if set(expanded) != expected or len(expanded) != len(expected):
        raise AssertionError("Added, missing or duplicate tickets")
    return True

if __name__ == "__main__":
    ts = [(1,2,3), (1,2,4), (1,4,3), (3,1,2)]
    print(render_formations(ts))