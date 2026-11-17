#!/usr/bin/env python3
"""Evaluate previously saved bets only against verified outcomes.
Never infer outcomes from model predictions. Payouts are 100-yen unit payments.
Input outcome CSV headers: venue,race_no,finish,payout_100,verified_at
Use only records with verified_at nonempty and finish of 3 distinct riders.
"""
import argparse
import csv
import json
from pathlib import Path

def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--picks",required=True)
    ap.add_argument("--results",required=True)
    ap.add_argument("--output",default="reports/live_audit.json")
    args=ap.parse_args()
    picks=read_csv(args.picks)
    results=read_csv(args.results)
    lookup={}
    for row in results:
        key=(row["venue"],int(row["race_no"]))
        if key in lookup:
            raise ValueError("Duplicate verified race: "+str(key))
        finish=row["finish"].strip()
        parts=finish.split("-")
        if not row.get("verified_at","").strip() or len(parts)!=3 or len(set(parts))!=3 or not all(x.isdigit() for x in parts):
            raise ValueError("Unverified or invalid actual result: "+str(key))
        lookup[key]=row
    summary={"races_with_verified_results":0,"hit_races":0,"stake":0,"payout":0,"rows":[]}
    seen=set()
    for row in picks:
        key=(row["venue"],int(row["race_no"]))
        ticket=row["ticket"].strip()
        if len(ticket.split("-"))!=3 or len(set(ticket.split("-")))!=3: raise ValueError("Bad ticket "+ticket)
        stake=int(row["stake_yen"])
        if stake<=0 or stake%100: raise ValueError("Ticket stake must be positive multiple of 100")
        if (key,ticket) in seen:raise ValueError("Duplicate ticket")
        seen.add((key,ticket))
    grouped={}
    for row in picks:
        key=(row["venue"],int(row["race_no"]))
        grouped.setdefault(key,[]).append(row)
    for key, tickets in sorted(grouped.items()):
        if key not in lookup: continue
        result=lookup[key]
        finish=result["finish"].strip()
        rate=int(result["payout_100"])
        if rate<0:raise ValueError("Payout cannot be negative")
        spent=sum(int(t["stake_yen"]) for t in tickets)
        won=sum(rate*int(t["stake_yen"])//100 for t in tickets if t["ticket"].strip()==finish)
        firsts={t["ticket"].split("-")[0] for t in tickets}
        seconds={t["ticket"].split("-")[1] for t in tickets if t["ticket"].split("-")[0]==finish.split("-")[0]}
        summary["rows"].append({"venue":key[0],"race_no":key[1],"finish":finish,
            "payout_100":rate,"stake":spent,"return":won,"hit":bool(won),
            "winner_covered":finish.split("-")[0] in firsts,
            "second_covered_given_winner":finish.split("-")[1] in seconds})
        summary["stake"]+=spent
        summary["payout"]+=won
        summary["races_with_verified_results"]+=1
        summary["hit_races"]+=int(won>0)
    n=summary["races_with_verified_results"]
    summary["hit_rate"]=summary["hit_races"]/n if n else None
    summary["roi"]=summary["payout"]/summary["stake"] if summary["stake"] else None
    Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    Path(args.output).write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=="__main__": main()
