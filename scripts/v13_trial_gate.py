#!/usr/bin/env python3
"""Fail-closed v13 test gate. Reuses previously documented v11/v12 benchmark summaries.
Never invent v13 accuracy when genuine verified pre-race lines are absent.
"""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/"src"))
from tactical_line_v13 import load_lines

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--benchmark",required=True)
    p.add_argument("--verified-lines")
    p.add_argument("--output",default="reports/v13_trial_status.json")
    args=p.parse_args()
    old=json.loads(Path(args.benchmark).read_text())
    lines=load_lines(args.verified_lines) if args.verified_lines else {}
    status="BLOCKED_NO_VERIFIED_PRERACE_LINES" if not lines else "NOT_EVALUATED_SPLIT_JOIN_REQUIRED"
    result={"status":status,
     "verified_line_races":len(lines),
     "v11_top5_pct":round(old["v11"]["top5_correct"]*100,2),
     "v12_top5_pct":round(old["v12"]["top5_correct"]*100,2),
     "baseline_test_races":old["test_n"],
     "v13_test_races":0,
     "v13_top5_pct":None,
     "v13_improvement_pp":None,
     "warning":"Baseline metrics are from an existing exploratory test; not new v13 outcomes"}
    Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    Path(args.output).write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=="__main__":main()
