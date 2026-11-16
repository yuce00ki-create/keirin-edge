#!/usr/bin/env python3
"""Public CSV acquisition and audit. No unofficial racing API or wagering automation."""
import argparse, csv, datetime as dt, hashlib, io, json, re, time
from collections import Counter
from pathlib import Path
from urllib.request import Request, urlopen

SOURCE = "https://api.github.com/repos/Kenseimk/keirin-data/contents/keirin_data?ref=main"
PATTERN = re.compile(r"^(20\d{2})_(0[1-9]|1[0-2])_keirin\.csv$")
NEEDED = {"race_id", "date", "banum", "rank", "race_score"}

def download(url, tries=3):
    err = None
    for attempt in range(tries):
        try:
            req = Request(url, headers={"User-Agent":"keirin-edge-research/1.0"})
            with urlopen(req, timeout=100) as response:
                return response.read()
        except Exception as exc:
            err = exc
            time.sleep(min(2 ** attempt, 5))
    raise RuntimeError(str(err))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutoff", default="2026-11-16")
    ap.add_argument("--max-files", type=int, default=0)
    ap.add_argument("--out", default="output")
    args = ap.parse_args()
    cutoff = dt.date.fromisoformat(args.cutoff)
    out = Path(args.out); (out / "csv").mkdir(parents=True, exist_ok=True)
    items = json.loads(download(SOURCE))
    candidates = []
    for item in items:
        match = PATTERN.match(item.get("name", ""))
        if match and item.get("download_url") and dt.date(int(match[1]), int(match[2]), 1) <= cutoff.replace(day=1):
            candidates.append(item)
    candidates.sort(key=lambda x:x["name"])
    if args.max_files > 0:
        candidates = candidates[:args.max_files]
    report = {"source":SOURCE,"cutoff":str(cutoff),"available_month_files":len(candidates),"files":[],"errors":[],"total_valid_rows":0,"unique_races":0,"excluded_post_cutoff":0}
    all_races = set()
    for item in candidates:
        try:
            data = download(item["download_url"])
            text_data = data.decode("utf-8-sig")
            reader = csv.DictReader(io.StringIO(text_data))
            missing = sorted(NEEDED - set(reader.fieldnames or []))
            if missing: raise ValueError("Missing columns: " + ",".join(missing))
            clean = io.StringIO()
            writer = csv.DictWriter(clean, fieldnames=reader.fieldnames)
            writer.writeheader()
            rows = 0; invalid = 0; dates = set()
            for row in reader:
                try:
                    day = dt.date.fromisoformat((row.get("date") or "")[:10])
                except ValueError:
                    invalid += 1
                    continue
                if day > cutoff:
                    report["excluded_post_cutoff"] += 1
                    continue
                if not (row.get("race_id") and row.get("banum")):
                    invalid += 1
                    continue
                rows += 1; dates.add(str(day))
                all_races.add((row["race_id"],str(day)))
                writer.writerow(row)
            payload = clean.getvalue().encode("utf-8")
            (out / "csv" / item["name"]).write_bytes(payload)
            report["files"].append({"file":item["name"],"rows":rows,"invalid":invalid,"dates":len(dates),"sha256":hashlib.sha256(payload).hexdigest()})
            report["total_valid_rows"] += rows
            print(f"{item['name']}: {rows} rows", flush=True)
        except Exception as exc:
            report["errors"].append({"file":item["name"],"error":str(exc)[:350]})
            print(f"{item['name']}: ERROR {exc}", flush=True)
    report["unique_races"] = len(all_races)
    (out / "audit.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({k:v for k,v in report.items() if k!="files"},ensure_ascii=False,indent=2))
    if not report["files"]:
        raise SystemExit("No valid CSV files acquired")
if __name__ == "__main__":
    main()
