#!/usr/bin/env python3
"""Convert a backlog.py run CSV into a sidecar-compatible cache JSON.

Backlog runs on a fast machine produce a row per asset (matched or not). The
sidecar uses a JSON cache to avoid reclassifying things. Same data underneath,
just a one-time format flip so the sidecar starts seeded instead of spending
hours on the first full scan.

Usage (from the repo root):
    python -m classifier.cache_from_csv <run-csv> <out-cache-json> \
        [--classes cat dog] [--model yolo11m]
"""

import argparse
import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path", help="Input backlog.py CSV")
    ap.add_argument("out_path", help="Output cache JSON")
    ap.add_argument("--classes", nargs="+", default=["cat", "dog"],
                    help="Class names whose confidence columns to capture")
    ap.add_argument("--model", default="yolo11m", help="Model name stamped into the cache")
    args = ap.parse_args()

    csv_path = Path(args.csv_path)
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Use the CSV's mtime as the classified_at timestamp - it's the closest thing
    # we have to "when YOLO scored this asset" without a per-row timestamp.
    mtime = datetime.fromtimestamp(os.path.getmtime(csv_path), timezone.utc)
    classified_at = mtime.strftime("%Y-%m-%dT%H:%M:%SZ")

    entries = {}
    counts = {"classified": 0, "video_long": 0, "errors_dropped": 0}
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            asset_id = row["asset_id"]
            kind = row["kind"]
            err = row.get("error", "")

            if row.get("cat_conf"):
                # Successful classification - capture configured classes' confs.
                confs = {}
                for klass in args.classes:
                    col = f"{klass}_conf"
                    if row.get(col):
                        confs[klass] = round(float(row[col]), 4)
                entries[asset_id] = {
                    "kind": kind,
                    "classified_at": classified_at,
                    "frames_scored": int(row.get("frames_scored") or 0),
                    "confs": confs,
                }
                counts["classified"] += 1
            elif err.startswith("skip: duration"):
                # Long video - cache the skip so sidecar doesn't re-fetch metadata
                # to make the same decision.
                duration_s = float(row.get("duration_s") or 0.0)
                entries[asset_id] = {
                    "kind": kind,
                    "classified_at": classified_at,
                    "skipped": "video_too_long",
                    "duration_s": duration_s,
                    "confs": {klass: 0.0 for klass in args.classes},
                }
                counts["video_long"] += 1
            else:
                # Decode/network errors - let sidecar retry on next cycle.
                counts["errors_dropped"] += 1

    payload = {"version": 1, "model": args.model, "entries": entries}
    with out_path.open("w") as f:
        json.dump(payload, f, separators=(",", ":"))

    print(f"wrote {out_path} ({len(entries)} entries)")
    for k, v in counts.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
