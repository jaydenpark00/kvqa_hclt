"""KVQA Ours-DirectAS — deterministic 500-sample PILOT subset.

Source of truth: the frozen KVQA final fixed-2K manifest. No new sampling, no resampling:
the pilot is the FIRST 500 rows of that manifest, order preserved. Writes the subset to
two locations and prints its sha256 / md5.

  python -m src.gqa.kvqa_directas_pilot_build
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os

from src.common import read_jsonl, resolve

SRC = "outputs/kvqa_final_hclt_vs_ours_as_2k/manifest.jsonl"
OUT1 = "outputs/kvqa_ours_directas/pilot500_manifest.jsonl"
OUT2 = "outputs/kvqa_ours_directas/pilot500/pilot500_manifest.jsonl"
N = 500


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--n", type=int, default=N)
    args = ap.parse_args()

    rows = read_jsonl(resolve(args.src))
    if len(rows) < args.n:
        raise SystemExit(f"[pilot-build] source has only {len(rows)} rows (< {args.n})")
    pick = rows[: args.n]                       # first N, order preserved, NO resampling
    ids = [r["id"] for r in pick]
    if len(set(ids)) != args.n:
        raise SystemExit("[pilot-build] non-unique ids in first N — abort")

    blob = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in pick).encode("utf-8")
    for out in (OUT1, OUT2):
        p = resolve(out)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "wb") as f:
            f.write(blob)

    print(f"[pilot-build] source      : {args.src}")
    print(f"[pilot-build] N           : {args.n}   (first {args.n} of {len(rows)}, order preserved, no resampling)")
    print(f"[pilot-build] id[0]       : {ids[0]}")
    print(f"[pilot-build] id[-1]      : {ids[-1]}")
    print(f"[pilot-build] bytes       : {len(blob)}")
    print(f"[pilot-build] sha256      : {hashlib.sha256(blob).hexdigest()}")
    print(f"[pilot-build] md5         : {hashlib.md5(blob).hexdigest()}")
    print(f"[pilot-build] -> {OUT1}")
    print(f"[pilot-build] -> {OUT2}")


if __name__ == "__main__":
    main()
