"""KVQA FINAL — merge generation shards + FROZEN leakage detector on each representation.

Same code path (main2000_diag regex defs) for Generic / HCLT-style / Ours-Raw / Ours-AS.
Gold (10 human answers) used for DIAGNOSTICS ONLY, after generation.

  NOVEL_GOLD_MENTION (10-gold): any unique normalized human-answer surface occurs in the text
     AND that surface is not already in the question.   + NOVEL_GOLD_MATCH_COUNT.
  RESULT_PROPOSITION / CANDIDATE_ENUMERATION: unchanged main2000_diag definitions.

  python -m src.gqa.kvqa_final_diag
"""
from __future__ import annotations

import argparse
import glob
import json

from src.common import read_jsonl, resolve
from src.gqa.main2000_diag import _cand_enum, _result_prop      # FROZEN detector
from src.gqa.kvqa_softscore import normalize_answer             # KVQA official-derived normalization

REPR = {
    "generic": ("generic_captions*.jsonl",        "id", "generic_caption"),
    "hclt":    ("hclt_high_density.jsonl",         "id", "hclt_high_density_caption"),
    "raw":     ("ours_raw_evidence*.jsonl",        "id", "raw_evidence"),
    "as":      ("ours_suppressed_evidence*.jsonl", "id", "suppressed_evidence"),
}


def _merge(outdir, pattern, id_key, text_key):
    out = {}
    for fp in sorted(glob.glob(f"{outdir}/{pattern}")):
        for r in read_jsonl(fp):
            v = r.get(text_key)
            if v is not None and v != "" and r.get(id_key):
                out[r[id_key]] = v
    return out


def _nospace(s):
    return normalize_answer(s).replace(" ", "")


def _flags(text, unique_gold_ns, q_ns):
    t = text or ""
    t_ns = _nospace(t)
    hits = [g for g in unique_gold_ns if g and g in t_ns]
    novel = [g for g in hits if g not in q_ns]
    return {
        "present": bool(t), "len": len(t),
        "gold_surface_count": len(hits),
        "novel_gold_match_count": len(novel),
        "any_novel_gold_mention": int(len(novel) > 0),
        "result_proposition": _result_prop(t),
        "candidate_enumeration": _cand_enum(t),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="outputs/kvqa_final_hclt_vs_ours_as_2k/manifest.jsonl")
    ap.add_argument("--dir", default="outputs/kvqa_final_hclt_vs_ours_as_2k")
    args = ap.parse_args()
    d = str(resolve(args.dir))
    man = {r["id"]: r for r in read_jsonl(resolve(args.manifest))}
    texts = {name: _merge(d, *spec) for name, spec in REPR.items()}

    # canonical merged generation files
    for pat, key, canon in [
        ("generic_captions.shard*.jsonl", "generic_caption", "generic_captions.jsonl"),
        ("ours_decompositions.shard*.jsonl", "decomposition_raw", "ours_decompositions.jsonl"),
        ("ours_raw_evidence.shard*.jsonl", "raw_evidence", "ours_raw_evidence.jsonl"),
        ("ours_suppressed_evidence.shard*.jsonl", "suppressed_evidence", "ours_suppressed_evidence.jsonl"),
    ]:
        rows = {}
        for fp in sorted(glob.glob(f"{d}/{pat}")):
            for r in read_jsonl(fp):
                if r.get(key) is not None:
                    rows[r["id"]] = r
        if rows:
            with open(f"{d}/{canon}", "w", encoding="utf-8") as f:
                for sid in man:
                    if sid in rows:
                        f.write(json.dumps(rows[sid], ensure_ascii=False) + "\n")

    per = {name: [] for name in REPR}
    for sid, m in man.items():
        uniq = sorted({_nospace(a) for a in m["answers_10"]} - {""})
        q_ns = _nospace(m["question"])
        for name in REPR:
            fl = _flags(texts[name].get(sid, ""), uniq, q_ns)
            per[name].append({"id": sid, "answer_type": m["answer_type"],
                              "question": m["question"], "unique_gold": uniq, **fl})

    for name in REPR:
        with open(f"{d}/leakage_{name}.jsonl", "w", encoding="utf-8") as f:
            for r in per[name]:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n = len(man)
    print(f"[kvqa-diag] n={n}  (gold used for diagnostics only)")
    print(f"{'repr':9} {'present':>8} {'avg_len':>8} {'NOVEL_GOLD':>14} {'RESULT_PROP':>13} {'CAND_ENUM':>11} {'avg_match_ct':>12}")
    for name in ("generic", "hclt", "raw", "as"):
        rr = per[name]
        p = sum(1 for r in rr if r["present"])
        al = sum(r["len"] for r in rr) / max(p, 1)
        ng = sum(r["any_novel_gold_mention"] for r in rr)
        rp = sum(r["result_proposition"] for r in rr)
        ce = sum(r["candidate_enumeration"] for r in rr)
        mc = sum(r["novel_gold_match_count"] for r in rr) / n
        print(f"{name:9} {p:>8} {al:>8.0f} {ng:>7}/{n} {ng/n:>5.1%} {rp:>7}/{n} {rp/n:>4.1%} {ce:>5}/{n} {ce/n:>4.1%} {mc:>12.3f}")
    print(f"[kvqa-diag] -> {d}/leakage_{{generic,hclt,raw,as}}.jsonl")


if __name__ == "__main__":
    main()
