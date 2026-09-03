"""KVQA FINAL — build the ONE fixed 2K evaluation manifest (deterministic, never resampled).

  * from the repo seed-2026 split's  test_annotation_indices  (NOT the official HCLT split)
  * kvqa-source records whose image file exists  (vizwiz archive not extracted here)
  * exactly 10 human answers per question (validated by the repo loader schema)
  * random.Random(20260903).sample(sorted(resolvable), 2000); ordered by annotation_index
  * FROZEN once written

  python -m src.gqa.kvqa_final_build
"""
from __future__ import annotations

import glob
import json
import os
import random
from collections import Counter

import os
ANN = os.environ.get("KVQA_ANNOTATIONS", "data/kvqa_raw/data/kvqa_annotations.json")
SPLIT = os.environ.get("KVQA_SPLIT", "assets/kvqa_8_1_1_seed2026.json")
IMG_GLOB = os.environ.get("KVQA_IMAGE_GLOB", "data/kvqa_raw/images/kvqa/*.jpg")
SEED = 20260903
N = 2000
OUT = "outputs/kvqa_final_hclt_vs_ours_as_2k/manifest.jsonl"
STATS = "outputs/kvqa_final_hclt_vs_ours_as_2k/sampling_stats.json"
OFFICIAL_ANSWER_TYPES = {"yes/no", "number", "other", "unanswerable"}


def main():
    from src.common import resolve
    ann = json.load(open(resolve(ANN), encoding="utf-8"))
    sp = json.load(open(SPLIT, encoding="utf-8"))
    test_idx = sp["test_annotation_indices"]
    img_by_name = {os.path.basename(p): os.path.abspath(p) for p in glob.glob(str(resolve(IMG_GLOB)))}

    resolvable = []
    src_counter = Counter()
    for i in test_idx:
        r = ann[i]
        src_counter[r["source"]] += 1
        if r["source"] != "kvqa":
            continue
        if r["image"] not in img_by_name:
            continue
        a = r["answers"]
        if not isinstance(a, list) or len(a) != 10:
            continue
        if r["answer_type"] not in OFFICIAL_ANSWER_TYPES:
            continue
        resolvable.append(i)

    resolvable.sort()
    if len(resolvable) < N:
        raise SystemExit(f"only {len(resolvable)} resolvable test records (< {N})")
    picked = sorted(random.Random(SEED).sample(resolvable, N))     # canonical order = annotation_index asc

    mpath = resolve(OUT)
    mpath.parent.mkdir(parents=True, exist_ok=True)
    if mpath.exists():
        raise SystemExit(f"{OUT} already exists — FROZEN, refusing to overwrite")

    rows = []
    for k, i in enumerate(picked):
        r = ann[i]
        stem = os.path.splitext(r["image"])[0]
        rows.append({
            "sample_index": k,
            "question_id": i,                       # KVQA has no explicit qid; annotation index is the id
            "sample_id": f"{i:06d}:{stem}",         # repo-format id
            "id": f"{i:06d}:{stem}",                # generic key used by the pipeline
            "image_id": stem,
            "image_name": r["image"],
            "image_path": img_by_name[r["image"]],
            "source": r["source"],
            "question": str(r["question"]).strip(),
            "answers_10": [str(x["answer"]) for x in r["answers"]],
            "answer_type": r["answer_type"],
            "answerable": int(r["answerable"]),
        })
    with open(mpath, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    stats = {
        "seed": SEED, "N": len(rows),
        "unique_question_ids": len({r["question_id"] for r in rows}),
        "unique_images": len({r["image_id"] for r in rows}),
        "split_file": SPLIT,
        "split_test_indices_total": len(test_idx),
        "test_source_counts": dict(src_counter),
        "resolvable_kvqa_test_records": len(resolvable),
        "answer_type_counts": dict(Counter(r["answer_type"] for r in rows)),
        "answerable_counts": dict(Counter(r["answerable"] for r in rows)),
        "order": "annotation_index ascending",
        "note": "fixed KVQA-2K evaluation subset; repo seed-2026 test split; kvqa-source with resolvable "
                "image; NOT an exact HCLT 2025 reproduction; FROZEN — never resampled.",
    }
    json.dump(stats, open(resolve(STATS), "w"), ensure_ascii=False, indent=2)
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    assert stats["N"] == N and stats["unique_question_ids"] == N


if __name__ == "__main__":
    main()
