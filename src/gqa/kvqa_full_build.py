"""KVQA FINAL FULL — build the manifest for the ENTIRE eligible test split.

Identical source / split / eligibility / ordering as src/gqa/kvqa_final_build.py.
The ONLY difference: the `random.Random(SEED).sample(resolvable, 2000)` subsample is removed —
every eligible record is kept. FROZEN once written.

  eligibility (unchanged): repo seed-2026 split `test_annotation_indices`
    -> source == "kvqa"  (vizwiz archive not extracted here)
    -> image file present in data/kvqa_raw/images/kvqa/*.jpg
    -> exactly 10 human answers
    -> answer_type in {yes/no, number, other, unanswerable}
    -> ordered by annotation_index ascending

  python -m src.gqa.kvqa_full_build
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
from collections import Counter

ANN = os.environ.get("KVQA_ANNOTATIONS", "data/kvqa_raw/data/kvqa_annotations.json")
SPLIT = os.environ.get("KVQA_SPLIT", "assets/kvqa_8_1_1_seed2026.json")
IMG_GLOB = os.environ.get("KVQA_IMAGE_GLOB", "data/kvqa_raw/images/kvqa/*.jpg")
OUT = "outputs/kvqa_final_full/manifest.jsonl"
STATS = "outputs/kvqa_final_full/manifest_stats.json"
OFFICIAL_ANSWER_TYPES = {"yes/no", "number", "other", "unanswerable"}


def main():
    from src.common import resolve
    ann = json.load(open(resolve(ANN), encoding="utf-8"))
    sp = json.load(open(SPLIT, encoding="utf-8"))
    test_idx = sp["test_annotation_indices"]
    img_by_name = {os.path.basename(p): os.path.abspath(p) for p in glob.glob(str(resolve(IMG_GLOB)))}

    resolvable, exc, srcc = [], Counter(), Counter()
    for i in test_idx:
        r = ann[i]
        srcc[r["source"]] += 1
        if r["source"] != "kvqa":
            exc["not_kvqa_source"] += 1
            continue
        if r["image"] not in img_by_name:
            exc["image_file_missing"] += 1
            continue
        a = r["answers"]
        if not isinstance(a, list) or len(a) != 10:
            exc["not_exactly_10_answers"] += 1
            continue
        if r["answer_type"] not in OFFICIAL_ANSWER_TYPES:
            exc["answer_type_not_official"] += 1
            continue
        resolvable.append(i)
    resolvable.sort()   # canonical order = annotation_index ascending, NO subsample

    mpath = resolve(OUT)
    mpath.parent.mkdir(parents=True, exist_ok=True)
    if mpath.exists():
        raise SystemExit(f"{OUT} already exists — FROZEN, refusing to overwrite")

    rows = []
    for k, i in enumerate(resolvable):
        r = ann[i]
        stem = os.path.splitext(r["image"])[0]
        rows.append({
            "sample_index": k,
            "question_id": i,
            "sample_id": f"{i:06d}:{stem}",
            "id": f"{i:06d}:{stem}",
            "image_id": stem,
            "image_name": r["image"],
            "image_path": img_by_name[r["image"]],
            "source": r["source"],
            "question": str(r["question"]).strip(),
            "answers_10": [str(x["answer"]) for x in r["answers"]],
            "answer_type": r["answer_type"],
            "answerable": int(r["answerable"]),
        })
    blob = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows).encode("utf-8")
    with open(mpath, "wb") as f:
        f.write(blob)

    stats = {
        "split_file": SPLIT,
        "raw_test_split_size": len(test_idx),
        "test_source_counts": dict(srcc),
        "exclusions": dict(exc),
        "eligible_N": len(rows),
        "distinct_question_ids": len({r["question_id"] for r in rows}),
        "distinct_image_ids": len({r["image_id"] for r in rows}),
        "answer_type_counts": dict(Counter(r["answer_type"] for r in rows)),
        "answerable_counts": dict(Counter(r["answerable"] for r in rows)),
        "order": "annotation_index ascending",
        "first_id": rows[0]["id"],
        "last_id": rows[-1]["id"],
        "sha256": hashlib.sha256(blob).hexdigest(),
        "md5": hashlib.md5(blob).hexdigest(),
        "bytes": len(blob),
        "note": "FULL eligible KVQA seed-2026 test split; same source/split/filter/order as the frozen "
                "2K build with the 2000-subsample removed; FROZEN once written.",
    }
    json.dump(stats, open(resolve(STATS), "w"), ensure_ascii=False, indent=2)
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    assert stats["eligible_N"] == stats["distinct_question_ids"]


if __name__ == "__main__":
    main()
