"""KVQA FINAL — the 9 answer conditions on the fixed 2K.  ONE common solver, greedy, 20 tok.

  iq            Image + Question
  generic_text  Generic caption + Q               (text-only)
  generic_mm    Image + Generic caption + Q
  hclt_text     HCLT-style caption + Q            (text-only)
  hclt_mm       Image + HCLT-style caption + Q
  raw_text      Raw structured evidence + Q       (text-only)
  raw_mm        Image + Raw evidence + Q
  as_text       Answer-suppressed evidence + Q    (text-only)
  as_mm         Image + Answer-suppressed evidence + Q

Solver prompts (configs/kvqa_final_hclt_vs_ours_as_2k.yaml) are identical for every method;
answer_type is never passed. Sharded + resumable per id per condition.

  python -m src.gqa.kvqa_final_solve --shard 0/6 --conditions iq,generic_text,generic_mm,hclt_text,hclt_mm,raw_text,raw_mm,as_text,as_mm
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
import time
import traceback

from src.runtime import setup_runtime   # MUST precede torch

# cond -> (uses_image, caption_source_or_None, prompt_key, out_name)
COND = {
    "iq":           (True,  None,       "iq_prompt",   "iq"),
    "generic_text": (False, "generic",  "text_prompt", "generic_text"),
    "generic_mm":   (True,  "generic",  "mm_prompt",   "generic_mm"),
    "hclt_text":    (False, "hclt",     "text_prompt", "hclt_text"),
    "hclt_mm":      (True,  "hclt",     "mm_prompt",   "hclt_mm"),
    "raw_text":     (False, "raw",      "text_prompt", "raw_text"),
    "raw_mm":       (True,  "raw",      "mm_prompt",   "raw_mm"),
    "as_text":      (False, "as",       "text_prompt", "as_text"),
    "as_mm":        (True,  "as",       "mm_prompt",   "as_mm"),
}
SRC = {
    "generic": ("generic_captions*.jsonl",        "id",        "generic_caption"),
    "hclt":    ("hclt_high_density.jsonl",         "id",        "hclt_high_density_caption"),
    "raw":     ("ours_raw_evidence*.jsonl",        "id",        "raw_evidence"),
    "as":      ("ours_suppressed_evidence*.jsonl", "id",        "suppressed_evidence"),
}


def fill(t, **kv):
    for k, v in kv.items():
        t = t.replace("{" + k + "}", str(v))
    return t


def _shard(rows, spec):
    if not spec:
        return rows
    i, n = (int(x) for x in spec.split("/"))
    return [r for k, r in enumerate(rows) if k % n == i]


def _load(d, pattern, id_key, text_key):
    out = {}
    for fp in sorted(glob.glob(f"{d}/{pattern}")):
        for line in open(fp, encoding="utf-8"):
            r = json.loads(line)
            if r.get(text_key) is not None and r.get(text_key) != "" and r.get(id_key):
                out[r[id_key]] = r[text_key]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/kvqa_final_hclt_vs_ours_as_2k.yaml")
    ap.add_argument("--qwen-config", default="configs/gqa_phase_c_prompts.yaml")
    ap.add_argument("--manifest", default="outputs/kvqa_final_hclt_vs_ours_as_2k/manifest.jsonl")
    ap.add_argument("--dir", default="outputs/kvqa_final_hclt_vs_ours_as_2k")
    ap.add_argument("--conditions", default=",".join(COND))
    ap.add_argument("--shard", default=None)
    ap.add_argument("--prefer-gpu", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    conds = [c.strip() for c in args.conditions.split(",") if c.strip() in COND]

    setup_runtime(prefer_gpu=args.prefer_gpu)

    import yaml
    from src.common import read_jsonl, JsonlWriter, resolve
    from src.gqa.qwen3vl import Qwen3VL

    cfg = yaml.safe_load(open(resolve(args.config), encoding="utf-8"))["solver"]
    qcfg = yaml.safe_load(open(resolve(args.qwen_config), encoding="utf-8"))
    mnt = int(cfg["max_new_tokens"])
    tmpl = {k: cfg[k] for k in ("text_prompt", "mm_prompt", "iq_prompt")}

    rows = read_jsonl(resolve(args.manifest))
    if args.limit:
        rows = rows[: args.limit]
    part = _shard(rows, args.shard)
    d = str(resolve(args.dir))
    ctx = {}
    for name in set(COND[c][1] for c in conds if COND[c][1]):
        ctx[name] = _load(d, *SRC[name])
        miss = [r["id"] for r in part if r["id"] not in ctx[name]]
        if miss:
            sys.exit(f"[kvqa-solve] {name}: missing representation for {len(miss)} rows (e.g. {miss[:3]})")

    sfx = "" if not args.shard else f".shard{args.shard.replace('/', '-')}"
    writers = {c: JsonlWriter(f"{d}/predictions_{COND[c][3]}{sfx}.jsonl") for c in conds}
    done = {c: {r["id"] for r in read_jsonl(f"{d}/predictions_{COND[c][3]}{sfx}.jsonl") if "pred" in r}
            for c in conds}

    model = Qwen3VL(qcfg)
    print(f"[kvqa-solve] model={model.model_name} conds={conds} shard={args.shard} rows={len(part)} mnt={mnt}", flush=True)

    t0 = time.time()
    for k, r in enumerate(part, 1):
        for c in conds:
            if r["id"] in done[c]:
                continue
            use_img, src, pkey, _ = COND[c]
            if src is None:
                prompt = fill(tmpl[pkey], question=r["question"])
            else:
                prompt = fill(tmpl[pkey], caption=ctx[src][r["id"]].strip(), question=r["question"])
            try:
                pred = (model.vqa_image(r["image_path"], prompt, mnt) if use_img
                        else model.vqa_text(prompt, mnt))
                writers[c].write({"id": r["id"], "image_id": r["image_id"], "condition": c,
                                  "question": r["question"], "answers_10": r["answers_10"],
                                  "answer_type": r["answer_type"], "pred": pred})
            except Exception as e:
                writers[c].write({"id": r["id"], "condition": c, "error": repr(e)})
                print(f"[kvqa-solve] FAIL {c} {r['id']}: {e!r}", file=sys.stderr); traceback.print_exc()
        if k % 100 == 0 or k == len(part):
            rt = k / (time.time() - t0)
            print(f"[kvqa-solve] {k}/{len(part)} {rt:.2f}/s eta~{(len(part)-k)/max(rt,1e-6)/60:.1f}m", flush=True)
    for w in writers.values():
        w.close()
    print(f"[kvqa-solve] done in {(time.time()-t0)/60:.1f}m", flush=True)


if __name__ == "__main__":
    main()
