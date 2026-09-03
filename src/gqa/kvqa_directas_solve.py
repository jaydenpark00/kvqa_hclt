"""KVQA Ours-DirectAS — the FROZEN KVQA solver applied to DirectAS-r2 evidence.

Two conditions only:
  directas_text : DirectAS evidence + Q               (text-only,  mm=False)
  directas_mm   : Image + DirectAS evidence + Q       (multimodal, mm=True)

The solver prompt text, max_new_tokens and greedy decoding are read VERBATIM from the frozen
configs/kvqa_final_hclt_vs_ours_as_2k.yaml `solver` block — the identical `text_prompt` /
`mm_prompt` that Ours-Raw and Ours-AS use. No method-specific prompt, no answer_type hint.
The frozen src/gqa/kvqa_final_solve.py is not modified. Sharded + resumable per id per condition.

  python -m src.gqa.kvqa_directas_solve --shard 0/10
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
import time
import traceback

from src.runtime import setup_runtime   # MUST precede torch

# name -> (uses_image, solver prompt key, output name)
COND = {
    "directas_text": (False, "text_prompt", "directas_text"),
    "directas_mm":   (True,  "mm_prompt",   "directas_mm"),
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--solver-config", default="configs/kvqa_final_hclt_vs_ours_as_2k.yaml")
    ap.add_argument("--qwen-config", default="configs/gqa_phase_c_prompts.yaml")
    ap.add_argument("--manifest", default="outputs/kvqa_ours_directas/pilot500/pilot500_manifest.jsonl")
    ap.add_argument("--dir", default="outputs/kvqa_ours_directas/pilot500")
    ap.add_argument("--evidence-glob", default="ours_directas_evidence*.jsonl")
    ap.add_argument("--conditions", default="directas_text,directas_mm")
    ap.add_argument("--shard", default=None)
    ap.add_argument("--prefer-gpu", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    conds = [c.strip() for c in args.conditions.split(",") if c.strip() in COND]

    setup_runtime(prefer_gpu=args.prefer_gpu)

    import yaml
    from src.common import read_jsonl, JsonlWriter, resolve
    from src.gqa.qwen3vl import Qwen3VL

    cfg = yaml.safe_load(open(resolve(args.solver_config), encoding="utf-8"))["solver"]
    qcfg = yaml.safe_load(open(resolve(args.qwen_config), encoding="utf-8"))
    mnt = int(cfg["max_new_tokens"])
    tmpl = {k: cfg[k] for k in ("text_prompt", "mm_prompt")}

    d = str(resolve(args.dir))
    ev = {}
    for fp in sorted(glob.glob(f"{d}/{args.evidence_glob}")):
        for r in read_jsonl(fp):
            if r.get("directas_evidence") and r.get("id"):
                ev[r["id"]] = r["directas_evidence"]

    rows = read_jsonl(resolve(args.manifest))
    if args.limit:
        rows = rows[: args.limit]
    part = _shard(rows, args.shard)
    miss = [r["id"] for r in part if r["id"] not in ev]
    if miss:
        sys.exit(f"[directas-solve] missing DirectAS evidence for {len(miss)} rows (e.g. {miss[:3]})")

    sfx = "" if not args.shard else f".shard{args.shard.replace('/', '-')}"
    writers = {c: JsonlWriter(f"{d}/predictions_{COND[c][2]}{sfx}.jsonl") for c in conds}
    done = {c: {r["id"] for r in read_jsonl(f"{d}/predictions_{COND[c][2]}{sfx}.jsonl") if "pred" in r}
            for c in conds}

    model = Qwen3VL(qcfg)
    print(f"[directas-solve] model={model.model_name} conds={conds} shard={args.shard} "
          f"rows={len(part)} mnt={mnt}", flush=True)

    t0 = time.time()
    for k, r in enumerate(part, 1):
        for c in conds:
            if r["id"] in done[c]:
                continue
            use_img, pkey, _ = COND[c]
            prompt = fill(tmpl[pkey], caption=ev[r["id"]].strip(), question=r["question"])
            try:
                pred = (model.vqa_image(r["image_path"], prompt, mnt) if use_img
                        else model.vqa_text(prompt, mnt))
                writers[c].write({"id": r["id"], "image_id": r["image_id"], "condition": c,
                                  "question": r["question"], "answers_10": r["answers_10"],
                                  "answer_type": r["answer_type"], "pred": pred})
            except Exception as e:
                writers[c].write({"id": r["id"], "condition": c, "error": repr(e)})
                print(f"[directas-solve] FAIL {c} {r['id']}: {e!r}", file=sys.stderr); traceback.print_exc()
        if k % 100 == 0 or k == len(part):
            rt = k / (time.time() - t0)
            print(f"[directas-solve] {k}/{len(part)} {rt:.2f}/s "
                  f"eta~{(len(part)-k)/max(rt,1e-6)/60:.1f}m", flush=True)
    for w in writers.values():
        w.close()
    print(f"[directas-solve] done in {(time.time()-t0)/60:.1f}m", flush=True)


if __name__ == "__main__":
    main()
