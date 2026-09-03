"""KVQA — re-solve MM conditions with the Yes/No Format-Corrected common solver.

Only difference from the frozen KVQA common solver: ONE line in text/mm/iq prompt asks for
English "Yes"/"No" instead of Korean "예/아니오" (KVQA stores yes/no gold in English).
Read from configs/kvqa_final_solver_yesno_fixed.yaml. Everything else — the other rules,
decoding (greedy, max_new_tokens=20), the fill() substitution, no answer_type hint,
no method-specific prompt — is imported verbatim from the frozen src/gqa/kvqa_final_solve.

Representations are REUSED, never regenerated:
  generic / hclt / as  <- --repr-dir  (frozen outputs/kvqa_final_hclt_vs_ours_as_2k)
  directas             <- --directas-dir (outputs/kvqa_ours_directas/full2k)
Predictions are written to a NEW --outdir; frozen predictions are never touched.

  python -m src.gqa.kvqa_yesno_fixed_solve --shard 0/10
"""
from __future__ import annotations

import argparse
import glob
import sys
import time
import traceback

from src.runtime import setup_runtime   # MUST precede torch
from src.gqa.kvqa_final_solve import fill, _shard, _load   # frozen helpers, unchanged

# cond -> (uses_image, source_key_or_None, prompt_key, out_name)
COND = {
    "iq":          (True,  None,       "iq_prompt", "iq"),
    "generic_mm":  (True,  "generic",  "mm_prompt", "generic_mm"),
    "hclt_mm":     (True,  "hclt",     "mm_prompt", "hclt_mm"),
    "as_mm":       (True,  "as",       "mm_prompt", "as_mm"),
    "raw_mm":      (True,  "raw",      "mm_prompt", "raw_mm"),
    "directas_mm": (True,  "directas", "mm_prompt", "directas_mm"),
}
SRC = {  # (glob, id_key, text_key) — same keys as the frozen solver + directas
    "generic":  ("generic_captions*.jsonl",         "id", "generic_caption"),
    "hclt":     ("hclt_high_density.jsonl",          "id", "hclt_high_density_caption"),
    "raw":      ("ours_raw_evidence*.jsonl",         "id", "raw_evidence"),
    "as":       ("ours_suppressed_evidence*.jsonl",  "id", "suppressed_evidence"),
    "directas": ("ours_directas_evidence*.jsonl",    "id", "directas_evidence"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/kvqa_final_solver_yesno_fixed.yaml")
    ap.add_argument("--qwen-config", default="configs/gqa_phase_c_prompts.yaml")
    ap.add_argument("--manifest", default="outputs/kvqa_final_hclt_vs_ours_as_2k/manifest.jsonl")
    ap.add_argument("--repr-dir", default="outputs/kvqa_final_hclt_vs_ours_as_2k")
    ap.add_argument("--directas-dir", default="outputs/kvqa_ours_directas/full2k")
    ap.add_argument("--outdir", default="outputs/kvqa_solver_yesno_fixed_2k")
    ap.add_argument("--conditions", default="iq,generic_mm,hclt_mm,as_mm,directas_mm")
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

    rdir = str(resolve(args.repr_dir))
    ddir = str(resolve(args.directas_dir))
    ctx = {}
    for name in set(COND[c][1] for c in conds if COND[c][1]):
        d = ddir if name == "directas" else rdir
        ctx[name] = _load(d, *SRC[name])
        miss = [r["id"] for r in part if r["id"] not in ctx[name]]
        if miss:
            sys.exit(f"[yesno-solve] {name}: missing representation for {len(miss)} rows (e.g. {miss[:3]})")

    outd = str(resolve(args.outdir))
    import os
    os.makedirs(outd, exist_ok=True)
    sfx = "" if not args.shard else f".shard{args.shard.replace('/', '-')}"
    writers = {c: JsonlWriter(f"{outd}/predictions_{COND[c][3]}{sfx}.jsonl") for c in conds}
    done = {c: {r["id"] for r in read_jsonl(f"{outd}/predictions_{COND[c][3]}{sfx}.jsonl") if "pred" in r}
            for c in conds}

    model = Qwen3VL(qcfg)
    print(f"[yesno-solve] model={model.model_name} conds={conds} shard={args.shard} "
          f"rows={len(part)} mnt={mnt}", flush=True)

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
                print(f"[yesno-solve] FAIL {c} {r['id']}: {e!r}", file=sys.stderr); traceback.print_exc()
        if k % 100 == 0 or k == len(part):
            rt = k / (time.time() - t0)
            print(f"[yesno-solve] {k}/{len(part)} {rt:.2f}/s "
                  f"eta~{(len(part)-k)/max(rt,1e-6)/60:.1f}m", flush=True)
    for w in writers.values():
        w.close()
    print(f"[yesno-solve] done in {(time.time()-t0)/60:.1f}m", flush=True)


if __name__ == "__main__":
    main()
