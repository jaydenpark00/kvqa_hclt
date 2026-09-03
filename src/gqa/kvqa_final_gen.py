"""KVQA FINAL — generation stages on the fixed 2K.  FROZEN prompts, NO gold anywhere.

stages (identical to the Korean-GQA final run; prompts read from the GQA frozen config):
  generic : IMAGE only              -> question-agnostic detailed generic caption   (300 tok)
  decomp  : QUESTION only           -> structured decomposition JSON                (128 tok, text)
  raw     : IMAGE + decomposition   -> raw structured visual evidence               (300 tok)
  supp    : QUESTION + raw evidence -> gold-free answer-suppressed evidence         (420 tok, text)

supp: model NEVER sees image / gold(10 answers) / prediction / correctness / leakage labels.
Sharded + resumable per id per stage.

  python -m src.gqa.kvqa_final_gen --stage generic,decomp,raw,supp --shard 0/6
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import sys
import time
import traceback

from src.runtime import setup_runtime   # MUST precede torch

STAGES = ("generic", "decomp", "raw", "supp")
_DECOMP_KEYS = ["target_object", "reference_object", "relation_to_check",
                "requested_property", "state_or_action_to_check", "answer_type"]
_SKELETON = {k: (None if k != "answer_type" else "other") for k in _DECOMP_KEYS}


def fill(template: str, **kv) -> str:
    out = template
    for k, v in kv.items():
        out = out.replace("{" + k + "}", str(v))
    return out


def clean(text: str) -> str:
    """Text hygiene only: drop the Unicode replacement char U+FFFD (a decode-failure sentinel;
    Qwen3-VL greedy occasionally emits a partial-byte BPE token as the very first output token).
    Gold-blind, method-agnostic, applied to every stage output. Does not change what the model
    computes; only removes malformed bytes so downstream text/leakage handling is well-formed."""
    return re.sub(r"�+", "", (text or "")).strip()


def _shard(rows, spec):
    if not spec:
        return rows
    i, n = (int(x) for x in spec.split("/"))
    return [r for k, r in enumerate(rows) if k % n == i]


def parse_decomposition(text: str):
    raw = text or ""
    m = re.search(r"\{.*\}", raw, re.S)
    obj = None
    if m:
        blob = m.group(0)
        for cand in (blob, blob.replace("'", '"'), re.sub(r",\s*([}\]])", r"\1", blob)):
            try:
                obj = json.loads(cand)
                break
            except Exception:
                continue
    if not isinstance(obj, dict):
        return dict(_SKELETON), "parse_fail"
    out, status = {}, "ok"
    for k in _DECOMP_KEYS:
        v = obj.get(k, _SKELETON[k])
        if k == "answer_type":
            v = v if isinstance(v, str) and v.strip() else "other"
        elif isinstance(v, str) and v.strip().lower() in ("", "null", "none"):
            v = None
        out[k] = v
    if any(k not in obj for k in _DECOMP_KEYS):
        status = "keys_filled"
    return out, status


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gqa-config", default="configs/gqa_final_hclt_vs_ours_as_2k.yaml")
    ap.add_argument("--qwen-config", default="configs/gqa_phase_c_prompts.yaml")
    ap.add_argument("--manifest", default="outputs/kvqa_final_hclt_vs_ours_as_2k/manifest.jsonl")
    ap.add_argument("--outdir", default="outputs/kvqa_final_hclt_vs_ours_as_2k")
    ap.add_argument("--stage", default="generic,decomp,raw,supp")
    ap.add_argument("--shard", default=None)
    ap.add_argument("--prefer-gpu", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    stages = [s.strip() for s in args.stage.split(",") if s.strip() in STAGES]

    setup_runtime(prefer_gpu=args.prefer_gpu)

    import yaml
    from src.common import read_jsonl, JsonlWriter, resolve
    from src.gqa.qwen3vl import Qwen3VL

    g = yaml.safe_load(open(resolve(args.gqa_config), encoding="utf-8"))
    qcfg = yaml.safe_load(open(resolve(args.qwen_config), encoding="utf-8"))
    g_instr = g["generic_caption"]["instruction"];  g_mnt = int(g["generic_caption"]["max_new_tokens"])
    d_instr = g["ours_decomposition"]["instruction"]; d_mnt = int(g["ours_decomposition"]["max_new_tokens"])
    r_instr = g["ours_raw_evidence"]["instruction"];  r_mnt = int(g["ours_raw_evidence"]["max_new_tokens"])
    s_instr = g["suppressor"]["adapted_evidence_suppressor_instruction"]
    s_mnt = int(g["suppressor"]["max_new_tokens"])
    assert set(re.findall(r"\{(\w+)\}", s_instr)) <= {"question", "raw_evidence"}, "suppressor field leak"

    rows = read_jsonl(resolve(args.manifest))
    if args.limit:
        rows = rows[: args.limit]
    sfx = "" if not args.shard else f".shard{args.shard.replace('/', '-')}"
    part = _shard(rows, args.shard)

    def done_ids(base, key):
        d = set()
        for fp in glob.glob(str(resolve(f"{args.outdir}/{base}*.jsonl"))):
            for r in read_jsonl(fp):
                if r.get(key) is not None and r.get(key) != "":
                    d.add(r["id"])
        return d

    def merged(base, key):
        out = {}
        for fp in sorted(glob.glob(str(resolve(f"{args.outdir}/{base}*.jsonl")))):
            for r in read_jsonl(fp):
                if r.get(key) is not None and r.get(key) != "":
                    out[r["id"]] = r[key]
        return out

    model = Qwen3VL(qcfg)
    print(f"[kvqa-gen] model={model.model_name} stages={stages} shard={args.shard} rows={len(part)}", flush=True)

    def meta(r):
        return {"id": r["id"], "image_id": r["image_id"], "question": r["question"],
                "answer_type": r["answer_type"], "answers_10": r["answers_10"]}

    if "generic" in stages:
        d = done_ids("generic_captions", "generic_caption")
        todo = [r for r in part if r["id"] not in d]
        w = JsonlWriter(f"{args.outdir}/generic_captions{sfx}.jsonl")
        print(f"[kvqa-gen] generic: {len(todo)} to generate", flush=True)
        t0 = time.time()
        for k, r in enumerate(todo, 1):
            try:
                cap = clean(model.caption(r["image_path"], g_instr, g_mnt))
                w.write({**meta(r), "generic_caption": cap, "len": len(cap)})
            except Exception as e:
                w.write({"id": r["id"], "error": repr(e)})
                print(f"[kvqa-gen] GENERIC FAIL {r['id']}: {e!r}", file=sys.stderr); traceback.print_exc()
            if k % 25 == 0 or k == len(todo):
                rt = k / (time.time() - t0)
                print(f"[kvqa-gen] generic {k}/{len(todo)} {rt:.2f}/s eta~{(len(todo)-k)/max(rt,1e-6)/60:.1f}m", flush=True)
        w.close()

    if "decomp" in stages:
        d = done_ids("ours_decompositions", "decomposition_raw")
        todo = [r for r in part if r["id"] not in d]
        w = JsonlWriter(f"{args.outdir}/ours_decompositions{sfx}.jsonl")
        print(f"[kvqa-gen] decomp: {len(todo)} to generate", flush=True)
        t0 = time.time()
        for k, r in enumerate(todo, 1):
            try:
                txt = clean(model.vqa_text(fill(d_instr, question=r["question"]), d_mnt))
                parsed, status = parse_decomposition(txt)
                w.write({**meta(r), "decomposition_raw": txt, "decomposition": parsed, "parse_status": status})
            except Exception as e:
                w.write({"id": r["id"], "error": repr(e)})
                print(f"[kvqa-gen] DECOMP FAIL {r['id']}: {e!r}", file=sys.stderr); traceback.print_exc()
            if k % 25 == 0 or k == len(todo):
                rt = k / (time.time() - t0)
                print(f"[kvqa-gen] decomp {k}/{len(todo)} {rt:.2f}/s eta~{(len(todo)-k)/max(rt,1e-6)/60:.1f}m", flush=True)
        w.close()

    if "raw" in stages:
        dmap = merged("ours_decompositions", "decomposition")
        miss = [r["id"] for r in part if r["id"] not in dmap]
        if miss:
            sys.exit(f"[kvqa-gen] raw: {len(miss)} rows lack a decomposition (e.g. {miss[:3]})")
        d = done_ids("ours_raw_evidence", "raw_evidence")
        todo = [r for r in part if r["id"] not in d]
        w = JsonlWriter(f"{args.outdir}/ours_raw_evidence{sfx}.jsonl")
        print(f"[kvqa-gen] raw: {len(todo)} to generate", flush=True)
        t0 = time.time()
        for k, r in enumerate(todo, 1):
            dj = json.dumps(dmap[r["id"]], ensure_ascii=False, indent=2)
            try:
                ev = clean(model.caption(r["image_path"], fill(r_instr, decomposition_json=dj), r_mnt))
                w.write({**meta(r), "decomposition_json": dj, "raw_evidence": ev, "len": len(ev)})
            except Exception as e:
                w.write({"id": r["id"], "error": repr(e)})
                print(f"[kvqa-gen] RAW FAIL {r['id']}: {e!r}", file=sys.stderr); traceback.print_exc()
            if k % 25 == 0 or k == len(todo):
                rt = k / (time.time() - t0)
                print(f"[kvqa-gen] raw {k}/{len(todo)} {rt:.2f}/s eta~{(len(todo)-k)/max(rt,1e-6)/60:.1f}m", flush=True)
        w.close()

    if "supp" in stages:
        emap = merged("ours_raw_evidence", "raw_evidence")
        miss = [r["id"] for r in part if r["id"] not in emap]
        if miss:
            sys.exit(f"[kvqa-gen] supp: {len(miss)} rows lack raw evidence (e.g. {miss[:3]})")
        d = done_ids("ours_suppressed_evidence", "suppressed_evidence")
        todo = [r for r in part if r["id"] not in d]
        w = JsonlWriter(f"{args.outdir}/ours_suppressed_evidence{sfx}.jsonl")
        print(f"[kvqa-gen] supp: {len(todo)} to rewrite", flush=True)
        t0 = time.time()
        for k, r in enumerate(todo, 1):
            raw_ev = emap[r["id"]]
            prompt = fill(s_instr, question=r["question"], raw_evidence=raw_ev)   # question + raw evidence ONLY
            try:
                s = clean(model.vqa_text(prompt, s_mnt))
                w.write({**meta(r), "raw_evidence": raw_ev, "suppressed_evidence": s,
                         "raw_len": len(raw_ev), "supp_len": len(s)})
            except Exception as e:
                w.write({"id": r["id"], "error": repr(e)})
                print(f"[kvqa-gen] SUPP FAIL {r['id']}: {e!r}", file=sys.stderr); traceback.print_exc()
            if k % 25 == 0 or k == len(todo):
                rt = k / (time.time() - t0)
                print(f"[kvqa-gen] supp {k}/{len(todo)} {rt:.2f}/s eta~{(len(todo)-k)/max(rt,1e-6)/60:.1f}m", flush=True)
        w.close()

    print("[kvqa-gen] done.", flush=True)


if __name__ == "__main__":
    main()
