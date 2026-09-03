"""KVQA  Ours-DirectAS  —  NEW condition.  ADDITIVE; the frozen Ours-Raw / Ours-AS pipeline
(src/gqa/kvqa_final_gen.py + configs/gqa_final_hclt_vs_ours_as_2k.yaml) is NOT touched.

Two stages (NO Stage-3 suppressor):
  pdecomp   : QUESTION only            -> 9-key PROTECTED decomposition JSON        (text)
              (frozen 6 keys + observation_target / protected_answer_value / evidence_to_preserve)
  pevidence : IMAGE + protected decomp -> directly answer-suppressed visual guidance

Never fed to the model: image at pdecomp, gold (10 answers), the original question at pevidence,
model prediction, correctness.  Same 2K manifest as the frozen experiment (comparability).
Sharded + resumable per id per stage.

  python -m src.gqa.kvqa_directas_gen --stage pdecomp,pevidence --shard 0/6
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

STAGES = ("pdecomp", "pevidence")

_BASE_KEYS = ["target_object", "reference_object", "relation_to_check",
              "requested_property", "state_or_action_to_check", "answer_type"]
_EXTRA_KEYS = ["observation_target", "protected_answer_value", "evidence_to_preserve"]
_PKEYS = _BASE_KEYS + _EXTRA_KEYS


def _skel() -> dict:
    d = {k: None for k in _PKEYS}
    d["answer_type"] = "other"
    d["evidence_to_preserve"] = []
    return d


def fill(template: str, **kv) -> str:
    out = template
    for k, v in kv.items():
        out = out.replace("{" + k + "}", str(v))
    return out


def clean(text: str) -> str:
    """Text hygiene only: drop the Unicode replacement char U+FFFD (a decode-failure sentinel
    Qwen3-VL greedy occasionally emits as the first output token). Gold-blind, method-agnostic,
    identical to the frozen kvqa_final_gen.clean; does not change what the model computes."""
    return re.sub(r"�+", "", (text or "")).strip()


def _shard(rows, spec):
    if not spec:
        return rows
    i, n = (int(x) for x in spec.split("/"))
    return [r for k, r in enumerate(rows) if k % n == i]


def parse_protected_decomposition(text: str):
    """Same recovery ladder as the frozen parse_decomposition, extended to 9 keys.
    evidence_to_preserve is coerced to a list[str]; other new keys behave like the base keys."""
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
        return _skel(), "parse_fail"
    out, status = {}, "ok"
    for k in _PKEYS:
        v = obj.get(k, _skel()[k])
        if k == "answer_type":
            v = v if isinstance(v, str) and v.strip() else "other"
        elif k == "evidence_to_preserve":
            if isinstance(v, str):
                s = v.strip()
                v = [s] if s and s.lower() not in ("null", "none") else []
            elif isinstance(v, list):
                v = [str(x).strip() for x in v if str(x).strip()
                     and str(x).strip().lower() not in ("null", "none")]
            else:
                v = []
        elif isinstance(v, str) and v.strip().lower() in ("", "null", "none"):
            v = None
        out[k] = v
    if any(k not in obj for k in _PKEYS):
        status = "keys_filled"
    return out, status


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--directas-config", default="configs/kvqa_ours_directas.yaml")
    ap.add_argument("--qwen-config", default="configs/gqa_phase_c_prompts.yaml")
    ap.add_argument("--manifest", default="outputs/kvqa_final_hclt_vs_ours_as_2k/manifest.jsonl")
    ap.add_argument("--outdir", default="outputs/kvqa_ours_directas")
    ap.add_argument("--stage", default="pdecomp,pevidence")
    ap.add_argument("--shard", default=None)
    ap.add_argument("--prefer-gpu", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    stages = [s.strip() for s in args.stage.split(",") if s.strip() in STAGES]

    setup_runtime(prefer_gpu=args.prefer_gpu)

    import yaml
    from src.common import read_jsonl, JsonlWriter, resolve
    from src.gqa.qwen3vl import Qwen3VL

    g = yaml.safe_load(open(resolve(args.directas_config), encoding="utf-8"))
    qcfg = yaml.safe_load(open(resolve(args.qwen_config), encoding="utf-8"))
    pd_instr = g["protected_decomposition"]["instruction"]
    pd_mnt = int(g["protected_decomposition"]["max_new_tokens"])
    pe_instr = g["protected_evidence"]["instruction"]
    pe_mnt = int(g["protected_evidence"]["max_new_tokens"])
    # wiring guards: Stage 1' sees only the question; Stage 2' sees only the decomposition JSON.
    assert set(re.findall(r"\{(\w+)\}", pd_instr)) <= {"question"}, "pdecomp field leak"
    assert set(re.findall(r"\{(\w+)\}", pe_instr)) <= {"decomposition_json"}, \
        "pevidence field leak (question / gold must NOT appear in Stage 2')"

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
    print(f"[directas-gen] model={model.model_name} stages={stages} shard={args.shard} rows={len(part)}", flush=True)

    def meta(r):
        # answers_10 is carried for downstream DIAGNOSTICS / review only — never fed to the model.
        return {"id": r["id"], "image_id": r["image_id"], "question": r["question"],
                "answer_type": r["answer_type"], "answers_10": r["answers_10"]}

    if "pdecomp" in stages:
        d = done_ids("ours_protected_decompositions", "decomposition_raw")
        todo = [r for r in part if r["id"] not in d]
        w = JsonlWriter(f"{args.outdir}/ours_protected_decompositions{sfx}.jsonl")
        print(f"[directas-gen] pdecomp: {len(todo)} to generate", flush=True)
        t0 = time.time()
        for k, r in enumerate(todo, 1):
            try:
                txt = clean(model.vqa_text(fill(pd_instr, question=r["question"]), pd_mnt))
                parsed, status = parse_protected_decomposition(txt)
                w.write({**meta(r), "decomposition_raw": txt, "decomposition": parsed, "parse_status": status})
            except Exception as e:
                w.write({"id": r["id"], "error": repr(e)})
                print(f"[directas-gen] PDECOMP FAIL {r['id']}: {e!r}", file=sys.stderr); traceback.print_exc()
            if k % 25 == 0 or k == len(todo):
                rt = k / (time.time() - t0)
                print(f"[directas-gen] pdecomp {k}/{len(todo)} {rt:.2f}/s eta~{(len(todo)-k)/max(rt,1e-6)/60:.1f}m", flush=True)
        w.close()

    if "pevidence" in stages:
        dmap = merged("ours_protected_decompositions", "decomposition")
        miss = [r["id"] for r in part if r["id"] not in dmap]
        if miss:
            sys.exit(f"[directas-gen] pevidence: {len(miss)} rows lack a protected decomposition (e.g. {miss[:3]})")
        d = done_ids("ours_directas_evidence", "directas_evidence")
        todo = [r for r in part if r["id"] not in d]
        w = JsonlWriter(f"{args.outdir}/ours_directas_evidence{sfx}.jsonl")
        print(f"[directas-gen] pevidence: {len(todo)} to generate", flush=True)
        t0 = time.time()
        for k, r in enumerate(todo, 1):
            dj = json.dumps(dmap[r["id"]], ensure_ascii=False, indent=2)
            try:
                ev = clean(model.caption(r["image_path"], fill(pe_instr, decomposition_json=dj), pe_mnt))
                w.write({**meta(r), "decomposition_json": dj, "directas_evidence": ev, "len": len(ev)})
            except Exception as e:
                w.write({"id": r["id"], "error": repr(e)})
                print(f"[directas-gen] PEVIDENCE FAIL {r['id']}: {e!r}", file=sys.stderr); traceback.print_exc()
            if k % 25 == 0 or k == len(todo):
                rt = k / (time.time() - t0)
                print(f"[directas-gen] pevidence {k}/{len(todo)} {rt:.2f}/s eta~{(len(todo)-k)/max(rt,1e-6)/60:.1f}m", flush=True)
        w.close()

    print("[directas-gen] done.", flush=True)


if __name__ == "__main__":
    main()
