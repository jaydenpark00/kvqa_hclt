"""MAIN 2K EVALUATION — merge generation shards + leakage diagnostics (CPU, gold DIAGNOSTIC-ONLY).

Merges -> canonical  qac.jsonl , strong_guides_raw.jsonl , strong_guides.jsonl
Then, using gold for DIAGNOSTICS ONLY (never regeneration / filtering):
  SG (final, suppressed):  NOVEL_GOLD_MENTION · RESULT_PROPOSITION(auto) · ANSWER_CANDIDATE_ENUMERATION(auto)
                           · GOLD_SURFACE_PRESENT
  QAC:                     NOVEL_GOLD_MENTION   (expected higher, allowed)

  python -m src.gqa.main2000_diag
"""
from __future__ import annotations

import argparse
import glob
import json
import re

from src.common import read_jsonl, resolve
from src.gqa import common as C

_CAND_ENUM = [
    re.compile(r"['\"“”][^'\"“”]{1,18}['\"“”]\s*(?:또는|혹은|/|,|·)\s*['\"“”][^'\"“”]{1,18}['\"“”]"),
    re.compile(r"\S{1,12}인지\s+\S{1,12}인지"),
    re.compile(r"(?:또는|혹은).{0,24}중\s*(?:하나|어느|어떤|무엇)"),
]
_CMP_OUT = re.compile(r"더\s*(?:크|작|길|짧|높|낮|많|적|넓|좁|두꺼|두껍|얇|가까|무겁|가벼)[가-힣]*?"
                      r"(?:다|다\.|습니다|다고|음|ㅁ)(?![가-힣])")
_DIR_OUT = re.compile(r"(?:왼쪽|오른쪽|위쪽|아래쪽|앞쪽|뒤쪽|왼편|오른편)(?:에|으로)?\s*"
                      r"(?:있다|있습니다|위치한다|위치해\s*있다|놓여\s*있다|자리한다)(?![가-힣])")
_YESNO_OUT = re.compile(r"(?:답은|결론은|따라서)\s*['\"]?(?:예|아니오|아니요|그렇다|아니다|맞다|틀리다)")
_INTERROG = re.compile(r"(?:는지|ㄴ지|은지|인지|지를|지 여부|지 확인|지 판단|지 비교)")


def _cand_enum(t):
    return int(any(p.search(t or "") for p in _CAND_ENUM))


def _result_prop(t):
    for line in re.split(r"[\n。]", t or ""):
        s = line.strip()
        for pat in (_CMP_OUT, _DIR_OUT, _YESNO_OUT):
            m = pat.search(s)
            if m and not _INTERROG.search(s[max(0, m.start() - 4): m.end() + 4]):
                return 1
    return 0


def _steps(g):
    g = g or ""
    for pat in (r"(?m)^\s*\d+[\.\)]\s+", r"(?m)^\s*[-•*·]\s+"):
        n = len(re.findall(pat, g))
        if n:
            return n
    return len([ln for ln in g.splitlines() if ln.strip()])


def _gold_in_q(gn, q):
    return int(gn.replace(" ", "") in C.normalize_ko(q).replace(" ", "")) if gn else 0


def _merge(pattern, key):
    out = {}
    for fp in sorted(glob.glob(pattern)):
        for r in read_jsonl(fp):
            if r.get(key):
                out[r["id"]] = r
    return out


def _write_canonical(path, man, src, fields):
    with open(resolve(path), "w", encoding="utf-8") as f:
        for sid in man:
            if sid in src:
                rec = {"id": sid, "image_id": man[sid]["image_id"], "question": man[sid]["question"],
                       "family": man[sid]["family"], "group": man[sid]["group"]}
                for k in fields:
                    rec[k] = src[sid].get(k, "")
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/korean_gqa_main2000_manifest.jsonl")
    ap.add_argument("--dir", default="outputs/korean_gqa_main2000")
    args = ap.parse_args()
    man = {r["id"]: r for r in read_jsonl(resolve(args.manifest))}

    qac = _merge(str(resolve(f"{args.dir}/qac*.jsonl")), "qaware_caption")
    raw = _merge(str(resolve(f"{args.dir}/strong_guides_raw*.jsonl")), "strong_guide_raw")
    sup = _merge(str(resolve(f"{args.dir}/strong_guides*.jsonl")), "strong_guide")

    _write_canonical(f"{args.dir}/qac.jsonl", man, qac, ["qaware_caption"])
    _write_canonical(f"{args.dir}/strong_guides_raw.jsonl", man, raw, ["strong_guide_raw"])
    _write_canonical(f"{args.dir}/strong_guides.jsonl", man, sup, ["strong_guide", "strong_guide_raw"])

    rows = []
    for sid, m in man.items():
        gn = C.normalize_ko(m["answer"])
        giq = _gold_in_q(gn, m["question"])
        qt = (qac.get(sid, {}) or {}).get("qaware_caption", "")
        sgt = (sup.get(sid, {}) or {}).get("strong_guide", "")
        rgt = (raw.get(sid, {}) or {}).get("strong_guide_raw", "") or \
              (sup.get(sid, {}) or {}).get("strong_guide_raw", "")
        q_surf = int(C.lexical_answer_present(gn, qt))
        s_surf = int(C.lexical_answer_present(gn, sgt))
        rows.append({
            "id": sid, "family": m["family"], "group": m["group"],
            "gold": m["answer"], "question": m["question"], "gold_in_question": giq,
            "qac_present": bool(qt), "sg_present": bool(sgt), "sg_raw_present": bool(rgt),
            "qac_len": len(qt), "sg_len": len(sgt), "sg_raw_len": len(rgt),
            "sg_steps": _steps(sgt),
            "qac_gold_surface": q_surf, "qac_novel_gold_mention": int(q_surf == 1 and giq == 0),
            "sg_gold_surface": s_surf, "sg_novel_gold_mention": int(s_surf == 1 and giq == 0),
            "sg_result_proposition": _result_prop(sgt),
            "sg_answer_candidate_enumeration": _cand_enum(sgt),
        })

    out = resolve(f"{args.dir}/leakage_diag.jsonl")
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n = len(rows)
    nq = sum(1 for r in rows if r["qac_present"])
    ns = sum(1 for r in rows if r["sg_present"])
    def p(k, d):
        v = sum(r[k] for r in rows)
        return f"{v}/{d} ({v/max(d,1):.1%})"
    print(f"[m2k-diag] n={n}  qac_present={nq}  sg_present={ns}  sg_raw_present={sum(1 for r in rows if r['sg_raw_present'])}")
    print(f"  QAC  avg_len {sum(r['qac_len'] for r in rows)/max(nq,1):.0f} · "
          f"NOVEL_GOLD_MENTION {p('qac_novel_gold_mention', nq)} · gold_surface {p('qac_gold_surface', nq)}")
    print(f"  SG   avg_len {sum(r['sg_len'] for r in rows)/max(ns,1):.0f} · steps {sum(r['sg_steps'] for r in rows)/max(ns,1):.1f}")
    print(f"       NOVEL_GOLD_MENTION           {p('sg_novel_gold_mention', ns)}")
    print(f"       RESULT_PROPOSITION (auto)    {p('sg_result_proposition', ns)}")
    print(f"       ANSWER_CANDIDATE_ENUM (auto) {p('sg_answer_candidate_enumeration', ns)}")
    print(f"       gold_surface                 {p('sg_gold_surface', ns)}")
    print(f"  -> {out}")


if __name__ == "__main__":
    main()
