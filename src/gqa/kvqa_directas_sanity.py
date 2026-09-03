"""KVQA  Ours-DirectAS  —  30-sample SANITY (fixed sanity30 ids).  CPU only, NO model.

Assembles, for the 30 frozen sanity ids:
  - frozen (read-only):  old 6-key decomposition, Ours-Raw evidence, Ours-AS suppressed evidence
  - new:                 9-key PROTECTED decomposition, Ours-DirectAS evidence
and computes the diagnostics requested for the sanity check (frozen detector defs):
  NovelGold (10-gold) · Result Proposition · Candidate Enumeration · representation length
  · Ours-AS <-> DirectAS exact / near duplicate ratio
  · malformed protected-decomposition JSON ratio
  · protected_answer_value directly contains a real answer value?
  · does any protected-decomposition field carry a NOVEL gold surface?

Writes  <dir>/sanity30_directas.jsonl  and  <dir>/sanity30_directas.html  and prints the tables.
Gold (10 human answers) is used for DIAGNOSTICS / display ONLY — never an input to generation.

  python -m src.gqa.kvqa_directas_sanity
"""
from __future__ import annotations

import argparse
import base64
import difflib
import glob
import io
import json
import re

from src.common import read_jsonl, resolve
from src.gqa.main2000_diag import _cand_enum, _result_prop      # FROZEN leakage detector (unchanged)
from src.gqa.kvqa_softscore import normalize_answer             # KVQA official-derived normalization

# AUXILIARY sanity-only signal (NOT the frozen detector, NOT used for scoring): KVQA yes_no gold is
# the English "Yes"/"No", so NovelGold and the Korean-form _result_prop regexes cannot see a
# state/existence conclusion narrated in Korean. Flag DirectAS text that still asserts the state.
_STATE_CONCL = re.compile(
    r"(우거져|빽빽|불투명|투명하(?:지|게|다)|밀도가?\s*높|가득|비어\s*있|"
    r"켜져\s*있|꺼져\s*있|열려\s*있|닫혀\s*있|"
    r"존재(?:하지\s*않|한다|합니다|하는 것)|없음을|없다는 것|없는 것으로|있음을|있는 것으로)"
    r"[^.\n]{0,24}"
    r"(확인할 수 있다|확인됩니다|확인된다|보인다|보입니다|판단할 수 있다|것으로 보|임을|음을|상태(?:이다|입니다|임))"
)


def _state_conclusion(text: str) -> int:
    return int(bool(_STATE_CONCL.search(text or "")))


# ------------------------------------------------------------------ diagnostics (frozen defs)
def _nospace(s):
    return normalize_answer(s or "").replace(" ", "")


def _flags(text, unique_gold_ns, q_ns):
    """Identical to src/gqa/kvqa_final_diag._flags (kept inline so sanity has no torch import)."""
    t = text or ""
    t_ns = _nospace(t)
    hits = [g for g in unique_gold_ns if g and g in t_ns]
    novel = [g for g in hits if g not in q_ns]
    return {
        "present": bool(t), "len": len(t),
        "gold_surface_count": len(hits),
        "novel_gold_match_count": len(novel),
        "any_novel_gold_mention": int(len(novel) > 0),
        "novel_gold_surfaces": novel,
        "result_proposition": _result_prop(t),
        "candidate_enumeration": _cand_enum(t),
    }


def _decomp_text_fields(dec: dict):
    """Every human-readable string inside a decomposition dict (6-key or 9-key)."""
    if not isinstance(dec, dict):
        return []
    vals = []
    for k, v in dec.items():
        if k == "answer_type":
            continue
        if isinstance(v, str) and v.strip():
            vals.append(v.strip())
        elif isinstance(v, list):
            vals.extend(str(x).strip() for x in v if str(x).strip())
    return vals


# ------------------------------------------------------------------ html helpers
def img_uri(path, maxw=430, q=68):
    try:
        from PIL import Image
        im = Image.open(path).convert("RGB")
        if im.width > maxw:
            im = im.resize((maxw, int(im.height * maxw / im.width)))
        b = io.BytesIO()
        im.save(b, "JPEG", quality=q)
        return "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode()
    except Exception:
        return ""


def esc(s):
    return str(s if s is not None else "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


CSS = ("<style>body{font-family:system-ui,Arial;margin:16px;background:#fafafa}"
       ".c{background:#fff;border:1px solid #ddd;border-radius:8px;padding:12px;margin:12px 0;max-width:1060px}"
       "img{max-width:430px;border:1px solid #ccc;border-radius:6px}"
       "pre{white-space:pre-wrap;margin:3px 0;font-size:12.5px}"
       ".q{background:#f6f6ff;border-left:3px solid #55a;padding:5px 8px}"
       ".gold{background:#f3f3f3;border-left:3px solid #999;padding:5px 8px;font-size:12px}"
       ".dold{background:#f3eefb;border-left:3px solid #85a;padding:5px 8px}"
       ".dnew{background:#efe8fb;border-left:3px solid #63a;padding:5px 8px}"
       ".raw{background:#f0f0f0;border-left:3px solid #999;padding:5px 8px}"
       ".as{background:#fff4e8;border-left:3px solid #e90;padding:5px 8px}"
       ".da{background:#e8f6ee;border-left:3px solid #2a8;padding:5px 8px}"
       ".fl{background:#fff0f0;border-left:3px solid #c55;padding:5px 8px;font-size:12px}"
       "table{border-collapse:collapse;font-size:12.5px;margin-top:6px}"
       "td,th{border:1px solid #e2e5ea;padding:3px 8px;text-align:right}"
       "td.l,th.l{text-align:left}</style>")


def _load(d, pat, tk, idk="id"):
    """Map id -> the value at key `tk` (str for evidence files, dict for decomposition files)."""
    o = {}
    for fp in sorted(glob.glob(f"{d}/{pat}")):
        for r in read_jsonl(fp):
            if r.get(tk) is not None and r.get(idk):
                o[r[idk]] = r[tk]
    return o


def flagstr(lk):
    return (f"NovelGold={'Y' if lk['any_novel_gold_mention'] else '·'}"
            f"({lk['novel_gold_match_count']})  "
            f"ResultProp={'Y' if lk['result_proposition'] else '·'}  "
            f"CandEnum={'Y' if lk['candidate_enumeration'] else '·'}  "
            f"len={lk['len']}"
            + (f"  surfaces={lk['novel_gold_surfaces']}" if lk['novel_gold_surfaces'] else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="outputs/kvqa_ours_directas")
    ap.add_argument("--frozen-dir", default="outputs/kvqa_final_hclt_vs_ours_as_2k")
    ap.add_argument("--manifest",
                    default="outputs/kvqa_final_hclt_vs_ours_as_2k/_sanity30_archive/_sanity30_manifest.jsonl")
    args = ap.parse_args()
    d = str(resolve(args.dir))
    fz = str(resolve(args.frozen_dir))

    man = read_jsonl(resolve(args.manifest))
    ids = [r["id"] for r in man]
    man_by = {r["id"]: r for r in man}

    old_dec = _load(fz, "ours_decompositions.jsonl", "decomposition")
    raw_ev = _load(fz, "ours_raw_evidence.jsonl", "raw_evidence")
    as_ev = _load(fz, "ours_suppressed_evidence.jsonl", "suppressed_evidence")
    new_dec = _load(d, "ours_protected_decompositions*.jsonl", "decomposition")
    da_ev = _load(d, "ours_directas_evidence*.jsonl", "directas_evidence")
    pstatus_by = {}
    for fp in sorted(glob.glob(f"{d}/ours_protected_decompositions*.jsonl")):
        for r in read_jsonl(fp):
            if r.get("id") and r.get("decomposition") is not None:
                pstatus_by[r["id"]] = r.get("parse_status")

    missing = [i for i in ids if i not in new_dec or i not in da_ev]
    if missing:
        raise SystemExit(f"[directas-sanity] {len(missing)} sanity ids lack new outputs — run generation first: {missing[:5]}")

    rows, cards = [], []
    agg = {k: {"present": 0, "len": 0, "novel": 0, "rp": 0, "ce": 0, "mc": 0} for k in ("raw", "as", "directas")}
    n = len(ids)
    malformed = 0
    pav_has_answer = 0
    decomp_novel_gold = 0
    as_da_exact = 0
    as_da_near = 0
    da_state_concl_n = 0

    for i in ids:
        m = man_by[i]
        uniq = sorted({_nospace(a) for a in m["answers_10"]} - {""})
        q_ns = _nospace(m["question"])

        f_raw = _flags(raw_ev.get(i, ""), uniq, q_ns)
        f_as = _flags(as_ev.get(i, ""), uniq, q_ns)
        f_da = _flags(da_ev.get(i, ""), uniq, q_ns)
        for key, fl in (("raw", f_raw), ("as", f_as), ("directas", f_da)):
            a = agg[key]
            a["present"] += int(fl["present"]); a["len"] += fl["len"]
            a["novel"] += fl["any_novel_gold_mention"]; a["rp"] += fl["result_proposition"]
            a["ce"] += fl["candidate_enumeration"]; a["mc"] += fl["novel_gold_match_count"]

        pstatus = pstatus_by.get(i)
        if pstatus not in ("ok", "keys_filled"):
            malformed += 1

        ndec = new_dec.get(i, {})
        pav = ndec.get("protected_answer_value")
        pav_ns = _nospace(pav or "")
        # a real leak only if the surface is NOT already a question token (multiple-choice
        # questions legitimately name their options, e.g. "앞면이야 뒷면이야?" -> pav "...앞면 여부...")
        pav_hit = [g for g in uniq if len(g) >= 2 and g in pav_ns and g not in q_ns]
        if pav_hit:
            pav_has_answer += 1

        dec_blob_ns = _nospace(" ".join(_decomp_text_fields(ndec)))
        dec_novel = [g for g in uniq if len(g) >= 2 and g in dec_blob_ns and g not in q_ns]
        if dec_novel:
            decomp_novel_gold += 1

        a_txt, da_txt = as_ev.get(i, ""), da_ev.get(i, "")
        ratio = difflib.SequenceMatcher(None, a_txt, da_txt).ratio() if (a_txt and da_txt) else 0.0
        if a_txt and da_txt and normalize_answer(a_txt) == normalize_answer(da_txt):
            as_da_exact += 1
        if ratio >= 0.85:
            as_da_near += 1

        collapsed = f_da["present"] and f_da["len"] < 25
        da_state_concl = _state_conclusion(da_txt)
        da_state_concl_n += da_state_concl
        success = (f_da["present"] and f_da["novel_gold_match_count"] == 0
                   and not f_da["result_proposition"] and not f_da["candidate_enumeration"]
                   and not pav_hit and not dec_novel and not da_state_concl and f_da["len"] >= 40)
        failure = (not f_da["present"] or f_da["novel_gold_match_count"] > 0
                   or f_da["result_proposition"] or collapsed or bool(pav_hit)
                   or bool(dec_novel) or da_state_concl)

        rows.append({
            "id": i, "answer_type": m["answer_type"], "question": m["question"],
            "unique_gold": uniq,
            "old_decomposition": old_dec.get(i), "protected_decomposition": ndec,
            "protected_decomposition_parse_status": pstatus,
            "raw_evidence": raw_ev.get(i, ""), "as_evidence": a_txt, "directas_evidence": da_txt,
            "flags_raw": f_raw, "flags_as": f_as, "flags_directas": f_da,
            "as_vs_directas_ratio": round(ratio, 3),
            "protected_answer_value": pav, "pav_contains_answer_value": pav_hit,
            "decomposition_novel_gold_surfaces": dec_novel,
            "directas_states_conclusion_aux": da_state_concl,
            "tag": "SUCCESS" if success and not failure else ("FAILURE" if failure else "ok"),
            "collapsed_generic": collapsed,
        })

        img = img_uri(m["image_path"])
        cards.append(
            "<div class=c>"
            f"<img src='{img}'><br>"
            f"<pre class=q><b>[{esc(m['answer_type'])}] {esc(i)}</b>\nQ: {esc(m['question'])}</pre>"
            f"<pre class=gold>GOLD (review only, 10): {esc(m['answers_10'])}\nunique-normalized: {esc(uniq)}</pre>"
            f"<pre class=dold>OLD decomposition (frozen 6-key):\n{esc(json.dumps(old_dec.get(i), ensure_ascii=False, indent=2))}</pre>"
            f"<pre class=dnew>NEW protected decomposition (9-key, status={esc(pstatus)}):\n{esc(json.dumps(ndec, ensure_ascii=False, indent=2))}</pre>"
            f"<pre class=raw>Ours-Raw (frozen):\n{esc(raw_ev.get(i, ''))}\n  » {flagstr(f_raw)}</pre>"
            f"<pre class=as>Ours-AS (frozen):\n{esc(a_txt)}\n  » {flagstr(f_as)}</pre>"
            f"<pre class=da>Ours-DirectAS (NEW):\n{esc(da_txt)}\n  » {flagstr(f_da)}</pre>"
            f"<pre class=fl>tag={rows[-1]['tag']}  AS↔DirectAS ratio={rows[-1]['as_vs_directas_ratio']}"
            f"  pav='{esc(pav)}'  pav_contains_answer={pav_hit}  decomp_novel_gold={dec_novel}</pre>"
            "</div>"
        )

    def r3(x):
        return round(x, 3)

    summary = {
        "n": n,
        "sanity_manifest": args.manifest,
        "aggregate": {
            k: {
                "present": f"{v['present']}/{n}",
                "avg_len": r3(v["len"] / max(v["present"], 1)),
                "novel_gold_pct": r3(v["novel"] / n),
                "result_prop_pct": r3(v["rp"] / n),
                "cand_enum_pct": r3(v["ce"] / n),
                "avg_novel_gold_match_ct": r3(v["mc"] / n),
            } for k, v in agg.items()
        },
        "malformed_protected_decomposition_ratio": f"{malformed}/{n}",
        "protected_answer_value_contains_real_answer_ratio": f"{pav_has_answer}/{n}",
        "protected_decomposition_any_field_novel_gold_ratio": f"{decomp_novel_gold}/{n}",
        "as_vs_directas_exact_dup_ratio": f"{as_da_exact}/{n}",
        "as_vs_directas_near_dup_ratio_ge_0.85": f"{as_da_near}/{n}",
        "directas_states_conclusion_aux_ratio": f"{da_state_concl_n}/{n}",
        "success_ids": [r["id"] for r in rows if r["tag"] == "SUCCESS"],
        "failure_ids": [r["id"] for r in rows if r["tag"] == "FAILURE"],
    }

    with open(f"{d}/sanity30_directas.jsonl", "w", encoding="utf-8") as f:
        f.write(json.dumps({"__summary__": summary}, ensure_ascii=False) + "\n")
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    head = ("<h2>KVQA Ours-DirectAS — 30-sample sanity</h2>"
            f"<pre class=c>{esc(json.dumps(summary, ensure_ascii=False, indent=2))}</pre>")
    with open(f"{d}/sanity30_directas.html", "w", encoding="utf-8") as f:
        f.write("<!doctype html><meta charset=utf-8>" + CSS + head + "".join(cards))

    # ---- stdout tables -------------------------------------------------------------
    print(f"\n[directas-sanity] n={n}   (gold used for diagnostics / display only)\n")
    print(f"{'repr':10} {'present':>8} {'avg_len':>8} {'NovelGold':>10} {'ResultProp':>11} {'CandEnum':>9} {'avgMatchCt':>11}")
    for k in ("raw", "as", "directas"):
        v = summary["aggregate"][k]
        print(f"{k:10} {v['present']:>8} {v['avg_len']:>8.1f} {v['novel_gold_pct']:>9.1%} "
              f"{v['result_prop_pct']:>10.1%} {v['cand_enum_pct']:>8.1%} {v['avg_novel_gold_match_ct']:>11.3f}")
    print()
    for k in ("malformed_protected_decomposition_ratio",
              "protected_answer_value_contains_real_answer_ratio",
              "protected_decomposition_any_field_novel_gold_ratio",
              "as_vs_directas_exact_dup_ratio",
              "as_vs_directas_near_dup_ratio_ge_0.85",
              "directas_states_conclusion_aux_ratio"):
        print(f"  {k:52} {summary[k]}")
    print(f"\n  SUCCESS ids ({len(summary['success_ids'])}): {summary['success_ids']}")
    print(f"  FAILURE ids ({len(summary['failure_ids'])}): {summary['failure_ids']}")
    print(f"\n[directas-sanity] -> {d}/sanity30_directas.jsonl")
    print(f"[directas-sanity] -> {d}/sanity30_directas.html")


if __name__ == "__main__":
    main()
