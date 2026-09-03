"""KVQA Ours-DirectAS-r2 — 500-sample PILOT analysis.

Reuses (never modifies):
  - src/gqa/kvqa_softscore.soft_accuracy / normalize_answer      (frozen evaluator)
  - src/gqa/main2000_diag._cand_enum / _result_prop              (frozen leakage detector)
  - src/gqa/kvqa_final_analyze.bmean / dpair / _tok_counter / BOOT_SEED   (frozen CI machinery)
  - src/gqa/kvqa_directas_sanity._state_conclusion               (auxiliary diagnostic, r2 sanity)

Scores DirectAS-r2 text/mm predictions, subsets the FROZEN scored_*.jsonl of the baseline
conditions to the same 500 ids, computes paired bootstrap 95% CIs, frozen leakage + auxiliary
diagnostics, representation length, and writes pilot_report.md + a console block.

  python -m src.gqa.kvqa_directas_pilot_analyze
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import statistics

import numpy as np

from src.common import read_jsonl, resolve
from src.gqa.kvqa_softscore import soft_accuracy, normalize_answer, selftest
from src.gqa.main2000_diag import _cand_enum, _result_prop
from src.gqa.kvqa_final_analyze import bmean, dpair, _tok_counter, BOOT_SEED
from src.gqa.kvqa_directas_sanity import _state_conclusion

FROZEN = "outputs/kvqa_final_hclt_vs_ours_as_2k"
PILOT = "outputs/kvqa_ours_directas/pilot500"

# heuristic: literal string / date / long-digit / price transcription inside evidence text
_OCR_LIT = re.compile(
    r"['\"“”][^'\"“”\n]{2,}['\"“”]"                       # quoted literal
    r"|\d{2,4}\s*[.\-/년]\s*\d{1,2}(?:\s*[.\-/월]\s*\d{1,2})?"  # date-ish
    r"|\d{4,}"                                             # 4+ digit run
    r"|\d[\d,]{2,}\s*원"                                   # price
)
_OCR_Q_KEY = re.compile(r"날짜|기한|유통|유효|제조일|가격|얼마|코드|시리얼|번호|며칠|일자|몇\s*년")


def _nospace(s):
    return normalize_answer(s or "").replace(" ", "")


def _flags(text, uniq_ns, q_ns):
    t = text or ""
    t_ns = _nospace(t)
    hits = [g for g in uniq_ns if g and g in t_ns]
    novel = [g for g in hits if g not in q_ns]
    return {
        "present": bool(t), "chars": len(t),
        "novel_gold_match_count": len(novel),
        "any_novel_gold_mention": int(len(novel) > 0),
        "novel_gold_surfaces": novel,
        "result_proposition": _result_prop(t),
        "candidate_enumeration": _cand_enum(t),
    }


def _merge_preds(d, name):
    m = {}
    for fp in sorted(glob.glob(f"{d}/predictions_{name}.jsonl") + glob.glob(f"{d}/predictions_{name}.shard*.jsonl")):
        for r in read_jsonl(fp):
            if "pred" in r and r.get("id"):
                m[r["id"]] = r["pred"]
    return m


def _load_text(d, glb, key):
    m = {}
    for fp in sorted(glob.glob(f"{d}/{glb}")):
        for r in read_jsonl(fp):
            if r.get(key) and r.get("id"):
                m[r["id"]] = r[key]
    return m


def _frozen_scored(cond, ids):
    """id -> soft_score from the FROZEN scored_<cond>.jsonl, restricted to `ids` (order = ids)."""
    by = {}
    for r in read_jsonl(f"{FROZEN}/scored_{cond}.jsonl"):
        by[r["id"]] = r.get("soft_score")
    return [by.get(i) for i in ids]


def _lenstats(texts, tok):
    chars = [len(t) for t in texts]
    toks = [tok(t) for t in texts] if tok else []
    out = {
        "n": len(texts),
        "mean_chars": round(statistics.fmean(chars), 1) if chars else None,
        "median_chars": statistics.median(chars) if chars else None,
        "mean_tokens": round(statistics.fmean(toks), 1) if toks else None,
        "median_tokens": round(float(np.median(toks)), 1) if toks else None,
        "p25_tokens": round(float(np.percentile(toks, 25)), 1) if toks else None,
        "p75_tokens": round(float(np.percentile(toks, 75)), 1) if toks else None,
    }
    return out


def _pct(x):
    return "n/a" if x != x else f"{100*x:.1f}%"


def _f4(x):
    return "n/a" if x is None or x != x else f"{x:.4f}"


def _dstr(dp):
    return f"{dp['delta']:+.4f}  95% CI [{dp['lo']:+.4f}, {dp['hi']:+.4f}]  " + \
           ("(excludes 0)" if dp["excl0"] else "(includes 0)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot-dir", default=PILOT)
    ap.add_argument("--manifest", default=f"{PILOT}/pilot500_manifest.jsonl")
    args = ap.parse_args()
    selftest()                                              # frozen-evaluator gate

    man = read_jsonl(resolve(args.manifest))
    ids = [r["id"] for r in man]
    m_by = {r["id"]: r for r in man}
    n = len(ids)
    d = str(resolve(args.pilot_dir))
    tok = _tok_counter()

    # ---- 1. score DirectAS-r2 text / mm ------------------------------------------------
    da_scores, da_pred = {}, {}
    gen_fail = {}
    for cond in ("directas_text", "directas_mm"):
        preds = _merge_preds(d, cond)
        gen_fail[cond] = [i for i in ids if i not in preds]
        sc, kk, npd = {}, {}, {}
        for i in ids:
            p = preds.get(i)
            if p is None:
                sc[i] = None
            else:
                s, k, np_ = soft_accuracy(p, m_by[i]["answers_10"])
                sc[i], kk[i], npd[i] = s, k, np_
        da_scores[cond] = sc
        da_pred[cond] = preds
        with open(f"{d}/scored_{cond}.jsonl", "w", encoding="utf-8") as f:
            for i in ids:
                p = preds.get(i)
                f.write(json.dumps({
                    "id": i, "answer_type": m_by[i]["answer_type"],
                    "raw_prediction": p,
                    "normalized_prediction": npd.get(i),
                    "answers_10_raw": m_by[i]["answers_10"],
                    "answers_10_normalized": [normalize_answer(a) for a in m_by[i]["answers_10"]],
                    "k": kk.get(i), "soft_score": sc[i],
                }, ensure_ascii=False) + "\n")

    # ---- 2. per-id soft-score arrays (aligned to `ids`) for every condition -----------
    S = {}
    for cond in ("iq", "generic_text", "generic_mm", "hclt_text", "hclt_mm",
                 "raw_text", "raw_mm", "as_text", "as_mm"):
        S[cond] = _frozen_scored(cond, ids)
    S["directas_text"] = [da_scores["directas_text"][i] for i in ids]
    S["directas_mm"] = [da_scores["directas_mm"][i] for i in ids]

    def acc(cond):
        mean, lo, hi, k = bmean(S[cond])
        return {"acc": mean, "lo": lo, "hi": hi, "n": k}

    ACC = {c: acc(c) for c in S}

    # image gains (paired: mm - text per sample)
    def gains(mm, txt):
        return [(a - b) if (a is not None and b is not None) else None for a, b in zip(S[mm], S[txt])]
    G = {
        "raw": gains("raw_mm", "raw_text"),
        "as": gains("as_mm", "as_text"),
        "directas": gains("directas_mm", "directas_text"),
    }
    gain_mean = {k: bmean(v)[0] for k, v in G.items()}

    # ---- 3. paired bootstrap deltas (seed = frozen BOOT_SEED) ------------------------
    D = {
        "directas_text_minus_raw_text": dpair(S["directas_text"], S["raw_text"]),
        "directas_mm_minus_raw_mm":     dpair(S["directas_mm"], S["raw_mm"]),
        "directas_text_minus_as_text":  dpair(S["directas_text"], S["as_text"]),
        "directas_mm_minus_as_mm":      dpair(S["directas_mm"], S["as_mm"]),
        "directas_gain_minus_raw_gain": dpair(G["directas"], G["raw"]),
        "directas_gain_minus_as_gain":  dpair(G["directas"], G["as"]),
    }

    # ---- 4. frozen leakage: DirectAS (fresh) + Raw/AS (subset frozen leakage_*.jsonl) --
    da_txt = _load_text(d, "ours_directas_evidence*.jsonl", "directas_evidence")
    da_dec = _load_text(d, "ours_protected_decompositions*.jsonl", "decomposition")
    fr_leak = {name: {r["id"]: r for r in read_jsonl(f"{FROZEN}/leakage_{name}.jsonl")}
               for name in ("raw", "as")}

    LK = {"raw": {"ng": [], "rp": [], "ce": []}, "as": {"ng": [], "rp": [], "ce": []},
          "directas": {"ng": [], "rp": [], "ce": []}}
    aux_state = {"raw": 0, "as": 0, "directas": 0}
    da_ng_ids, da_rp_ids, da_state_ids, ocr_ids = [], [], [], []
    raw_txt = _load_text(FROZEN, "ours_raw_evidence*.jsonl", "raw_evidence")
    as_txt = _load_text(FROZEN, "ours_suppressed_evidence*.jsonl", "suppressed_evidence")

    for i in ids:
        mrow = m_by[i]
        uniq = sorted({_nospace(a) for a in mrow["answers_10"]} - {""})
        q_ns = _nospace(mrow["question"])
        for name, txtmap in (("raw", raw_txt), ("as", as_txt), ("directas", da_txt)):
            if name in fr_leak:                        # use the FROZEN per-sample flags verbatim
                fl = fr_leak[name].get(i, {})
                ng = int(fl.get("any_novel_gold_mention") or 0)
                rp = int(fl.get("result_proposition") or 0)
                ce = int(fl.get("candidate_enumeration") or 0)
            else:
                f = _flags(txtmap.get(i, ""), uniq, q_ns)
                ng, rp, ce = f["any_novel_gold_mention"], f["result_proposition"], f["candidate_enumeration"]
            LK[name]["ng"].append(ng); LK[name]["rp"].append(rp); LK[name]["ce"].append(ce)
            sc = _state_conclusion(txtmap.get(i, ""))
            aux_state[name] += sc
            if name == "directas":
                if ng:
                    da_ng_ids.append(i)
                if rp:
                    da_rp_ids.append(i)
                if sc:
                    da_state_ids.append(i)
        # date / OCR transcription heuristic (DirectAS only)
        dec = da_dec.get(i) or {}
        at = dec.get("answer_type") if isinstance(dec, dict) else None
        if at == "text" or _OCR_Q_KEY.search(mrow["question"] or ""):
            if _OCR_LIT.search(da_txt.get(i, "")):
                ocr_ids.append(i)

    leak_rate = {name: {k: (sum(v) / n) for k, v in LK[name].items()} for name in LK}

    # ---- 5. representation length -----------------------------------------------------
    LEN = {
        "raw": _lenstats([raw_txt[i] for i in ids if i in raw_txt], tok),
        "as": _lenstats([as_txt[i] for i in ids if i in as_txt], tok),
        "directas": _lenstats([da_txt[i] for i in ids if i in da_txt], tok),
    }

    # ---- 6. representative cue-preservation examples + failure cases -----------------
    cue_words = ("왼쪽", "오른쪽", "상단", "하단", "중앙", "가장자리", "위치", "영역", "뒤쪽", "앞쪽", "옆")
    rep_ok, rep_fail = [], []
    for i in ids:
        dt, ast = da_txt.get(i, ""), as_txt.get(i, "")
        fl = _flags(dt, sorted({_nospace(a) for a in m_by[i]["answers_10"]} - {""}), _nospace(m_by[i]["question"]))
        cue = sum(w in dt for w in cue_words)
        if (fl["any_novel_gold_mention"] == 0 and not fl["result_proposition"]
                and not _state_conclusion(dt) and len(dt) >= max(60, len(ast) + 25) and cue >= 2):
            rep_ok.append((i, len(ast), len(dt), cue))
        if (fl["any_novel_gold_mention"] or fl["result_proposition"] or _state_conclusion(dt)
                or i in ocr_ids or len(dt) < 25):
            rep_fail.append(i)
    rep_ok.sort(key=lambda x: (x[3], x[2] - x[1]), reverse=True)
    rep_ok_ids = [i for i, *_ in rep_ok[:5]]
    rep_fail_ids = rep_fail[:5]

    # ================================ report =========================================
    RLAB = {"iq": "I+Q", "generic": "Generic", "hclt": "HCLT-style",
            "raw": "Ours-Raw", "as": "Ours-AS", "directas": "DirectAS-r2"}
    L = []
    P = L.append
    P("# KVQA Ours-DirectAS-r2 — 500-sample pilot\n")
    P(f"N = {n}   subset = first {n} of `{FROZEN}/manifest.jsonl` (order preserved, no resampling)\n")
    P(f"paired bootstrap: 10 000 resamples, seed {BOOT_SEED} (frozen `boot_diff`)\n")
    P(f"DirectAS generation failures: text {len(gen_fail['directas_text'])}/{n}, "
      f"mm {len(gen_fail['directas_mm'])}/{n}\n")

    P("\n## 9. Output table\n")
    P("| Method | Text Acc | MM Acc | Image Gain | NovelGold | ResultProp | CandEnum | StateConcl(aux) | Mean Tok | Med Tok |")
    P("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    P(f"| I+Q | — | {_f4(ACC['iq']['acc'])} | — | — | — | — | — | — | — |")
    for rk in ("generic", "hclt"):
        t, mm = ACC[f"{rk}_text"], ACC[f"{rk}_mm"]
        P(f"| {RLAB[rk]} | {_f4(t['acc'])} | {_f4(mm['acc'])} | {mm['acc']-t['acc']:+.4f} | "
          f"— | — | — | — | — | — |")
    for rk in ("raw", "as", "directas"):
        t, mm = ACC[f"{rk}_text"], ACC[f"{rk}_mm"]
        ln = LEN[rk]
        P(f"| {RLAB[rk]} | {_f4(t['acc'])} | {_f4(mm['acc'])} | {mm['acc']-t['acc']:+.4f} | "
          f"{_pct(leak_rate[rk]['ng'])} | {_pct(leak_rate[rk]['rp'])} | {_pct(leak_rate[rk]['ce'])} | "
          f"{aux_state[rk]}/{n} | {ln['mean_tokens']} | {ln['median_tokens']} |")

    P("\n### single-condition 95% CI (bootstrap of the mean)\n")
    for c in ("iq", "raw_text", "raw_mm", "as_text", "as_mm", "directas_text", "directas_mm"):
        a = ACC[c]
        P(f"- {c:14} acc {_f4(a['acc'])}  95% CI [{_f4(a['lo'])}, {_f4(a['hi'])}]  (n={a['n']})")

    P("\n## 10. KEY DELTAS\n```")
    P("Raw -> DirectAS")
    P(f"  Text : {_dstr(D['directas_text_minus_raw_text'])}")
    P(f"  MM   : {_dstr(D['directas_mm_minus_raw_mm'])}")
    P(f"  NovelGold : DirectAS {_pct(leak_rate['directas']['ng'])}  -  Raw {_pct(leak_rate['raw']['ng'])}  "
      f"= {100*(leak_rate['directas']['ng']-leak_rate['raw']['ng']):+.1f} pp")
    P("")
    P("AS -> DirectAS")
    P(f"  Text : {_dstr(D['directas_text_minus_as_text'])}")
    P(f"  MM   : {_dstr(D['directas_mm_minus_as_mm'])}")
    P(f"  NovelGold : DirectAS {_pct(leak_rate['directas']['ng'])}  -  AS {_pct(leak_rate['as']['ng'])}  "
      f"= {100*(leak_rate['directas']['ng']-leak_rate['as']['ng']):+.1f} pp")
    P("")
    P("IMAGE DEPENDENCE")
    P(f"  Raw image gain      = {gain_mean['raw']:+.4f}")
    P(f"  AS image gain       = {gain_mean['as']:+.4f}")
    P(f"  DirectAS image gain = {gain_mean['directas']:+.4f}")
    P(f"  DirectAS gain - Raw gain = {_dstr(D['directas_gain_minus_raw_gain'])}")
    P(f"  DirectAS gain - AS gain  = {_dstr(D['directas_gain_minus_as_gain'])}")
    P("```")

    P("\n## 5. Frozen leakage vs auxiliary diagnostic\n")
    P("**Frozen Leakage Metrics** (main2000_diag defs; Raw/AS use the frozen per-sample "
      "`leakage_*.jsonl`, DirectAS computed with the same defs):\n")
    P("| repr | NovelGold | ResultProp | CandEnum |")
    P("|---|---:|---:|---:|")
    for rk in ("raw", "as", "directas"):
        P(f"| {RLAB[rk]} | {_pct(leak_rate[rk]['ng'])} | {_pct(leak_rate[rk]['rp'])} | {_pct(leak_rate[rk]['ce'])} |")
    P("\n**Auxiliary Diagnostic** (NOT a frozen metric; does not replace the above):\n")
    P(f"- State / Yes-No conclusion flag (KO regex, r2-sanity): "
      f"Raw {aux_state['raw']}/{n}, AS {aux_state['as']}/{n}, DirectAS {aux_state['directas']}/{n}")
    P(f"- Date/code/OCR literal-transcription heuristic (DirectAS, text-type or date-keyword Q): "
      f"{len(ocr_ids)}/{n}  ids={ocr_ids}")

    P("\n## 7. Representation length (Qwen tokenizer)\n")
    P("| repr | mean char | median char | mean tok | median tok | p25 tok | p75 tok |")
    P("|---|---:|---:|---:|---:|---:|---:|")
    for rk in ("raw", "as", "directas"):
        ln = LEN[rk]
        P(f"| {RLAB[rk]} | {ln['mean_chars']} | {ln['median_chars']} | {ln['mean_tokens']} | "
          f"{ln['median_tokens']} | {ln['p25_tokens']} | {ln['p75_tokens']} |")
    if LEN["directas"]["mean_tokens"] and LEN["as"]["mean_tokens"]:
        P(f"\nDirectAS mean tokens = {LEN['directas']['mean_tokens']}  "
          f"vs Ours-AS {LEN['as']['mean_tokens']} (x{LEN['directas']['mean_tokens']/LEN['as']['mean_tokens']:.2f})  "
          f"vs Ours-Raw {LEN['raw']['mean_tokens']} (x{LEN['directas']['mean_tokens']/LEN['raw']['mean_tokens']:.2f})")

    P("\n## 8. DirectAS preserves cue better than AS — 5 examples\n")
    for i in rep_ok_ids:
        P(f"### {i}  [{m_by[i]['answer_type']}]  Q: {m_by[i]['question']}")
        P(f"- gold(uniq-ns): {sorted({_nospace(a) for a in m_by[i]['answers_10']} - {''})}")
        P(f"- Ours-AS  ({len(as_txt.get(i,''))} ch): {as_txt.get(i,'')}")
        P(f"- DirectAS ({len(da_txt.get(i,''))} ch): {da_txt.get(i,'')}\n")

    P("\n## 9. DirectAS failure cases — 5\n")
    for i in rep_fail_ids:
        fl = _flags(da_txt.get(i, ""), sorted({_nospace(a) for a in m_by[i]['answers_10']} - {''}), _nospace(m_by[i]['question']))
        P(f"### {i}  [{m_by[i]['answer_type']}]  Q: {m_by[i]['question']}")
        P(f"- gold(uniq-ns): {sorted({_nospace(a) for a in m_by[i]['answers_10']} - {''})}")
        P(f"- flags: NovelGold={fl['any_novel_gold_mention']}({fl['novel_gold_surfaces']}) "
          f"ResultProp={fl['result_proposition']} StateConcl={_state_conclusion(da_txt.get(i,''))} "
          f"OCRlit={'Y' if i in ocr_ids else '·'}")
        P(f"- DirectAS: {da_txt.get(i,'')}\n")

    # ---- 11. auto verdict ----------------------------------------------------------
    def incl0(k):
        return not D[k]["excl0"]
    P("\n## 11. AUTO VERDICT\n")
    ng_raw_pp = 100 * (leak_rate["raw"]["ng"] - leak_rate["directas"]["ng"])
    ng_as_pp = 100 * (leak_rate["as"]["ng"] - leak_rate["directas"]["ng"])
    P(f"**A. ANSWER EXPOSURE** — DirectAS-r2 NovelGold {_pct(leak_rate['directas']['ng'])} "
      f"vs Raw {_pct(leak_rate['raw']['ng'])} ({ng_raw_pp:+.1f} pp) and AS {_pct(leak_rate['as']['ng'])} "
      f"({ng_as_pp:+.1f} pp). Auxiliary state-conclusion flag "
      f"{aux_state['directas']}/{n} (Raw {aux_state['raw']}/{n}, AS {aux_state['as']}/{n}); "
      f"OCR-literal heuristic {len(ocr_ids)}/{n}.")

    dt = D["directas_text_minus_raw_text"]
    P(f"\n**B. TEXT-ONLY SHORTCUT** — DirectAS_Text - Raw_Text = {dt['delta']:+.4f} "
      f"[{dt['lo']:+.4f}, {dt['hi']:+.4f}]. "
      + ("A drop is expected by design and is not counted as failure."
         if dt["delta"] < 0 else "No drop observed."))

    dm_raw = D["directas_mm_minus_raw_mm"]
    dm_as = D["directas_mm_minus_as_mm"]
    def mm_phrase(dp, base):
        if not dp["excl0"]:
            return f"comparable to {base}-MM (95% CI includes 0; no statistically significant difference)"
        return (f"significantly BELOW {base}-MM — DirectAS reduces multimodal utility"
                if dp["delta"] < 0 else f"significantly ABOVE {base}-MM")
    P(f"\n**C. MULTIMODAL UTILITY** — DirectAS_MM {_f4(ACC['directas_mm']['acc'])} "
      f"[{_f4(ACC['directas_mm']['lo'])}, {_f4(ACC['directas_mm']['hi'])}]. "
      f"vs Raw_MM: {mm_phrase(dm_raw,'Raw')} ({dm_raw['delta']:+.4f} [{dm_raw['lo']:+.4f}, {dm_raw['hi']:+.4f}]). "
      f"vs AS_MM: {mm_phrase(dm_as,'AS')} ({dm_as['delta']:+.4f} [{dm_as['lo']:+.4f}, {dm_as['hi']:+.4f}]).")

    gr = D["directas_gain_minus_raw_gain"]
    P(f"\n**D. IMAGE DEPENDENCE** — image gain Raw {gain_mean['raw']:+.4f} / AS {gain_mean['as']:+.4f} / "
      f"DirectAS {gain_mean['directas']:+.4f}. DirectAS gain - Raw gain = {gr['delta']:+.4f} "
      f"[{gr['lo']:+.4f}, {gr['hi']:+.4f}] "
      + ("(excludes 0): greater behavioral dependence on the original image than Ours-Raw."
         if gr["excl0"] and gr["delta"] > 0
         else "(includes 0): no significant change in image dependence vs Ours-Raw." if not gr["excl0"]
         else "(excludes 0, negative)."))

    over = (dt["delta"] < 0 and dm_raw["excl0"] and dm_raw["delta"] < 0
            and dm_as["excl0"] and dm_as["delta"] < 0)
    intended = (dt["delta"] < 0 and not (dm_raw["excl0"] and dm_raw["delta"] < 0)
                and not (dm_as["excl0"] and dm_as["delta"] < 0))
    P("\n**E. OVER-SUPPRESSION CHECK** — "
      + ("PATTERN MATCHES OVER-SUPPRESSION: DirectAS_Text << Raw_Text AND DirectAS_MM significantly "
         "below both Raw_MM and AS_MM." if over
         else "Consistent with the intended pattern: DirectAS_Text below Raw_Text, but DirectAS_MM "
              "comparable to Raw/AS-MM (no significant MM degradation)." if intended
              else "Mixed: see C — DirectAS_MM differs significantly from at least one of Raw/AS-MM; "
                   "not a clean over-suppression pattern either."))

    P(f"\n**Q: Does DirectAS-r2 reduce answer exposure without materially degrading multimodal QA utility?**\n")
    ans = ("YES — answer exposure is reduced (NovelGold "
           f"{ng_raw_pp:+.1f} pp vs Raw, {ng_as_pp:+.1f} pp vs AS) and DirectAS_MM shows no statistically "
           "significant degradation vs Raw_MM / AS_MM on this 500 pilot."
           if (ng_raw_pp >= 0 and not (dm_raw["excl0"] and dm_raw["delta"] < 0)
               and not (dm_as["excl0"] and dm_as["delta"] < 0))
           else "NO / QUALIFIED — see C (multimodal utility) and A (exposure); the pilot does not "
                "cleanly support the claim.")
    P(ans)

    P("\n## full-2K recommendation\n")
    P("(analyst note — user decides) "
      + ("Proceed to full 2K: exposure down, MM utility statistically comparable."
         if ("YES" in ans) else
         "Do NOT auto-proceed; review C/E before committing to full 2K."))

    txt = "\n".join(L) + "\n"
    with open(f"{d}/pilot_report.md", "w", encoding="utf-8") as f:
        f.write(txt)
    print(txt)
    print(f"[directas-pilot] -> {d}/pilot_report.md")
    print(f"[directas-pilot] -> {d}/scored_directas_text.jsonl , scored_directas_mm.jsonl")


if __name__ == "__main__":
    main()
