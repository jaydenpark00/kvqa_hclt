"""KVQA FINAL 2K — representation comparison: Generic / HCLT-style / Ours-AS / Ours-DirectAS-r2.

Four axes (Text-only is EXCLUDED from the main comparison):
  1. Multimodal QA utility        (frozen common MM solver, KVQA soft VQA acc)
  2. Lexical Answer Exposure       (= the frozen NOVEL_GOLD_MENTION metric, presentation label only)
  3. Representation / token efficiency  (Qwen tokenizer; Token Reduction vs Generic)
  4. Multimodal accuracy by question type  (manifest.answer_type = OFFICIAL KVQA label)

Nothing frozen is regenerated or redefined. Generic / HCLT / Ours-AS reuse the frozen
outputs/kvqa_final_hclt_vs_ours_as_2k/ scored + leakage + representation files. DirectAS-r2
reuses the frozen r2 config; its MM predictions come from outputs/kvqa_ours_directas/full2k/.

  python -m src.gqa.kvqa_directas_final_compare
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import statistics

import numpy as np

from src.common import read_jsonl, resolve
from src.gqa.kvqa_softscore import soft_accuracy, normalize_answer, selftest
from src.gqa.main2000_diag import _cand_enum, _result_prop
from src.gqa.kvqa_final_analyze import bmean, dpair, _tok_counter, BOOT_SEED
from src.gqa.kvqa_directas_sanity import _state_conclusion
from src.gqa.kvqa_directas_pilot_analyze import _OCR_LIT, _OCR_Q_KEY, _nospace

FROZEN = "outputs/kvqa_final_hclt_vs_ours_as_2k"
F2 = "outputs/kvqa_ours_directas/full2k"

METHODS = ["generic", "hclt", "as", "directas"]
LABEL = {"generic": "Generic Caption", "hclt": "HCLT-style", "as": "Ours-AS",
         "directas": "Ours-DirectAS-r2", "iq": "I+Q (reference)"}
REP = {  # method -> (dir, glob, text_key)
    "generic":  (FROZEN, "generic_captions.jsonl",        "generic_caption"),
    "hclt":     (FROZEN, "hclt_high_density.jsonl",       "hclt_high_density_caption"),
    "as":       (FROZEN, "ours_suppressed_evidence.jsonl", "suppressed_evidence"),
    "directas": (F2,     "ours_directas_evidence*.jsonl",  "directas_evidence"),
}
MM_SCORED = {"generic": f"{FROZEN}/scored_generic_mm.jsonl",
             "hclt":    f"{FROZEN}/scored_hclt_mm.jsonl",
             "as":      f"{FROZEN}/scored_as_mm.jsonl"}
QTYPES = ["number", "yes/no", "unanswerable", "other"]


def _load_text(d, glb, key):
    m = {}
    for fp in sorted(glob.glob(f"{d}/{glb}")):
        for r in read_jsonl(fp):
            if r.get(key) and r.get("id"):
                m[r["id"]] = r[key]
    return m


def _flags(text, uniq_ns, q_ns):
    t = text or ""
    t_ns = _nospace(t)
    hits = [g for g in uniq_ns if g and g in t_ns]
    novel = [g for g in hits if g not in q_ns]
    return {
        "any_novel_gold_mention": int(len(novel) > 0),
        "result_proposition": _result_prop(t),
        "candidate_enumeration": _cand_enum(t),
    }


def _pct(x):
    return "n/a" if x is None or x != x else f"{100*x:.1f}%"


def _f4(x):
    return "n/a" if x is None or x != x else f"{x:.4f}"


def _dstr(dp):
    return (f"{dp['delta']:+.4f}  [{dp['lo']:+.4f}, {dp['hi']:+.4f}]  "
            + ("(excludes 0)" if dp["excl0"] else "(includes 0)"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frozen-dir", default=FROZEN)
    ap.add_argument("--directas-dir", default=F2)
    ap.add_argument("--manifest", default=f"{FROZEN}/manifest.jsonl")
    args = ap.parse_args()
    selftest()

    man = read_jsonl(resolve(args.manifest))
    ids = [r["id"] for r in man]
    m_by = {r["id"]: r for r in man}
    n = len(ids)

    # ---- representation texts + tokens -------------------------------------------------
    tok = _tok_counter()
    reptxt = {mth: _load_text(*(REP[mth][0], REP[mth][1], REP[mth][2])) for mth in METHODS}
    for mth in METHODS:
        miss = [i for i in ids if i not in reptxt[mth]]
        if miss:
            raise SystemExit(f"[final-compare] {mth}: missing representation for {len(miss)} ids (e.g. {miss[:3]})")

    # ---- MM soft scores per id ------------------------------------------------------
    S = {}
    for mth in ("generic", "hclt", "as"):
        by = {r["id"]: r.get("soft_score") for r in read_jsonl(MM_SCORED[mth])}
        S[mth] = [by.get(i) for i in ids]
    # I+Q reference
    iqby = {r["id"]: r.get("soft_score") for r in read_jsonl(f"{FROZEN}/scored_iq.jsonl")}
    S["iq"] = [iqby.get(i) for i in ids]
    # DirectAS-r2: score fresh from full2k predictions
    dpred = {}
    for fp in sorted(glob.glob(f"{args.directas_dir}/predictions_directas_mm.jsonl")
                     + glob.glob(f"{args.directas_dir}/predictions_directas_mm.shard*.jsonl")):
        for r in read_jsonl(fp):
            if "pred" in r and r.get("id"):
                dpred[r["id"]] = r["pred"]
    miss = [i for i in ids if i not in dpred]
    if miss:
        raise SystemExit(f"[final-compare] DirectAS MM predictions missing for {len(miss)} ids (e.g. {miss[:3]})")
    da_sc, da_k, da_np = {}, {}, {}
    for i in ids:
        s, k, npd = soft_accuracy(dpred[i], m_by[i]["answers_10"])
        da_sc[i], da_k[i], da_np[i] = s, k, npd
    S["directas"] = [da_sc[i] for i in ids]
    with open(f"{args.directas_dir}/scored_directas_mm.jsonl", "w", encoding="utf-8") as f:
        for i in ids:
            f.write(json.dumps({
                "id": i, "answer_type": m_by[i]["answer_type"],
                "raw_prediction": dpred[i], "normalized_prediction": da_np[i],
                "answers_10_raw": m_by[i]["answers_10"],
                "answers_10_normalized": [normalize_answer(a) for a in m_by[i]["answers_10"]],
                "k": da_k[i], "soft_score": da_sc[i],
            }, ensure_ascii=False) + "\n")

    # ---- leakage flags per id -----------------------------------------------------
    fr_leak = {mth: {r["id"]: r for r in read_jsonl(f"{FROZEN}/leakage_{mth}.jsonl")}
               for mth in ("generic", "hclt", "as")}
    FL = {mth: {"ng": [], "rp": [], "ce": [], "state": [], "ocr": []} for mth in METHODS}
    for i in ids:
        mrow = m_by[i]
        uniq = sorted({_nospace(a) for a in mrow["answers_10"]} - {""})
        q_ns = _nospace(mrow["question"])
        q_ocr = bool(_OCR_Q_KEY.search(mrow["question"] or ""))
        for mth in METHODS:
            txt = reptxt[mth].get(i, "")
            if mth in fr_leak:
                fr = fr_leak[mth].get(i, {})
                ng = int(fr.get("any_novel_gold_mention") or 0)
                rp = int(fr.get("result_proposition") or 0)
                ce = int(fr.get("candidate_enumeration") or 0)
            else:
                f = _flags(txt, uniq, q_ns)
                ng, rp, ce = f["any_novel_gold_mention"], f["result_proposition"], f["candidate_enumeration"]
            FL[mth]["ng"].append(ng); FL[mth]["rp"].append(rp); FL[mth]["ce"].append(ce)
            FL[mth]["state"].append(_state_conclusion(txt))
            FL[mth]["ocr"].append(int(q_ocr and bool(_OCR_LIT.search(txt))))
    rate = {mth: {k: sum(v) / n for k, v in FL[mth].items()} for mth in METHODS}

    # ---- tokens ----------------------------------------------------------------------
    LEN = {}
    for mth in METHODS:
        texts = [reptxt[mth][i] for i in ids]
        chars = [len(t) for t in texts]
        toks = [tok(t) for t in texts] if tok else []
        LEN[mth] = {
            "mean_chars": round(statistics.fmean(chars), 1),
            "median_chars": statistics.median(chars),
            "mean_tokens": round(statistics.fmean(toks), 1) if toks else None,
            "median_tokens": round(float(np.median(toks)), 1) if toks else None,
            "p25_tokens": round(float(np.percentile(toks, 25)), 1) if toks else None,
            "p75_tokens": round(float(np.percentile(toks, 75)), 1) if toks else None,
        }
    gtok = LEN["generic"]["mean_tokens"]
    for mth in METHODS:
        mt = LEN[mth]["mean_tokens"]
        LEN[mth]["token_reduction_vs_generic_pct"] = round((1 - mt / gtok) * 100, 1) if (mt and gtok) else None

    # ---- acc + CI -------------------------------------------------------------------
    ACC = {}
    for mth in METHODS + ["iq"]:
        mean, lo, hi, k = bmean(S[mth])
        ACC[mth] = {"acc": mean, "lo": lo, "hi": hi, "n": k}

    # ---- pairwise bootstrap (whole 2K) --------------------------------------------
    PW = {
        "MM: HCLT - Generic":     dpair(S["hclt"], S["generic"]),
        "MM: Ours-AS - HCLT":     dpair(S["as"], S["hclt"]),
        "MM: DirectAS-r2 - HCLT": dpair(S["directas"], S["hclt"]),
        "MM: DirectAS-r2 - Ours-AS": dpair(S["directas"], S["as"]),
    }
    EXP = {
        "Exposure: DirectAS-r2 - HCLT":   dpair([float(x) for x in FL["directas"]["ng"]], [float(x) for x in FL["hclt"]["ng"]]),
        "Exposure: DirectAS-r2 - Ours-AS": dpair([float(x) for x in FL["directas"]["ng"]], [float(x) for x in FL["as"]["ng"]]),
        "Exposure: Ours-AS - HCLT":       dpair([float(x) for x in FL["as"]["ng"]], [float(x) for x in FL["hclt"]["ng"]]),
    }

    # ---- question-type table (OFFICIAL manifest.answer_type) ----------------------
    idx_by_type = {t: [k for k, i in enumerate(ids) if m_by[i]["answer_type"] == t] for t in QTYPES}
    QT = {}
    for t in QTYPES:
        idxs = idx_by_type[t]
        row = {"N": len(idxs)}
        for mth in METHODS:
            vals = [S[mth][k] for k in idxs if S[mth][k] is not None]
            row[mth] = statistics.fmean(vals) if vals else float("nan")
        # key paired deltas within type
        sub = lambda arr: [arr[k] for k in idxs]
        row["directas_minus_as"] = dpair(sub(S["directas"]), sub(S["as"]))
        row["directas_minus_hclt"] = dpair(sub(S["directas"]), sub(S["hclt"]))
        row["small_n"] = len(idxs) < 50
        QT[t] = row

    # ---- supplementary: OCR-keyword slice (NON-authoritative, heuristic) ----------
    ocr_idx = [k for k, i in enumerate(ids) if _OCR_Q_KEY.search(m_by[i]["question"] or "")]
    OCR_SLICE = {"N": len(ocr_idx)}
    for mth in METHODS:
        vals = [S[mth][k] for k in ocr_idx if S[mth][k] is not None]
        OCR_SLICE[mth] = statistics.fmean(vals) if vals else float("nan")

    # ---- trade-off CSV/JSON -----------------------------------------------------
    trade = []
    for mth in METHODS:
        trade.append({
            "method": LABEL[mth],
            "lexical_answer_exposure_pct": round(100 * rate[mth]["ng"], 2),
            "mm_accuracy": round(ACC[mth]["acc"], 4),
            "mm_ci_lo": round(ACC[mth]["lo"], 4),
            "mm_ci_hi": round(ACC[mth]["hi"], 4),
            "mean_tokens": LEN[mth]["mean_tokens"],
            "token_reduction_vs_generic_pct": LEN[mth]["token_reduction_vs_generic_pct"],
            "mm_acc_per_100_tokens": round(ACC[mth]["acc"] / LEN[mth]["mean_tokens"] * 100, 4) if LEN[mth]["mean_tokens"] else None,
        })
    with open(f"{args.directas_dir}/tradeoff.json", "w", encoding="utf-8") as f:
        json.dump({"n": n, "iq_mm_accuracy": round(ACC["iq"]["acc"], 4), "methods": trade}, f, ensure_ascii=False, indent=2)
    with open(f"{args.directas_dir}/tradeoff.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(trade[0].keys()))
        w.writeheader()
        for r in trade:
            w.writerow(r)

    # ================================ report =====================================
    L = []
    P = L.append
    P("# KVQA FINAL 2K — representation comparison\n")
    P(f"N = {n}   (frozen `{FROZEN}/manifest.jsonl`, same ids / order, no resampling)\n")
    P("Generic / HCLT-style / Ours-AS = frozen full-2K outputs (not regenerated). "
      "Ours-DirectAS-r2 = frozen r2 config (`configs/kvqa_ours_directas.yaml`, revision r2), "
      "generated on the same 2K manifest; no Stage-3 suppressor.\n")
    P("**Text-only accuracy is excluded from this comparison by design.**\n")
    P(f"Paired bootstrap: 10 000 resamples, seed {BOOT_SEED} (frozen `boot_diff`).\n")
    P("\n> **Lexical Answer Exposure** is identical to the frozen NOVEL_GOLD_MENTION metric used in the\n"
      "> original experiment: it measures whether at least one normalized human-answer surface appears in\n"
      "> the representation while not already appearing in the question. It is a lexical surface proxy\n"
      "> (lexical answer-surface exposure), not a complete or semantic measure of answer leakage.\n")

    P("\n## Table 1 — Main\n")
    P("| Method | MM Acc ↑ | Lexical Answer Exposure ↓ | Mean Tokens ↓ | Token Reduction vs Generic ↑ |")
    P("|---|---:|---:|---:|---:|")
    for mth in METHODS:
        a, ln = ACC[mth], LEN[mth]
        P(f"| {LABEL[mth]} | {_f4(a['acc'])} [{_f4(a['lo'])}, {_f4(a['hi'])}] | {_pct(rate[mth]['ng'])} | "
          f"{ln['mean_tokens']} | {ln['token_reduction_vs_generic_pct']:+.1f}% |")
    P(f"| _{LABEL['iq']}_ | _{_f4(ACC['iq']['acc'])} [{_f4(ACC['iq']['lo'])}, {_f4(ACC['iq']['hi'])}]_ | _—_ | _—_ | _—_ |")
    P("\n_I+Q is a reference baseline only; no representation / token / exposure metric applies to it._\n")

    P("\n## Table 2 — Multimodal accuracy by question type (OFFICIAL KVQA `answer_type`)\n")
    P("| Question Type | N | Generic | HCLT | Ours-AS | DirectAS-r2 | DirectAS−AS (95% CI) | DirectAS−HCLT (95% CI) |")
    P("|---|---:|---:|---:|---:|---|---|")
    for t in QTYPES:
        r = QT[t]
        flag = "  ⚠︎N<50" if r["small_n"] else ""
        P(f"| {t}{flag} | {r['N']} | {_f4(r['generic'])} | {_f4(r['hclt'])} | {_f4(r['as'])} | {_f4(r['directas'])} | "
          f"{_dstr(r['directas_minus_as'])} | {_dstr(r['directas_minus_hclt'])} |")
    P("\n_All four question types are the dataset-provided labels validated against `OFFICIAL_ANSWER_TYPES` "
      "in `kvqa_final_build.py`; the same label set is applied to every method. A finer OCR/text-like "
      "split is not available in the frozen labels and is not introduced here (would require a new classifier). "
      "See the heuristic keyword slice below (non-authoritative).__\n")
    P(f"\n_Heuristic OCR-keyword slice (question contains 날짜/기한/유통/가격/코드/번호/… — NON-authoritative, "
      f"not used for claims): N={OCR_SLICE['N']}  Generic {_f4(OCR_SLICE['generic'])}  HCLT {_f4(OCR_SLICE['hclt'])}  "
      f"Ours-AS {_f4(OCR_SLICE['as'])}  DirectAS-r2 {_f4(OCR_SLICE['directas'])}_\n")

    P("\n## Table 3 — Pairwise statistics (paired bootstrap, whole 2K)\n")
    P("| Comparison | Delta | 95% CI | excludes 0 |")
    P("|---|---:|---|---:|")
    for k, v in PW.items():
        P(f"| {k} | {v['delta']:+.4f} | [{v['lo']:+.4f}, {v['hi']:+.4f}] | {'yes' if v['excl0'] else 'no'} |")
    for k, v in EXP.items():
        P(f"| {k} (rate) | {v['delta']:+.4f} | [{v['lo']:+.4f}, {v['hi']:+.4f}] | {'yes' if v['excl0'] else 'no'} |")

    P("\n## Table 4 — Secondary diagnostics\n")
    P("| Method | ResultProp | CandEnum | State/Yes-No Conclusion (aux) | OCR Literal (aux) |")
    P("|---|---:|---:|---:|---:|")
    for mth in METHODS:
        P(f"| {LABEL[mth]} | {_pct(rate[mth]['rp'])} | {_pct(rate[mth]['ce'])} | {_pct(rate[mth]['state'])} | {_pct(rate[mth]['ocr'])} |")
    P("\n_ResultProp / CandEnum are the frozen main2000_diag definitions (Generic/HCLT/AS from the frozen "
      "`leakage_*.jsonl`; DirectAS computed with the identical defs). State/Yes-No Conclusion and OCR Literal "
      "are auxiliary heuristics introduced for r2 review — NOT frozen metrics and NOT part of the main "
      "exposure claim; DirectAS's longer visual descriptions raise the state-conclusion flag._\n")

    P("\n## Representation length (Qwen tokenizer)\n")
    P("| Method | mean char | median char | mean tok | median tok | p25 tok | p75 tok | MM Acc / 100 tok (suppl.) |")
    P("|---|---:|---:|---:|---:|---:|---:|---:|")
    for mth in METHODS:
        ln = LEN[mth]
        eff = ACC[mth]["acc"] / ln["mean_tokens"] * 100 if ln["mean_tokens"] else float("nan")
        P(f"| {LABEL[mth]} | {ln['mean_chars']} | {ln['median_chars']} | {ln['mean_tokens']} | "
          f"{ln['median_tokens']} | {ln['p25_tokens']} | {ln['p75_tokens']} | {_f4(eff)} |")
    P("\n_MM Acc / 100 tokens is a supplementary diagnostic only; shorter is not inherently better._\n")

    P("\n## Utility–Exposure and Utility–Length trade-off\n```")
    P(f"{'method':20} {'exposure%':>10} {'MM acc':>9} {'mean tok':>9}")
    for mth in METHODS:
        P(f"{LABEL[mth]:20} {100*rate[mth]['ng']:>9.1f}% {ACC[mth]['acc']:>9.4f} {LEN[mth]['mean_tokens']:>9}")
    P(f"{'(ref) I+Q':20} {'—':>10} {ACC['iq']['acc']:>9.4f} {'—':>9}")
    P("```")
    P(f"\nplot data -> `{args.directas_dir}/tradeoff.csv` , `{args.directas_dir}/tradeoff.json`\n")

    # ---- interpretation ---------------------------------------------------------
    best_mm = max(METHODS, key=lambda m: ACC[m]["acc"])
    low_exp = min(METHODS, key=lambda m: rate[m]["ng"])
    da_as = PW["MM: DirectAS-r2 - Ours-AS"]
    da_hclt = PW["MM: DirectAS-r2 - HCLT"]
    as_hclt = PW["MM: Ours-AS - HCLT"]
    hclt_gen = PW["MM: HCLT - Generic"]
    exp_da_as = EXP["Exposure: DirectAS-r2 - Ours-AS"]
    P("\n## Interpretation\n")
    P(f"1. Highest MM accuracy: **{LABEL[best_mm]}** ({_f4(ACC[best_mm]['acc'])}); the four representations "
      f"span {_f4(min(ACC[m]['acc'] for m in METHODS))}–{_f4(max(ACC[m]['acc'] for m in METHODS))}, "
      f"all below the I+Q reference {_f4(ACC['iq']['acc'])}.")
    P(f"2. Significance: HCLT−Generic {_dstr(hclt_gen)}; Ours-AS−HCLT {_dstr(as_hclt)}; "
      f"DirectAS−HCLT {_dstr(da_hclt)}; DirectAS−Ours-AS {_dstr(da_as)}. "
      f"MM differences among the auxiliary representations are "
      f"{'mostly within noise' if not any(v['excl0'] for v in PW.values()) else 'partly significant (see Table 3)'}.")
    P(f"3. Lowest lexical answer exposure: **{LABEL[low_exp]}** ({_pct(rate[low_exp]['ng'])}). "
      f"Order: Generic {_pct(rate['generic']['ng'])} · HCLT {_pct(rate['hclt']['ng'])} · "
      f"Ours-AS {_pct(rate['as']['ng'])} · DirectAS-r2 {_pct(rate['directas']['ng'])}.")
    P(f"4. DirectAS vs AS on exposure: {exp_da_as['delta']:+.4f} [{exp_da_as['lo']:+.4f}, {exp_da_as['hi']:+.4f}] "
      f"({'significant' if exp_da_as['excl0'] else 'not significant'}) — "
      f"DirectAS {'does' if (exp_da_as['excl0'] and exp_da_as['delta'] < 0) else 'does not clearly'} "
      f"reduce lexical exposure further than Ours-AS.")
    P(f"5. Does DirectAS lose MM utility vs AS? DirectAS_MM − AS_MM = {_dstr(da_as)} — "
      f"{'no significant loss' if not da_as['excl0'] else ('SIGNIFICANT LOSS' if da_as['delta'] < 0 else 'significant gain')}.")
    P(f"6. Token cost: Generic {LEN['generic']['mean_tokens']} → HCLT {LEN['hclt']['mean_tokens']} "
      f"({LEN['hclt']['token_reduction_vs_generic_pct']:+.1f}%) → Ours-AS {LEN['as']['mean_tokens']} "
      f"({LEN['as']['token_reduction_vs_generic_pct']:+.1f}%) → DirectAS-r2 {LEN['directas']['mean_tokens']} "
      f"({LEN['directas']['token_reduction_vs_generic_pct']:+.1f}%). DirectAS is the longest representation.")
    small = [t for t in QTYPES if QT[t]["small_n"]]
    P(f"7. Question type: per Table 2. {'Categories flagged N<50: ' + ', '.join(small) + ' (unstable, not used for claims). ' if small else ''}"
      f"number N={QT['number']['N']}, yes/no N={QT['yes/no']['N']}, unanswerable N={QT['unanswerable']['N']}, other N={QT['other']['N']}.")
    P("8. Best utility/exposure trade-off: see the trade-off block — the representation that keeps MM accuracy "
      "closest to the group max while sitting lowest on lexical answer exposure.")

    P("\n## Research questions\n")
    P(f"**R1 — Does question-conditioned answer suppression preserve MM utility better than "
      f"generic/post-hoc representations?**  Ours-AS_MM {_f4(ACC['as']['acc'])} vs Generic "
      f"{_f4(ACC['generic']['acc'])} / HCLT {_f4(ACC['hclt']['acc'])}; Ours-AS−HCLT {_dstr(as_hclt)}.")
    P(f"\n**R2 — Can DirectAS match Ours-AS MM accuracy while reducing lexical answer exposure?**  "
      f"MM DirectAS−AS {_dstr(da_as)}; Exposure DirectAS−AS {exp_da_as['delta']:+.4f} "
      f"[{exp_da_as['lo']:+.4f}, {exp_da_as['hi']:+.4f}].")
    P(f"\n**R3 — Token-cost trade-off of DirectAS's richer guidance?**  "
      f"mean tokens {LEN['directas']['mean_tokens']} vs Ours-AS {LEN['as']['mean_tokens']} "
      f"(x{LEN['directas']['mean_tokens']/LEN['as']['mean_tokens']:.2f}), vs Generic {LEN['generic']['mean_tokens']} "
      f"(Token Reduction {LEN['directas']['token_reduction_vs_generic_pct']:+.1f}%).")
    P(f"\n**R4 — Do the relative advantages differ across question types?**  See Table 2 / point 7.")

    txt = "\n".join(L) + "\n"
    out = f"{args.directas_dir}/final_compare_report.md"
    with open(out, "w", encoding="utf-8") as f:
        f.write(txt)
    print(txt)
    print(f"[final-compare] -> {out}")
    print(f"[final-compare] -> {args.directas_dir}/scored_directas_mm.jsonl , tradeoff.csv , tradeoff.json")


if __name__ == "__main__":
    main()
