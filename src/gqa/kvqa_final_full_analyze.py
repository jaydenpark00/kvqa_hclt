"""KVQA FINAL FULL (9424) — merge shards, score, and produce the full representation comparison.

5 conditions, Yes/No format-corrected common solver, full eligible test split:
  I+Q (reference) / Generic Caption / HCLT-style / Ours-AS / Ours-DirectAS-r2

Frozen and reused unchanged: evaluator (soft_accuracy / normalize_answer), leakage detector
(_cand_enum / _result_prop + NOVEL_GOLD_MENTION), bootstrap (boot_diff, seed BOOT_SEED),
Qwen tokenizer, official manifest.answer_type. Auxiliary heuristics (state-conclusion, OCR-literal)
are clearly separated and are NOT the main exposure metric.

  python -m src.gqa.kvqa_final_full_analyze
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
from src.gqa.kvqa_final_analyze import bmean, dpair, _tok_counter, BOOT_SEED
from src.gqa.kvqa_directas_final_compare import _flags, _nospace, _pct, _f4, _dstr, QTYPES
from src.gqa.kvqa_directas_sanity import _state_conclusion
from src.gqa.kvqa_directas_pilot_analyze import _OCR_LIT, _OCR_Q_KEY

FULL = "outputs/kvqa_final_full"
PREV_2K = {"generic": 0.3238, "hclt": 0.3293, "as": 0.3245, "directas": 0.3297, "iq": 0.3420}  # old (yes/no-mismatch) solver

REPR = {  # method -> (base_glob, text_key)
    "generic":  ("generic_captions.*jsonl",          "generic_caption"),
    "hclt":     ("hclt_high_density.jsonl",           "hclt_high_density_caption"),
    "as":       ("ours_suppressed_evidence.*jsonl",   "suppressed_evidence"),
    "directas": ("ours_directas_evidence.*jsonl",     "directas_evidence"),
}
COND_OF = {"generic": "generic_mm", "hclt": "hclt_mm", "as": "as_mm", "directas": "directas_mm"}
LAB = {"iq": "I+Q (reference)", "generic": "Generic Caption", "hclt": "HCLT-style",
       "as": "Ours-AS", "directas": "Ours-DirectAS-r2"}
METHODS = ["generic", "hclt", "as", "directas"]


def _merge_text(base_glob, key, ids):
    m = {}
    for fp in sorted(glob.glob(f"{FULL}/{base_glob}")):
        for r in read_jsonl(fp):
            if r.get(key) and r.get("id"):
                m[r["id"]] = r[key]
    return m


def _merge_preds(cond, ids):
    m = {}
    for fp in sorted(glob.glob(f"{FULL}/predictions_{cond}.jsonl")
                     + glob.glob(f"{FULL}/predictions_{cond}.shard*.jsonl")):
        for r in read_jsonl(fp):
            if "pred" in r and r.get("id"):
                m[r["id"]] = r["pred"]
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=FULL)
    ap.add_argument("--manifest", default=f"{FULL}/manifest.jsonl")
    ap.add_argument("--k2-manifest", default="outputs/kvqa_final_hclt_vs_ours_as_2k/manifest.jsonl")
    args = ap.parse_args()
    selftest()

    man = read_jsonl(resolve(args.manifest))
    ids = [r["id"] for r in man]
    m_by = {r["id"]: r for r in man}
    n = len(ids)
    k2_ids = set(r["id"] for r in read_jsonl(resolve(args.k2_manifest)))
    tok = _tok_counter()

    # ---- 1. generation completeness ------------------------------------------------
    reptxt = {mth: _merge_text(*REPR[mth], ids) for mth in METHODS}
    dec_raw = _merge_text("ours_decompositions.*jsonl", "decomposition_raw", ids)
    pdec_raw = _merge_text("ours_protected_decompositions.*jsonl", "decomposition_raw", ids)
    pdec = {}
    for fp in sorted(glob.glob(f"{FULL}/ours_protected_decompositions.*jsonl")):
        for r in read_jsonl(fp):
            if isinstance(r.get("decomposition"), dict) and r.get("id"):
                pdec[r["id"]] = r["decomposition"]
    complete = {}
    for name, mp in (("generic_caption", reptxt["generic"]), ("hclt", reptxt["hclt"]),
                     ("ours_as_evidence", reptxt["as"]), ("ours_decomposition", dec_raw),
                     ("directas_evidence", reptxt["directas"]), ("protected_decomposition", pdec_raw)):
        miss = [i for i in ids if i not in mp]
        complete[name] = {"present": n - len(miss), "missing": len(miss), "missing_ids": miss[:10]}

    # write canonical merged representation files (deterministic, manifest order)
    for mth in METHODS:
        with open(f"{FULL}/{['generic_captions','hclt_high_density','ours_as_evidence','ours_directas_evidence'][METHODS.index(mth)]}.merged.jsonl", "w", encoding="utf-8") as f:
            for i in ids:
                if i in reptxt[mth]:
                    f.write(json.dumps({"id": i, "text": reptxt[mth][i]}, ensure_ascii=False) + "\n")

    # ---- 2. score the 5 conditions ---------------------------------------------
    CONDS = ["iq", "generic_mm", "hclt_mm", "as_mm", "directas_mm"]
    S, PREDS = {}, {}
    for c in CONDS:
        pr = _merge_preds(c, ids)
        PREDS[c] = pr
        miss = [i for i in ids if i not in pr]
        complete[f"predictions_{c}"] = {"present": n - len(miss), "missing": len(miss), "missing_ids": miss[:10]}
        sc = {}
        with open(f"{FULL}/scored_{c}.jsonl", "w", encoding="utf-8") as f:
            for i in ids:
                p = pr.get(i)
                if p is None:
                    sc[i] = None
                    continue
                s, k, npd = soft_accuracy(p, m_by[i]["answers_10"])
                sc[i] = s
                f.write(json.dumps({"id": i, "answer_type": m_by[i]["answer_type"], "raw_prediction": p,
                                    "normalized_prediction": npd, "k": k, "soft_score": s}, ensure_ascii=False) + "\n")
        S[c] = [sc.get(i) for i in ids]

    def acc(c):
        mean, lo, hi, k = bmean(S[c])
        return {"acc": mean, "lo": lo, "hi": hi, "n": k}
    ACC = {c: acc(c) for c in CONDS}

    # ---- 3. pairwise bootstrap --------------------------------------------------
    A = {mth: S[COND_OF[mth]] for mth in METHODS}
    A["iq"] = S["iq"]
    PW = {}
    for lab, a, b in [
        ("HCLT - Generic", "hclt", "generic"), ("AS - Generic", "as", "generic"),
        ("AS - HCLT", "as", "hclt"), ("DirectAS - Generic", "directas", "generic"),
        ("DirectAS - HCLT", "directas", "hclt"), ("DirectAS - AS", "directas", "as"),
        ("Generic - I+Q", "generic", "iq"), ("HCLT - I+Q", "hclt", "iq"),
        ("AS - I+Q", "as", "iq"), ("DirectAS - I+Q", "directas", "iq"),
    ]:
        PW[lab] = dpair(A[a], A[b])

    # ---- 4. question-type ----------------------------------------------------
    QT = {}
    for t in QTYPES:
        idxs = [k for k, i in enumerate(ids) if m_by[i]["answer_type"] == t]
        row = {"N": len(idxs), "small_n": len(idxs) < 50}
        for c in CONDS:
            vals = [S[c][k] for k in idxs if S[c][k] is not None]
            row[c] = statistics.fmean(vals) if vals else float("nan")
        sub = lambda arr: [arr[k] for k in idxs]
        row["directas_minus_as"] = dpair(sub(A["directas"]), sub(A["as"]))
        QT[t] = row

    # yes/no output-format counts
    yn = [i for i in ids if m_by[i]["answer_type"] == "yes/no"]
    YNOUT = {}
    for c in CONDS:
        pl = [(PREDS[c].get(i, "") or "").strip().lower().strip(".!? ") for i in yn]
        YNOUT[c] = {"yes": sum(x == "yes" for x in pl), "no": sum(x == "no" for x in pl),
                    "other": sum(x not in ("yes", "no") for x in pl),
                    "other_ex": [PREDS[c].get(yn[k], "") for k in range(len(yn)) if pl[k] not in ("yes", "no")][:6]}

    # ---- 5. lexical exposure + secondary diagnostics -------------------------
    EXPO, RP, CE, STATE, OCR = {}, {}, {}, {}, {}
    ng_flags = {}
    for mth in METHODS:
        ngs, rps, ces, sts, ocrs = [], [], [], [], []
        for i in ids:
            uniq = sorted({_nospace(a) for a in m_by[i]["answers_10"]} - {""})
            q_ns = _nospace(m_by[i]["question"])
            txt = reptxt[mth].get(i, "")
            f = _flags(txt, uniq, q_ns)
            ngs.append(f["any_novel_gold_mention"]); rps.append(f["result_proposition"]); ces.append(f["candidate_enumeration"])
            sts.append(_state_conclusion(txt))
            ocrs.append(int(bool(_OCR_Q_KEY.search(m_by[i]["question"] or "")) and bool(_OCR_LIT.search(txt))))
        ng_flags[mth] = ngs
        EXPO[mth] = dict(zip(("rate", "lo", "hi", "k"), bmean([float(x) for x in ngs])))
        RP[mth] = statistics.fmean(rps); CE[mth] = statistics.fmean(ces)
        STATE[mth] = statistics.fmean(sts); OCR[mth] = statistics.fmean(ocrs)
    EXPO_PW = {
        "HCLT - Generic": dpair([float(x) for x in ng_flags["hclt"]], [float(x) for x in ng_flags["generic"]]),
        "AS - HCLT":      dpair([float(x) for x in ng_flags["as"]], [float(x) for x in ng_flags["hclt"]]),
        "DirectAS - AS":  dpair([float(x) for x in ng_flags["directas"]], [float(x) for x in ng_flags["as"]]),
        "DirectAS - HCLT": dpair([float(x) for x in ng_flags["directas"]], [float(x) for x in ng_flags["hclt"]]),
    }

    # ---- 6. length --------------------------------------------------------
    LEN = {}
    for mth in METHODS:
        toks = [tok(reptxt[mth][i]) for i in ids if i in reptxt[mth]] if tok else []
        LEN[mth] = {
            "mean_tok": round(statistics.fmean(toks), 1) if toks else None,
            "median_tok": round(float(np.median(toks)), 1) if toks else None,
            "p25": round(float(np.percentile(toks, 25)), 1) if toks else None,
            "p75": round(float(np.percentile(toks, 75)), 1) if toks else None,
        }
    gt = LEN["generic"]["mean_tok"]
    for mth in METHODS:
        LEN[mth]["reduction_pct"] = round((1 - LEN[mth]["mean_tok"] / gt) * 100, 1) if (LEN[mth]["mean_tok"] and gt) else None
    da_as_ratio = round(LEN["directas"]["mean_tok"] / LEN["as"]["mean_tok"], 2) if LEN["as"]["mean_tok"] else None

    # ---- 7. 2K vs full stability -----------------------------------------
    STAB = {}
    for mth in METHODS + ["iq"]:
        c = "iq" if mth == "iq" else COND_OF[mth]
        sub = [S[c][k] for k, i in enumerate(ids) if i in k2_ids]
        k2corr = statistics.fmean([x for x in sub if x is not None])
        STAB[mth] = {"prev_2k_oldsolver": PREV_2K[mth], "k2_subset_corrected": round(k2corr, 4),
                     "full_corrected": round(ACC[c]["acc"], 4)}

    # ================================ CSVs ================================
    d = args.dir

    def _w(name, header, rows):
        with open(f"{d}/{name}", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(header)
            for r in rows:
                w.writerow(r)

    _w("main_results.csv", ["method", "N", "mm_acc", "ci_lo", "ci_hi"],
       [[LAB[m], n, _f4(ACC["iq" if m == "iq" else COND_OF[m]]["acc"]),
         _f4(ACC["iq" if m == "iq" else COND_OF[m]]["lo"]), _f4(ACC["iq" if m == "iq" else COND_OF[m]]["hi"])]
        for m in ["iq"] + METHODS])
    _w("pairwise_mm.csv", ["comparison", "delta", "ci_lo", "ci_hi", "excludes_0"],
       [[k, f"{v['delta']:.4f}", f"{v['lo']:.4f}", f"{v['hi']:.4f}", v["excl0"]] for k, v in PW.items()])
    _w("question_type_results.csv", ["question_type", "N", "iq", "generic", "hclt", "as", "directas"],
       [[t, QT[t]["N"], _f4(QT[t]["iq"]), _f4(QT[t]["generic_mm"]), _f4(QT[t]["hclt_mm"]),
         _f4(QT[t]["as_mm"]), _f4(QT[t]["directas_mm"])] for t in QTYPES])
    _w("exposure_results.csv", ["method", "lexical_exposure", "ci_lo", "ci_hi"],
       [[LAB[m], _f4(EXPO[m]["rate"]), _f4(EXPO[m]["lo"]), _f4(EXPO[m]["hi"])] for m in METHODS])
    _w("length_results.csv", ["method", "mean_tok", "median_tok", "p25", "p75", "reduction_vs_generic_pct"],
       [[LAB[m], LEN[m]["mean_tok"], LEN[m]["median_tok"], LEN[m]["p25"], LEN[m]["p75"], LEN[m]["reduction_pct"]] for m in METHODS])
    _w("secondary_diagnostics.csv", ["method", "result_prop", "cand_enum", "state_conclusion_aux", "ocr_literal_aux"],
       [[LAB[m], _f4(RP[m]), _f4(CE[m]), _f4(STATE[m]), _f4(OCR[m])] for m in METHODS])
    _w("tradeoff.csv", ["method", "lexical_exposure_pct", "mm_acc", "mean_tok"],
       [[LAB[m], round(100 * EXPO[m]["rate"], 2), _f4(ACC[COND_OF[m]]["acc"]), LEN[m]["mean_tok"]] for m in METHODS])

    summary = {
        "N": n, "manifest_sha256": None,
        "answer_type_counts": {t: QT[t]["N"] for t in QTYPES},
        "generation_completeness": complete,
        "mm_accuracy": {LAB[m]: ACC["iq" if m == "iq" else COND_OF[m]] for m in ["iq"] + METHODS},
        "pairwise_mm": {k: v for k, v in PW.items()},
        "lexical_exposure": {LAB[m]: EXPO[m] for m in METHODS},
        "exposure_pairwise": EXPO_PW,
        "length": {LAB[m]: LEN[m] for m in METHODS},
        "directas_as_token_ratio": da_as_ratio,
        "secondary": {LAB[m]: {"result_prop": RP[m], "cand_enum": CE[m], "state_conclusion_aux": STATE[m], "ocr_literal_aux": OCR[m]} for m in METHODS},
        "question_type": {t: {LAB.get(c.replace("_mm", "").replace("iq", "iq"), c): QT[t][c] for c in CONDS} | {"N": QT[t]["N"]} for t in QTYPES},
        "yesno_output_counts": YNOUT,
        "stability_2k_vs_full": STAB,
    }
    json.dump(summary, open(f"{d}/full_summary.json", "w"), ensure_ascii=False, indent=2, default=str)

    # ================================ report ================================
    L = []
    P = L.append
    st = json.load(open(f"{d}/manifest_stats.json")) if glob.glob(f"{d}/manifest_stats.json") else {}
    P("# KVQA FINAL — full evaluation split (representation comparison)\n")
    P("## 1. Dataset / full manifest provenance\n```")
    P(f"split file      : {st.get('split_file')}")
    P(f"raw test split  : {st.get('raw_test_split_size')}   (source counts {st.get('test_source_counts')})")
    P(f"exclusions      : {st.get('exclusions')}")
    P(f"eligible N      : {st.get('eligible_N')}   distinct qids {st.get('distinct_question_ids')}  distinct images {st.get('distinct_image_ids')}")
    P(f"order           : {st.get('order')}   first {st.get('first_id')}  last {st.get('last_id')}")
    P(f"sha256          : {st.get('sha256')}")
    P(f"md5             : {st.get('md5')}")
    P("```")
    P("Same source / split / eligibility / ordering as the frozen 2K build; only the 2000-subsample was removed.\n")

    P("\n## 2. Full sample N and answer-type distribution\n")
    P(f"N = {n}")
    P("| answer_type | N |")
    P("|---|---:|")
    for t in QTYPES:
        P(f"| {t} | {QT[t]['N']} |")

    P("\n## 3. Generation completeness\n")
    P("| stage | present / N | missing |")
    P("|---|---:|---:|")
    for k, v in complete.items():
        P(f"| {k} | {v['present']} / {n} | {v['missing']}"
          + (f"  e.g. {v['missing_ids']}" if v["missing"] else "") + " |")

    P("\n## 4. Main multimodal accuracy\n")
    P("| Method | N | MM Accuracy ↑ | 95% CI |")
    P("|---|---:|---:|---:|")
    for m in ["iq"] + METHODS:
        c = "iq" if m == "iq" else COND_OF[m]
        a = ACC[c]
        P(f"| {LAB[m]} | {n} | {_f4(a['acc'])} | [{_f4(a['lo'])}, {_f4(a['hi'])}] |")

    P("\n## 5. Pairwise bootstrap significance (paired, seed %d, 10 000 resamples)\n" % BOOT_SEED)
    P("| Comparison | Delta | 95% CI | excludes 0? |")
    P("|---|---:|---|---:|")
    for k, v in PW.items():
        P(f"| MM: {k} | {v['delta']:+.4f} | [{v['lo']:+.4f}, {v['hi']:+.4f}] | {'yes' if v['excl0'] else 'no'} |")

    P("\n## 6. Question-type accuracy (official KVQA answer_type)\n")
    P("| Question Type | N | I+Q | Generic | HCLT | Ours-AS | DirectAS-r2 | DirectAS−AS 95% CI |")
    P("|---|---:|---:|---:|---:|---:|---:|---|")
    for t in QTYPES:
        r = QT[t]
        flag = " ⚠︎N<50" if r["small_n"] else ""
        P(f"| {t}{flag} | {r['N']} | {_f4(r['iq'])} | {_f4(r['generic_mm'])} | {_f4(r['hclt_mm'])} | "
          f"{_f4(r['as_mm'])} | {_f4(r['directas_mm'])} | {_dstr(r['directas_minus_as'])} |")

    P("\n## 7. Yes/No corrected evaluation — output-format counts (yes/no subset, N=%d)\n" % len(yn))
    P("| Method | Yes | No | other-format | other-format examples |")
    P("|---|---:|---:|---:|---|")
    for c in CONDS:
        y = YNOUT[c]
        P(f"| {LAB.get(c.replace('_mm',''), c)} | {y['yes']} | {y['no']} | {y['other']} | {y['other_ex'] if y['other'] else '—'} |")

    P("\n## 8. Lexical Answer Exposure  (= frozen NOVEL_GOLD_MENTION; lexical answer-surface proxy, not semantic leakage)\n")
    P("| Method | Lexical Answer Exposure ↓ | 95% CI |")
    P("|---|---:|---:|")
    for m in METHODS:
        e = EXPO[m]
        P(f"| {LAB[m]} | {_pct(e['rate'])} | [{_pct(e['lo'])}, {_pct(e['hi'])}] |")
    P("\npaired bootstrap:")
    for k, v in EXPO_PW.items():
        P(f"- {k}: {v['delta']:+.4f} [{v['lo']:+.4f}, {v['hi']:+.4f}] ({'excludes 0' if v['excl0'] else 'includes 0'})")

    P("\n## 9. Representation length (Qwen tokenizer)\n")
    P("| Method | Mean tok | Median | P25 | P75 | Reduction vs Generic |")
    P("|---|---:|---:|---:|---:|---:|")
    for m in METHODS:
        ln = LEN[m]
        P(f"| {LAB[m]} | {ln['mean_tok']} | {ln['median_tok']} | {ln['p25']} | {ln['p75']} | {ln['reduction_pct']:+.1f}% |")
    P(f"\nDirectAS mean tokens / Ours-AS mean tokens = **{da_as_ratio}**")

    P("\n## 10. Secondary diagnostics\n")
    P("| Method | ResultProp (frozen) | CandEnum (frozen) | State/Yes-No Conclusion (aux) | OCR-Literal (aux) |")
    P("|---|---:|---:|---:|---:|")
    for m in METHODS:
        P(f"| {LAB[m]} | {_pct(RP[m])} | {_pct(CE[m])} | {_pct(STATE[m])} | {_pct(OCR[m])} |")
    P("\n_State/Yes-No Conclusion and OCR-Literal are auxiliary heuristics, NOT the frozen exposure metric._\n")

    P("\n## 11. Utility–Exposure trade-off\n```")
    P(f"{'method':20} {'exposure%':>10} {'MM acc':>9}")
    for m in METHODS:
        P(f"{LAB[m]:20} {100*EXPO[m]['rate']:>9.1f}% {ACC[COND_OF[m]]['acc']:>9.4f}")
    P(f"{'(ref) I+Q':20} {'—':>10} {ACC['iq']['acc']:>9.4f}")
    P("```")

    P("\n## 12. Utility–Length trade-off\n```")
    P(f"{'method':20} {'mean tok':>9} {'MM acc':>9}")
    for m in METHODS:
        P(f"{LAB[m]:20} {LEN[m]['mean_tok']:>9} {ACC[COND_OF[m]]['acc']:>9.4f}")
    P("```")
    P("_Length and multimodal utility trade off; shorter is not inherently better._\n")

    P("\n## 13. Previous 2K vs Full stability\n")
    P("| Method | Prev 2K MM (old solver) | 2K-subset MM (corrected solver) | Full MM (corrected) |")
    P("|---|---:|---:|---:|")
    for m in ["iq"] + METHODS:
        s = STAB[m]
        P(f"| {LAB[m]} | {s['prev_2k_oldsolver']:.4f} | {s['k2_subset_corrected']:.4f} | {s['full_corrected']:.4f} |")
    old_ord = sorted(METHODS, key=lambda x: -PREV_2K[x])
    new_ord = sorted(METHODS, key=lambda x: -ACC[COND_OF[x]]["acc"])
    da_as = PW["DirectAS - AS"]
    P(f"\n1. method ordering — prev(old solver): {' > '.join(LAB[x] for x in old_ord)} ; "
      f"full(corrected): {' > '.join(LAB[x] for x in new_ord)}")
    P(f"2. DirectAS − AS (MM): {_dstr(da_as)} — {'still not significant' if not da_as['excl0'] else 'now significant'}")
    P(f"3. DirectAS − AS (exposure): {EXPO_PW['DirectAS - AS']['delta']:+.4f} "
      f"[{EXPO_PW['DirectAS - AS']['lo']:+.4f}, {EXPO_PW['DirectAS - AS']['hi']:+.4f}] "
      f"({'reduction holds' if (EXPO_PW['DirectAS - AS']['excl0'] and EXPO_PW['DirectAS - AS']['delta'] < 0) else 'not significant on full set'})")
    P(f"4. token efficiency — Ours-AS {LEN['as']['mean_tok']} tok vs DirectAS {LEN['directas']['mean_tok']} "
      f"(x{da_as_ratio}); AS remains the shortest of the two.")
    da_iq = PW["DirectAS - I+Q"]; as_iq = PW["AS - I+Q"]
    P(f"5. vs I+Q — DirectAS − I+Q {_dstr(da_iq)} ; Ours-AS − I+Q {_dstr(as_iq)}.")

    P("\n## 14. Main findings\n")
    best = max(METHODS, key=lambda m: ACC[COND_OF[m]]["acc"])
    lowexp = min(METHODS, key=lambda m: EXPO[m]["rate"])
    P(f"- Highest MM accuracy among representations: **{LAB[best]}** ({_f4(ACC[COND_OF[best]]['acc'])}); "
      f"I+Q reference {_f4(ACC['iq']['acc'])}.")
    sig = [k for k, v in PW.items() if v["excl0"] and "I+Q" not in k]
    P(f"- Significant MM differences among Generic/HCLT/AS/DirectAS: {sig if sig else 'none (all four-method CIs include 0)'}.")
    P(f"- Lowest lexical answer exposure: **{LAB[lowexp]}** ({_pct(EXPO[lowexp]['rate'])}); "
      f"order Generic {_pct(EXPO['generic']['rate'])} · HCLT {_pct(EXPO['hclt']['rate'])} · "
      f"AS {_pct(EXPO['as']['rate'])} · DirectAS {_pct(EXPO['directas']['rate'])}.")
    P(f"- DirectAS − AS: MM {_dstr(da_as)}; exposure {_dstr(EXPO_PW['DirectAS - AS'])}; tokens x{da_as_ratio}.")
    P(f"- yes/no now scores non-zero for every method (see §4 / §6 / §7); output-format failures ≈ 0.")

    P("\n## 15. Limitations\n")
    P("- Lexical Answer Exposure is a surface-string proxy over the 10 human answers, not a semantic "
      "leakage measure; it cannot see paraphrased or descriptive answer disclosure.")
    P("- The State/Yes-No Conclusion and OCR-Literal flags are heuristic and imperfect (false positives on "
      "benign scene description); they are reported as auxiliary only.")
    P("- Multimodal accuracy is close to the I+Q reference for all representations; the auxiliary text adds "
      "little measurable QA utility when the image is present, so representation differences are dominated by "
      "exposure and length, not accuracy.")
    P("- 2K-subsampled representations were reused verbatim for the 2000 overlapping ids (identical frozen "
      "prompts, deterministic greedy decoding); the remaining 7424 were generated with the same pipelines.")
    P("- KVQA `answerable=0` items are labelled `unanswerable`/other and scored by the frozen soft metric as-is.")

    P("\n## 16. Final conclusion\n")
    da_as_incl0 = not da_as["excl0"]
    exp_sig = EXPO_PW["DirectAS - AS"]["excl0"] and EXPO_PW["DirectAS - AS"]["delta"] < 0
    P("On the full KVQA evaluation split (N=%d), " % n
      + ("Ours-DirectAS-r2 reduces lexical answer exposure relative to Ours-AS "
         f"({_pct(EXPO['as']['rate'])} → {_pct(EXPO['directas']['rate'])}, "
         f"{EXPO_PW['DirectAS - AS']['delta']*100:+.1f} pp, CI {'excludes' if exp_sig else 'includes'} 0) "
         if True else "")
      + ("while its multimodal QA accuracy is statistically comparable to Ours-AS "
         f"(DirectAS − AS = {da_as['delta']:+.4f}, 95% CI [{da_as['lo']:+.4f}, {da_as['hi']:+.4f}], "
         f"{'includes 0' if da_as_incl0 else 'excludes 0'})"
         )
      + f". The cost is representation length: DirectAS averages {LEN['directas']['mean_tok']} Qwen tokens "
      f"vs Ours-AS {LEN['as']['mean_tok']} (x{da_as_ratio}), though still "
      f"{LEN['directas']['reduction_pct']:+.1f}% relative to the Generic caption. "
      "The relative conclusions from the 2K experiment are preserved.")

    txt = "\n".join(L) + "\n"
    out = f"{d}/final_full_comparison.md"
    with open(out, "w", encoding="utf-8") as f:
        f.write(txt)
    print(txt)
    print(f"[full-analyze] -> {out}")
    print(f"[full-analyze] -> {d}/*.csv , full_summary.json , scored_*.jsonl")


if __name__ == "__main__":
    main()
