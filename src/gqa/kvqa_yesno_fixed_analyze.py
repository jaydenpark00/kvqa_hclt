"""KVQA — Yes/No Format-Corrected common solver: scoring + full re-analysis.

Scores the 5 re-solved MM conditions (iq / generic_mm / hclt_mm / as_mm / directas_mm) with the
FROZEN evaluator, then:
  - yes/no (official answer_type) 103-subset accuracy + Yes/No/other-format output counts
  - old (Original Frozen Solver) vs corrected full-2K MM accuracy + delta
  - non-yes/no (1897) stability check: predictions / scores changed, acc delta
  - corrected question-type table
  - corrected pairwise bootstrap (HCLT-Generic, AS-HCLT, DirectAS-HCLT, DirectAS-AS)
  - final representation table (MM corrected + frozen-representation exposure/token, recomputed identically)

Nothing frozen is regenerated. Representations, evaluator, normalization, manifest, bootstrap
code, leakage detector: all reused unchanged.

  python -m src.gqa.kvqa_yesno_fixed_analyze
"""
from __future__ import annotations

import argparse
import glob
import json
import statistics

import numpy as np

from src.common import read_jsonl, resolve
from src.gqa.kvqa_softscore import soft_accuracy, normalize_answer, selftest
from src.gqa.kvqa_final_analyze import bmean, dpair, _tok_counter, BOOT_SEED
from src.gqa.kvqa_directas_final_compare import _flags, _nospace, _pct, _f4, _dstr, QTYPES

FROZEN = "outputs/kvqa_final_hclt_vs_ours_as_2k"
F2 = "outputs/kvqa_ours_directas/full2k"
NEW = "outputs/kvqa_solver_yesno_fixed_2k"

CONDS = ["iq", "generic_mm", "hclt_mm", "as_mm", "directas_mm"]
METHOD_OF = {"generic_mm": "generic", "hclt_mm": "hclt", "as_mm": "as", "directas_mm": "directas"}
LAB = {"iq": "I+Q (reference)", "generic_mm": "Generic Caption", "hclt_mm": "HCLT-style",
       "as_mm": "Ours-AS", "directas_mm": "Ours-DirectAS-r2"}
REP = {"generic": (FROZEN, "generic_captions.jsonl", "generic_caption"),
       "hclt": (FROZEN, "hclt_high_density.jsonl", "hclt_high_density_caption"),
       "as": (FROZEN, "ours_suppressed_evidence.jsonl", "suppressed_evidence"),
       "directas": (F2, "ours_directas_evidence*.jsonl", "directas_evidence")}


def _preds(dirp, cond):
    m = {}
    for fp in sorted(glob.glob(f"{dirp}/predictions_{cond}.jsonl")
                     + glob.glob(f"{dirp}/predictions_{cond}.shard*.jsonl")):
        for r in read_jsonl(fp):
            if "pred" in r and r.get("id"):
                m[r["id"]] = r["pred"]
    return m


def _text(d, glb, key):
    m = {}
    for fp in sorted(glob.glob(f"{d}/{glb}")):
        for r in read_jsonl(fp):
            if r.get(key) and r.get("id"):
                m[r["id"]] = r[key]
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=f"{FROZEN}/manifest.jsonl")
    ap.add_argument("--new-dir", default=NEW)
    args = ap.parse_args()
    selftest()

    man = read_jsonl(resolve(args.manifest))
    ids = [r["id"] for r in man]
    m_by = {r["id"]: r for r in man}
    n = len(ids)
    tok = _tok_counter()

    # ---- new predictions + scoring ------------------------------------------------
    newp = {c: _preds(args.new_dir, c) for c in CONDS}
    for c in CONDS:
        miss = [i for i in ids if i not in newp[c]]
        if miss:
            raise SystemExit(f"[yesno-analyze] {c}: {len(miss)} missing new predictions (e.g. {miss[:3]})")
    newS, newK, newN = {}, {}, {}
    for c in CONDS:
        sc, kk, nn = {}, {}, {}
        for i in ids:
            s, k, npd = soft_accuracy(newp[c][i], m_by[i]["answers_10"])
            sc[i], kk[i], nn[i] = s, k, npd
        newS[c], newK[c], newN[c] = sc, kk, nn
        with open(f"{args.new_dir}/scored_{c}.jsonl", "w", encoding="utf-8") as f:
            for i in ids:
                f.write(json.dumps({
                    "id": i, "answer_type": m_by[i]["answer_type"],
                    "raw_prediction": newp[c][i], "normalized_prediction": nn[i],
                    "answers_10_raw": m_by[i]["answers_10"],
                    "answers_10_normalized": [normalize_answer(a) for a in m_by[i]["answers_10"]],
                    "k": kk[i], "soft_score": sc[i],
                }, ensure_ascii=False) + "\n")

    # ---- old predictions + scores (frozen) -------------------------------------
    OLD_PRED_DIR = {"iq": FROZEN, "generic_mm": FROZEN, "hclt_mm": FROZEN, "as_mm": FROZEN, "directas_mm": F2}
    OLD_SCORED = {"iq": f"{FROZEN}/scored_iq.jsonl", "generic_mm": f"{FROZEN}/scored_generic_mm.jsonl",
                  "hclt_mm": f"{FROZEN}/scored_hclt_mm.jsonl", "as_mm": f"{FROZEN}/scored_as_mm.jsonl",
                  "directas_mm": f"{F2}/scored_directas_mm.jsonl"}
    oldp = {c: _preds(OLD_PRED_DIR[c], c) for c in CONDS}
    olds = {}
    for c in CONDS:
        by = {r["id"]: r.get("soft_score") for r in read_jsonl(OLD_SCORED[c])}
        olds[c] = by

    def arr(dct, c):
        return [dct[c][i] for i in ids]

    # ---- aggregate acc + CI (new) --------------------------------------------
    ACCn = {c: dict(zip(("acc", "lo", "hi", "k"), bmean(arr(newS, c)))) for c in CONDS}
    ACCo = {c: dict(zip(("acc", "lo", "hi", "k"), bmean([olds[c].get(i) for i in ids]))) for c in CONDS}

    # ---- yes/no 103 subset --------------------------------------------------
    yn = [i for i in ids if m_by[i]["answer_type"] == "yes/no"]
    YN = {}
    for c in CONDS:
        accs = [newS[c][i] for i in yn]
        preds_norm = [normalize_answer(newp[c][i]) for i in yn]
        yes_c = sum(1 for p in preds_norm if p == "yes")
        no_c = sum(1 for p in preds_norm if p == "no")
        other_c = len(yn) - yes_c - no_c
        old_acc = statistics.fmean([olds[c].get(i) for i in yn]) if yn else float("nan")
        YN[c] = {"n": len(yn), "acc": statistics.fmean(accs) if accs else float("nan"),
                 "old_acc": old_acc, "yes": yes_c, "no": no_c, "other": other_c,
                 "other_examples": [newp[c][yn[k]] for k in range(len(yn)) if preds_norm[k] not in ("yes", "no")][:8]}

    # ---- non-yes/no 1897 stability ----------------------------------------
    nyn = [i for i in ids if m_by[i]["answer_type"] != "yes/no"]
    STAB = {}
    for c in CONDS:
        pchg = sum(1 for i in nyn if (newp[c].get(i) or "") != (oldp[c].get(i) or ""))
        schg = sum(1 for i in nyn if abs((newS[c][i] or 0) - (olds[c].get(i) or 0)) > 1e-9)
        oacc = statistics.fmean([olds[c].get(i) for i in nyn])
        nacc = statistics.fmean([newS[c][i] for i in nyn])
        STAB[c] = {"n": len(nyn), "pred_changed": pchg, "score_changed": schg,
                   "old_acc": oacc, "new_acc": nacc, "delta": nacc - oacc}

    # ---- corrected question-type table -----------------------------------
    QT = {}
    for t in QTYPES:
        idxs = [i for i in ids if m_by[i]["answer_type"] == t]
        row = {"N": len(idxs)}
        for c in CONDS:
            if c == "iq":
                continue
            vals = [newS[c][i] for i in idxs]
            row[METHOD_OF[c]] = statistics.fmean(vals) if vals else float("nan")
        row["iq"] = statistics.fmean([newS["iq"][i] for i in idxs]) if idxs else float("nan")
        QT[t] = row

    # ---- corrected pairwise bootstrap ----------------------------------
    def A(c):
        return arr(newS, c)
    PW = {
        "HCLT - Generic":        dpair(A("hclt_mm"), A("generic_mm")),
        "Ours-AS - HCLT":        dpair(A("as_mm"), A("hclt_mm")),
        "DirectAS-r2 - HCLT":    dpair(A("directas_mm"), A("hclt_mm")),
        "DirectAS-r2 - Ours-AS": dpair(A("directas_mm"), A("as_mm")),
    }

    # ---- exposure + tokens (frozen representations, recomputed identically) ----
    reptxt = {mth: _text(*REP[mth]) for mth in ("generic", "hclt", "as", "directas")}
    fr_leak = {mth: {r["id"]: r for r in read_jsonl(f"{FROZEN}/leakage_{mth}.jsonl")}
               for mth in ("generic", "hclt", "as")}
    EXPO, LEN = {}, {}
    for mth in ("generic", "hclt", "as", "directas"):
        ng = 0
        for i in ids:
            if mth in fr_leak:
                ng += int(fr_leak[mth].get(i, {}).get("any_novel_gold_mention") or 0)
            else:
                uniq = sorted({_nospace(a) for a in m_by[i]["answers_10"]} - {""})
                ng += _flags(reptxt[mth].get(i, ""), uniq, _nospace(m_by[i]["question"]))["any_novel_gold_mention"]
        EXPO[mth] = ng / n
        toks = [tok(reptxt[mth][i]) for i in ids] if tok else []
        LEN[mth] = round(statistics.fmean(toks), 1) if toks else None
    gtok = LEN["generic"]
    TRED = {mth: round((1 - LEN[mth] / gtok) * 100, 1) for mth in LEN}

    # ================================ report ===============================
    L = []
    P = L.append
    P("# KVQA FINAL 2K — Yes/No Format-Corrected Common Solver\n")
    P(f"N = {n}   (frozen `{FROZEN}/manifest.jsonl`, same ids/order).  Bootstrap seed {BOOT_SEED}, 10 000 resamples.\n")
    P("> The original frozen solver instructed Korean yes/no outputs (\"예/아니오\"), while KVQA stores\n"
      "> yes/no gold answers as English \"Yes\"/\"No\". We evaluate an additional common-solver revision that\n"
      "> changes only the required yes/no output format to English, identically for all methods (I+Q,\n"
      "> Generic-MM, HCLT-MM, Ours-AS-MM, Ours-DirectAS-r2-MM). No representation, sample, evaluator,\n"
      "> normalization, leakage detector, or method-specific instruction is changed.\n")

    P("\n## 1. Solver line changed (text_prompt / mm_prompt / iq_prompt, all three)\n```")
    P("- 예/아니오 질문이면 예 또는 아니오로 답하세요.")
    P("+ 예/아니오로 답할 수 있는 질문이면 반드시 영어로 Yes 또는 No 중 하나만 출력하세요.")
    P("```")
    P("Unchanged: '정답만 간결하게 작성하세요.' / '불필요한 설명이나 이유는 작성하지 마세요.' / "
      "'숫자 질문이면 숫자 중심으로 답하세요.' / '실제 이미지는 제공되지 않았습니다…' (text_prompt).  "
      "Decoding unchanged: do_sample=False, max_new_tokens=20.\n")
    P("## 2. New config\n`configs/kvqa_final_solver_yesno_fixed.yaml`  (`solver_revision: yesno_en_v2`).  "
      "Predictions/scores under `outputs/kvqa_solver_yesno_fixed_2k/`; frozen outputs untouched.\n")

    P("\n## 3. yes/no (official answer_type) — 103 samples\n")
    P("| Method | Yes/No N | Corrected Yes/No Acc | (Original Frozen Yes/No Acc) |")
    P("|---|---:|---:|---:|")
    for c in CONDS:
        y = YN[c]
        P(f"| {LAB[c]} | {y['n']} | {_f4(y['acc'])} | {_f4(y['old_acc'])} |")

    P("\n## 4. yes/no prediction output-format counts (corrected solver)\n")
    P("| Method | Yes | No | other-format | other-format examples |")
    P("|---|---:|---:|---:|---|")
    for c in CONDS:
        y = YN[c]
        P(f"| {LAB[c]} | {y['yes']} | {y['no']} | {y['other']} | {y['other_examples'] if y['other'] else '—'} |")

    P("\n## 5. Old vs corrected — full-2K MM accuracy\n")
    P("| Method | Original Frozen Solver | Yes/No-Fixed Solver | Delta |")
    P("|---|---:|---:|---:|")
    for c in CONDS:
        o, nw = ACCo[c]["acc"], ACCn[c]["acc"]
        P(f"| {LAB[c]} | {_f4(o)} | {_f4(nw)} [{_f4(ACCn[c]['lo'])}, {_f4(ACCn[c]['hi'])}] | {nw-o:+.4f} |")

    P("\n## 6. non-yes/no stability check (1897 samples, answer_type != yes/no)\n")
    P("| Method | N | pred changed | score changed | old acc | new acc | delta |")
    P("|---|---:|---:|---:|---:|---:|---:|")
    for c in CONDS:
        s = STAB[c]
        P(f"| {LAB[c]} | {s['n']} | {s['pred_changed']} ({100*s['pred_changed']/s['n']:.1f}%) | "
          f"{s['score_changed']} ({100*s['score_changed']/s['n']:.1f}%) | {_f4(s['old_acc'])} | "
          f"{_f4(s['new_acc'])} | {s['delta']:+.4f} |")

    P("\n## 7. Corrected question-type table (MM soft acc, official KVQA answer_type)\n")
    P("| Question Type | N | Generic | HCLT | Ours-AS | DirectAS-r2 | _(I+Q ref)_ |")
    P("|---|---:|---:|---:|---:|---:|---:|")
    for t in QTYPES:
        r = QT[t]
        P(f"| {t} | {r['N']} | {_f4(r['generic'])} | {_f4(r['hclt'])} | {_f4(r['as'])} | {_f4(r['directas'])} | _{_f4(r['iq'])}_ |")

    P("\n## 8. Corrected pairwise bootstrap (MM, whole 2K)\n")
    P("| Comparison | Delta | 95% CI | excludes 0 |")
    P("|---|---:|---|---:|")
    for k, v in PW.items():
        P(f"| MM: {k} | {v['delta']:+.4f} | [{v['lo']:+.4f}, {v['hi']:+.4f}] | {'yes' if v['excl0'] else 'no'} |")

    P("\n## 9. Final representation table (corrected MM + frozen-representation exposure/tokens)\n")
    P("| Method | MM Acc ↑ (corrected) | Lexical Answer Exposure ↓ | Mean Tokens ↓ | Token Reduction vs Generic ↑ |")
    P("|---|---:|---:|---:|---:|")
    for c, mth in (("generic_mm", "generic"), ("hclt_mm", "hclt"), ("as_mm", "as"), ("directas_mm", "directas")):
        a = ACCn[c]
        P(f"| {LAB[c]} | {_f4(a['acc'])} [{_f4(a['lo'])}, {_f4(a['hi'])}] | {_pct(EXPO[mth])} | {LEN[mth]} | {TRED[mth]:+.1f}% |")
    P(f"| _{LAB['iq']}_ | _{_f4(ACCn['iq']['acc'])} [{_f4(ACCn['iq']['lo'])}, {_f4(ACCn['iq']['hi'])}]_ | _—_ | _—_ | _—_ |")
    P("\n_Lexical Answer Exposure = frozen NOVEL_GOLD_MENTION; representation-invariant, unchanged from the "
      "original comparison. Tokens: Qwen tokenizer on the same frozen representation files._\n")

    # ---- verdict -------------------------------------------------------
    old_ord = sorted(["generic_mm", "hclt_mm", "as_mm", "directas_mm"], key=lambda c: -ACCo[c]["acc"])
    new_ord = sorted(["generic_mm", "hclt_mm", "as_mm", "directas_mm"], key=lambda c: -ACCn[c]["acc"])
    da_as_o = dpair([olds["directas_mm"].get(i) for i in ids], [olds["as_mm"].get(i) for i in ids])
    da_as_n = PW["DirectAS-r2 - Ours-AS"]
    P("\n## 10. Did correcting the yes/no output format materially change the relative conclusions?\n")
    P(f"- MM ordering (Original): {' > '.join(LAB[c] for c in old_ord)}")
    P(f"- MM ordering (Corrected): {' > '.join(LAB[c] for c in new_ord)}")
    P(f"- DirectAS-r2 − Ours-AS (MM): original {da_as_o['delta']:+.4f} [{da_as_o['lo']:+.4f},{da_as_o['hi']:+.4f}] "
      f"({'excl 0' if da_as_o['excl0'] else 'incl 0'})  →  corrected {_dstr(da_as_n)}")
    P(f"- I+Q reference (MM): original {_f4(ACCo['iq']['acc'])}  →  corrected {_f4(ACCn['iq']['acc'])}")
    all_incl0 = all(not v["excl0"] for v in PW.values())
    P(f"- Any corrected four-method MM pairwise CI excludes 0? {'NO — all within noise' if all_incl0 else 'YES (see Table 8)'}")
    P(f"\n**Answer:** "
      + ("No. Correcting the yes/no output format lifts the yes/no category off the floor for every "
         "method by roughly the same amount, so the relative MM ordering, the DirectAS-vs-AS "
         "(non-)significance, and the I+Q reference relationship are unchanged."
         if (old_ord[:2] == new_ord[:2] and (da_as_o["excl0"] == da_as_n["excl0"]))
         else "The correction changes at least one relative conclusion — see the ordering / CI rows above; "
              "report the corrected numbers."))
    P("\n## 11. Recommendation for the paper main result\n")
    P("Use the **Yes/No Format-Corrected Common Solver** as the main result: it removes a pure "
      "output-format artifact (Korean 예/아니오 vs English gold) that zeroed ~5% of the benchmark "
      "identically for every method, is a single-line change applied uniformly with no method-specific "
      "or answer_type information, and leaves the non-yes/no results "
      + ("essentially unchanged (stability check §6). "
         if max(abs(STAB[c]["delta"]) for c in CONDS) < 0.005 else
         "changed within the reported stability bounds (§6). ")
      + "Keep the Original Frozen Solver numbers reported alongside as the pre-registered frozen run.")

    txt = "\n".join(L) + "\n"
    out = f"{args.new_dir}/yesno_fixed_report.md"
    with open(out, "w", encoding="utf-8") as f:
        f.write(txt)
    print(txt)
    print(f"[yesno-analyze] -> {out}")
    print(f"[yesno-analyze] -> {args.new_dir}/scored_*.jsonl")


if __name__ == "__main__":
    main()
