"""KVQA FINAL — soft-accuracy scoring + analysis + report + console block (section 21).

PRIMARY = KVQA soft VQA accuracy  mean(min(exact-normalized-match-count / 3, 1))  over 10 answers.
Paired bootstrap 95% CI (seed 20240301, 10k) on sample-level soft scores.

  python -m src.gqa.kvqa_final_analyze
"""
from __future__ import annotations

import argparse
import glob
import json
import statistics
from collections import Counter

import numpy as np

from src.common import read_jsonl, resolve
from src.gqa.analyze_phase_b import boot_diff
from src.gqa.kvqa_softscore import soft_accuracy, normalize_answer, selftest

BOOT_SEED = 20240301
CATS = ("yes/no", "number", "other", "unanswerable")
REPRS = ["generic", "hclt", "raw", "as"]
RLAB = {"generic": "Generic", "hclt": "HCLT-style [HighRel + HighDensity]", "raw": "Ours-Raw", "as": "Ours-AS"}
PRED = {"iq": "iq",
        "generic_text": "generic_text", "generic_mm": "generic_mm",
        "hclt_text": "hclt_text", "hclt_mm": "hclt_mm",
        "raw_text": "raw_text", "raw_mm": "raw_mm",
        "as_text": "as_text", "as_mm": "as_mm"}
TEXTC = {"generic": "generic_text", "hclt": "hclt_text", "raw": "raw_text", "as": "as_text"}
MMC = {"generic": "generic_mm", "hclt": "hclt_mm", "raw": "raw_mm", "as": "as_mm"}


def _preds(d, name):
    m = {}
    for fp in sorted(glob.glob(f"{d}/predictions_{name}.jsonl") + glob.glob(f"{d}/predictions_{name}.shard*.jsonl")):
        for r in read_jsonl(fp):
            if "pred" in r:
                m[r["id"]] = r["pred"]
    return m


def bmean(vals, n=10000):
    v = np.array([x for x in vals if x is not None], float)
    if not len(v):
        return float("nan"), float("nan"), float("nan"), 0
    rng = np.random.default_rng(BOOT_SEED)
    m = np.sort(v[rng.integers(0, len(v), size=(n, len(v)))].mean(1))
    return float(v.mean()), float(m[int(.025 * n)]), float(m[int(.975 * n) - 1]), len(v)


def dpair(a, b):
    d = boot_diff(a, b)
    return {"delta": d[0], "lo": d[1], "hi": d[2], "excl0": bool(d[1] > 0 or d[2] < 0)}


def _f(x):
    return "n/a" if x != x else f"{x:.4f}"


def _d(x):
    return f"{x['delta']:+.4f} [{x['lo']:+.4f}, {x['hi']:+.4f}]" + ("  (excludes 0)" if x["excl0"] else "  (includes 0)")


def _tok_counter():
    try:
        from transformers import AutoTokenizer
        tk = AutoTokenizer.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")
        return lambda s: len(tk(s, add_special_tokens=False)["input_ids"])
    except Exception as e:
        print(f"[kvqa-analyze] tokenizer unavailable ({e!r}); token counts = n/a")
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="outputs/kvqa_final_hclt_vs_ours_as_2k/manifest.jsonl")
    ap.add_argument("--dir", default="outputs/kvqa_final_hclt_vs_ours_as_2k")
    ap.add_argument("--summary", default="outputs/kvqa_final_hclt_vs_ours_as_2k/summary.json")
    ap.add_argument("--report", default="reports/kvqa_final_hclt_vs_ours_as_2k.md")
    args = ap.parse_args()
    selftest()                                             # section 11 gate
    d = str(resolve(args.dir))
    man = read_jsonl(resolve(args.manifest))
    n = len(man)

    reptxt = {}
    for name, (pat, idk, tk) in {
        "generic": ("generic_captions*.jsonl", "id", "generic_caption"),
        "hclt": ("hclt_high_density.jsonl", "id", "hclt_high_density_caption"),
        "raw": ("ours_raw_evidence*.jsonl", "id", "raw_evidence"),
        "as": ("ours_suppressed_evidence*.jsonl", "id", "suppressed_evidence"),
    }.items():
        mm = {}
        for fp in sorted(glob.glob(f"{d}/{pat}")):
            for r in read_jsonl(fp):
                if r.get(tk):
                    mm[r["id"]] = r[tk]
        reptxt[name] = mm
    sent = {r["id"]: r.get("sentences", []) for r in read_jsonl(f"{d}/generic_sentences.jsonl")} \
        if glob.glob(f"{d}/generic_sentences.jsonl") else {}
    hcl_sel = {r["id"]: r.get("selected_sentences", []) for r in read_jsonl(f"{d}/hclt_high_density.jsonl")} \
        if glob.glob(f"{d}/hclt_high_density.jsonl") else {}
    leak = {name: {r["id"]: r for r in read_jsonl(f"{d}/leakage_{name}.jsonl")} for name in REPRS}
    preds = {c: _preds(d, PRED[c]) for c in PRED}

    # ---------- per-row soft scoring for every condition ----------
    rows = []
    for m in man:
        sid = m["id"]
        row = {"id": sid, "question": m["question"], "answer_type": m["answer_type"],
               "answers_10": m["answers_10"]}
        for c in PRED:
            p = preds[c].get(sid)
            if p is None:
                row[f"score_{c}"] = None
                row[f"pred_{c}"] = None
            else:
                s, k, npd = soft_accuracy(p, m["answers_10"])
                row[f"score_{c}"] = s
                row[f"pred_{c}"] = p
                row[f"k_{c}"] = k
                row[f"npred_{c}"] = npd
        rows.append(row)

    # scored_*.jsonl (section 19)
    for c in PRED:
        with open(f"{d}/scored_{c}.jsonl", "w", encoding="utf-8") as f:
            for m, r in zip(man, rows):
                p = r[f"pred_{c}"]
                rec = {"id": r["id"], "answer_type": r["answer_type"],
                       "raw_prediction": p,
                       "normalized_prediction": r.get(f"npred_{c}"),
                       "answers_10_raw": m["answers_10"],
                       "answers_10_normalized": [normalize_answer(a) for a in m["answers_10"]],
                       "k": r.get(f"k_{c}"), "soft_score": r[f"score_{c}"]}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def acc(c):
        return bmean([r[f"score_{c}"] for r in rows])

    def rate(name, key):
        v = [leak[name].get(r["id"], {}).get(key) for r in rows]
        v = [x for x in v if x is not None]
        return (sum(v) / len(v)) if v else float("nan")

    # ---------- main table ----------
    tok = _tok_counter()

    def length_stats(name):
        ids = [r["id"] for r in rows if r["id"] in reptxt[name]]
        texts = [reptxt[name][i] for i in ids]
        if name == "generic":
            sc = [len(sent.get(i, [])) for i in ids]
        elif name == "hclt":
            sc = [len(hcl_sel.get(i, [])) for i in ids]
        else:
            sc = [max(1, t.count(".") + t.count("!") + t.count("?") + t.count("\n")) for t in texts]
        chars = [len(t) for t in texts]
        toks = [tok(t) for t in texts] if tok else []
        return {
            "n": len(texts),
            "mean_sentences": round(statistics.fmean(sc), 2) if sc else None,
            "median_sentences": statistics.median(sc) if sc else None,
            "mean_chars": round(statistics.fmean(chars), 1) if chars else None,
            "mean_tokens": round(statistics.fmean(toks), 1) if toks else None,
            "p25_tokens": (round(np.percentile(toks, 25), 1) if toks else None),
            "p75_tokens": (round(np.percentile(toks, 75), 1) if toks else None),
        }

    table = {}
    for name in REPRS:
        ta, tlo, thi, _ = acc(TEXTC[name])
        ma, mlo, mhi, _ = acc(MMC[name])
        table[name] = {
            "text_acc": ta, "text_ci": [tlo, thi],
            "mm_acc": ma, "mm_ci": [mlo, mhi],
            "image_gain": ma - ta,
            "novel_gold": rate(name, "any_novel_gold_mention"),
            "novel_gold_match_count": rate(name, "novel_gold_match_count"),
            "result_prop": rate(name, "result_proposition"),
            "cand_enum": rate(name, "candidate_enumeration"),
            "length": length_stats(name),
        }
    iq_acc = acc("iq")

    # ---------- comparisons + paired CIs (section 14 & 15) ----------
    def ad(cx, cy):     # score_cx - score_cy paired
        return dpair([r[f"score_{cx}"] for r in rows], [r[f"score_{cy}"] for r in rows])

    def ld(a, b, key):  # leakage prop diff a - b paired
        return dpair([leak[a].get(r["id"], {}).get(key) for r in rows],
                     [leak[b].get(r["id"], {}).get(key) for r in rows])

    cmp = {
        "Generic->HCLT": {
            "text": ad("hclt_text", "generic_text"), "mm": ad("hclt_mm", "generic_mm"),
            "novel_gold": ld("hclt", "generic", "any_novel_gold_mention"),
            "result_prop": ld("hclt", "generic", "result_proposition"),
            "cand_enum": ld("hclt", "generic", "candidate_enumeration")},
        "HCLT->Ours-Raw": {
            "text": ad("raw_text", "hclt_text"), "mm": ad("raw_mm", "hclt_mm"),
            "novel_gold": ld("raw", "hclt", "any_novel_gold_mention"),
            "result_prop": ld("raw", "hclt", "result_proposition"),
            "cand_enum": ld("raw", "hclt", "candidate_enumeration")},
        "Raw->AS": {
            "text": ad("as_text", "raw_text"), "mm": ad("as_mm", "raw_mm"),
            "novel_gold": ld("as", "raw", "any_novel_gold_mention"),
            "result_prop": ld("as", "raw", "result_proposition"),
            "cand_enum": ld("as", "raw", "candidate_enumeration")},
        "HCLT->Ours-AS": {
            "text": ad("as_text", "hclt_text"), "mm": ad("as_mm", "hclt_mm"),
            "novel_gold": ld("as", "hclt", "any_novel_gold_mention"),
            "result_prop": ld("as", "hclt", "result_proposition"),
            "cand_enum": ld("as", "hclt", "candidate_enumeration")},
    }
    raw_gain = ad("raw_mm", "raw_text")
    as_gain = ad("as_mm", "as_text")
    gain_diff = dpair([r["score_as_mm"] - r["score_as_text"] for r in rows if r["score_as_mm"] is not None and r["score_as_text"] is not None],
                      [r["score_raw_mm"] - r["score_raw_text"] for r in rows if r["score_raw_mm"] is not None and r["score_raw_text"] is not None])

    # ---------- per-category (secondary) ----------
    bycat = {}
    for cat in CATS:
        cr = [r for r in rows if r["answer_type"] == cat]
        bycat[cat] = {"n": len(cr),
                      **{c: bmean([r[f"score_{c}"] for r in cr])[0] for c in
                         ("iq", "generic_text", "generic_mm", "hclt_text", "hclt_mm",
                          "raw_text", "raw_mm", "as_text", "as_mm")}}

    # ---------- verdict (section 22) — evidence-bounded, no over-claim ----------
    def word(dd):
        if not dd["excl0"]:
            return "not statistically significant"
        return "significantly higher" if dd["delta"] > 0 else "significantly lower"
    A_text = cmp["HCLT->Ours-Raw"]["text"]
    A_mm = cmp["HCLT->Ours-Raw"]["mm"]
    A = ("SUPPORTED" if (A_text["excl0"] and A_text["delta"] > 0 and A_mm["delta"] > 0)
         else "PARTIAL" if (A_text["delta"] > 0 or A_mm["delta"] > 0) else "NOT SUPPORTED")
    B_ng = cmp["Raw->AS"]["novel_gold"]
    B_util = cmp["Raw->AS"]["text"]
    B = (f"novel-gold {word(B_ng)} ({_d(B_ng)}); text-only utility cost {_d(B_util)}; "
         f"mm utility {_d(cmp['Raw->AS']['mm'])}")
    C_text = cmp["HCLT->Ours-AS"]["text"]
    C_mm = cmp["HCLT->Ours-AS"]["mm"]
    C_ng = cmp["HCLT->Ours-AS"]["novel_gold"]
    C = (f"Ours-AS vs HCLT: text {word(C_text)} ({_d(C_text)}), mm {word(C_mm)} ({_d(C_mm)}), "
         f"novel-gold {word(C_ng)} ({_d(C_ng)})")

    S = {"n": n, "primary_metric": "KVQA soft VQA accuracy (min(match/3,1) over 10 answers)",
         "iq_mm_acc": {"acc": iq_acc[0], "ci": [iq_acc[1], iq_acc[2]]},
         "main_table": table, "by_category": bycat,
         "comparisons": cmp,
         "image_dependence": {"raw_gain": raw_gain, "as_gain": as_gain, "as_gain_minus_raw_gain": gain_diff},
         "verdict": {"A_structured_acquisition": A, "B_suppression_effect": B, "C_final_method": C}}
    json.dump(S, open(resolve(args.summary), "w"), ensure_ascii=False, indent=2)

    # representative / failure examples (gold used only for this post-hoc grouping)
    def lk(name, sid, key):
        return leak[name].get(sid, {}).get(key)
    reps = {
        "hclt_leaks_as_clean_both_ok": [r["id"] for r in rows
            if lk("hclt", r["id"], "any_novel_gold_mention") and not lk("as", r["id"], "any_novel_gold_mention")
            and (r["score_hclt_text"] or 0) >= 2/3 and (r["score_as_text"] or 0) >= 2/3][:8],
        "as_text_much_worse_than_raw_text": [r["id"] for r in rows
            if r["score_raw_text"] is not None and r["score_as_text"] is not None
            and r["score_raw_text"] - r["score_as_text"] >= 2/3][:8],
        "as_mm_recovers_from_as_text": [r["id"] for r in rows
            if r["score_as_text"] is not None and r["score_as_mm"] is not None
            and r["score_as_mm"] - r["score_as_text"] >= 2/3][:8],
        "as_still_flagged": [r["id"] for r in rows
            if lk("as", r["id"], "any_novel_gold_mention") or lk("as", r["id"], "result_proposition")
            or lk("as", r["id"], "candidate_enumeration")][:8],
        "raw_beats_hclt_text": [r["id"] for r in rows
            if r["score_raw_text"] is not None and r["score_hclt_text"] is not None
            and r["score_raw_text"] - r["score_hclt_text"] >= 2/3][:8],
    }
    json.dump(reps, open(resolve(f"{args.dir}/representative_examples.json"), "w"), ensure_ascii=False, indent=2)

    # ---------- markdown report ----------
    L = []
    L.append("# KVQA — FINAL: Generic vs HCLT-style (High-relevance + High-density Top-3) vs "
             "Ours-Raw vs Ours-AS  (Text-only + Multimodal)\n")
    L.append(f"Fixed KVQA-2K evaluation subset (repo seed-2026 test split, kvqa-source, resolvable "
             f"image; **not an exact HCLT 2025 reproduction**). N={n}. Backbone Qwen3-VL-8B-Instruct, "
             "greedy. One common answer solver (no answer_type hint) for every method/condition. "
             "PRIMARY = KVQA soft VQA accuracy over 10 human answers "
             "(SKTBrain/BAN-KVQA normalization, verbatim). Paired bootstrap 95% CI, seed 20240301.\n")

    L.append("## Main table\n")
    L.append("| Method | Text-only Acc | Multimodal Acc | Image Gain | Novel Gold | Result Prop | Cand Enum | Mean Tokens |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    L.append(f"| I+Q | — | {_f(iq_acc[0])} | — | — | — | — | — |")
    for name in REPRS:
        t = table[name]
        L.append(f"| {RLAB[name]} | {_f(t['text_acc'])} | {_f(t['mm_acc'])} | {t['image_gain']:+.4f} | "
                 f"{_f(t['novel_gold'])} | {_f(t['result_prop'])} | {_f(t['cand_enum'])} | "
                 f"{t['length']['mean_tokens'] if t['length']['mean_tokens'] is not None else 'n/a'} |")
    L.append("\nImage Gain = Multimodal Acc − Text-only Acc.\n")

    L.append("## Representation length (section 16)\n")
    L.append("| repr | mean sent | median sent | mean chars | mean tok | p25 tok | p75 tok |")
    L.append("|---|---:|---:|---:|---:|---:|---:|")
    for name in REPRS:
        x = table[name]["length"]
        L.append(f"| {RLAB[name]} | {x['mean_sentences']} | {x['median_sentences']} | {x['mean_chars']} | "
                 f"{x['mean_tokens']} | {x['p25_tokens']} | {x['p75_tokens']} |")
    L.append("")

    L.append("## Key comparisons (paired bootstrap 95% CI)\n")
    for label, blk in cmp.items():
        L.append(f"**{label}**")
        L.append(f"- Text-only: {_d(blk['text'])}")
        L.append(f"- Multimodal: {_d(blk['mm'])}")
        L.append(f"- Novel Gold: {_d(blk['novel_gold'])}")
        L.append(f"- Result Prop: {_d(blk['result_prop'])}   ·   Candidate Enum: {_d(blk['cand_enum'])}\n")
    L.append("**Image dependence**")
    L.append(f"- Raw image gain (Raw_MM − Raw_Text): {_d(raw_gain)}")
    L.append(f"- AS image gain (AS_MM − AS_Text): {_d(as_gain)}")
    L.append(f"- (AS gain − Raw gain): {_d(gain_diff)}\n")

    L.append("## Per-category soft accuracy (secondary)\n")
    L.append("| answer_type | N | I+Q(mm) | Gen-T | Gen-MM | HCLT-T | HCLT-MM | Raw-T | Raw-MM | AS-T | AS-MM |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for cat in CATS:
        x = bycat[cat]
        L.append(f"| {cat} | {x['n']} | {_f(x['iq'])} | {_f(x['generic_text'])} | {_f(x['generic_mm'])} | "
                 f"{_f(x['hclt_text'])} | {_f(x['hclt_mm'])} | {_f(x['raw_text'])} | {_f(x['raw_mm'])} | "
                 f"{_f(x['as_text'])} | {_f(x['as_mm'])} |")
    L.append("")

    L.append("## Verdict (section 22)\n")
    L.append(f"- **A. Structured acquisition (Ours-Raw vs HCLT-style):** {A}  "
             f"(text {_d(A_text)}; mm {_d(A_mm)})")
    L.append(f"- **B. Suppression effect (Ours-Raw → Ours-AS):** {B}")
    L.append(f"- **C. Final method (Ours-AS vs HCLT-style):** {C}")
    L.append("")
    L.append("## Limitations\n")
    L.append("- `NOVEL_GOLD_MENTION` is a lexical gold-surface proxy over the 10 human answers; it can "
             "miss semantic synonym / paraphrase leakage and can over-fire on short number/color surfaces.\n"
             "- HCLT-style here fuses arrangement (high-relevance) and density (Top-3) into one adapted "
             "pipeline on Qwen3-VL-8B — not an exact HCLT 2025 reproduction (different backbone, "
             "unpublished caption prompt, 6–10-sentence generic target).\n"
             "- `unanswerable` questions are kept; the common solver has no explicit unanswerable rule.\n"
             "- Accuracy CIs that include 0 are reported as 'not statistically significant', never as "
             "'better'.\n")

    open(resolve(args.report), "w", encoding="utf-8").write("\n".join(L) + "\n")

    # ---------- console block (section 21) ----------
    def blk(name):
        t = table[name]
        return (f"{RLAB[name]}{' [FINAL]' if name == 'as' else ''}\n"
                f"- Text-only Acc: {t['text_acc']:.4f}\n- Multimodal Acc: {t['mm_acc']:.4f}\n"
                f"- Image Gain: {t['image_gain']:+.4f}\n- Novel Gold: {t['novel_gold']:.4f}\n"
                f"- Result Prop: {t['result_prop']:.4f}\n- Candidate Enum: {t['cand_enum']:.4f}\n"
                f"- Mean Tokens: {t['length']['mean_tokens']}")
    print("\nKVQA FINAL 2K\n\nN = " + str(n) + "\n")
    print(f"I+Q\n- Multimodal Acc: {iq_acc[0]:.4f}\n")
    for name in REPRS:
        print(blk(name) + "\n")
    print("=" * 40 + "\nKEY DELTAS\n" + "=" * 40)
    for label in ("Generic->HCLT", "HCLT->Ours-Raw", "Raw->AS", "HCLT->Ours-AS"):
        b = cmp[label]
        print(f"\n{label}\nText-only: {_d(b['text'])}\nMultimodal: {_d(b['mm'])}\nNovel Gold: {_d(b['novel_gold'])}")
    print("\nIMAGE DEPENDENCE")
    print(f"Raw image gain  = {_d(raw_gain)}")
    print(f"AS image gain   = {_d(as_gain)}")
    print(f"AS gain - Raw gain = {_d(gain_diff)}")
    print(f"\nVERDICT\nA structured acquisition: {A}\nB suppression: {B}\nC final: {C}")
    print(f"\n[kvqa-analyze] wrote {args.summary} + {args.report}")


if __name__ == "__main__":
    main()
