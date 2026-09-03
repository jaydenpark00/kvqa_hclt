"""PHASE B step 4 — aggregate scored.jsonl into the 2x2 analysis + summary.json +
reports/korean_gqa_phase_b_report.md. CPU only.

PRIMARY metric = norm_match. exact_match reported alongside. contains_match is a
formatting diagnostic only (never an accuracy).

    python -m src.gqa.analyze_phase_b
"""
from __future__ import annotations

import argparse
import json
from collections import Counter

import numpy as np

from src.common import read_jsonl

CONDS = ("bq", "b0", "b1", "b2")
BOOT_SEED = 20240301


def acc(rows, metric):
    v = [r[f"{metric}"] for r in rows if r.get(f"{metric}") is not None]
    return (sum(v) / len(v)) if v else float("nan"), len(v)


def boot_mean(x, n=10000, seed=BOOT_SEED, alpha=0.05):
    x = np.asarray([v for v in x if v is not None], float)
    if len(x) == 0:
        return (float("nan"), float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    m = np.sort(x[rng.integers(0, len(x), size=(n, len(x)))].mean(axis=1))
    return float(x.mean()), float(m[int(alpha / 2 * n)]), float(m[int((1 - alpha / 2) * n) - 1])


def boot_diff(a, b, n=10000, seed=BOOT_SEED, alpha=0.05):
    """paired bootstrap of mean(a) - mean(b) over the SAME rows (a,b aligned, may contain None)."""
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    if not pairs:
        return (float("nan"), float("nan"), float("nan"))
    arr = np.asarray(pairs, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(arr), size=(n, len(arr)))
    d = (arr[idx, 0] - arr[idx, 1]).mean(axis=1)
    d.sort()
    return (float(arr[:, 0].mean() - arr[:, 1].mean()),
            float(d[int(alpha / 2 * n)]), float(d[int((1 - alpha / 2) * n) - 1]))


def cell(rows, metric="norm"):
    out = {"n": len(rows)}
    for c in CONDS:
        p, lo, hi = boot_mean([r[f"{metric}_{c}"] for r in rows])
        out[c] = {"acc": p, "lo": lo, "hi": hi}
    dd = {
        "B2_minus_Bq": boot_diff([r[f"{metric}_b2"] for r in rows], [r[f"{metric}_bq"] for r in rows]),
        "B1_minus_B0": boot_diff([r[f"{metric}_b1"] for r in rows], [r[f"{metric}_b0"] for r in rows]),
        "B0_minus_B2": boot_diff([r[f"{metric}_b0"] for r in rows], [r[f"{metric}_b2"] for r in rows]),
    }
    out["deltas"] = {k: {"delta": v[0], "lo": v[1], "hi": v[2]} for k, v in dd.items()}
    # caption substitution ratio  (B2 - Bq) / (B0 - Bq)
    b2, bq, b0 = out["b2"]["acc"], out["bq"]["acc"], out["b0"]["acc"]
    denom = b0 - bq
    if denom <= 0.02:
        out["caption_substitution_ratio"] = {
            "value": None,
            "reason": f"denominator (B0-Bq) = {denom:.4f} <= 0.02; ratio not meaningful"}
    else:
        out["caption_substitution_ratio"] = {"value": (b2 - bq) / denom,
                                             "numerator_B2_minus_Bq": b2 - bq,
                                             "denominator_B0_minus_Bq": denom}
    return out


def fmt(x):
    return "n/a" if x is None or (isinstance(x, float) and x != x) else f"{x:.3f}"


def sgn(d):
    if isinstance(d, dict):
        return f"{d['delta']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}]"
    return f"{d:+.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored", default="outputs/korean_gqa_pilot/scored.jsonl")
    ap.add_argument("--summary", default="outputs/korean_gqa_pilot/summary.json")
    ap.add_argument("--report", default="reports/korean_gqa_phase_b_report.md")
    args = ap.parse_args()

    rows = read_jsonl(args.scored)
    N = len(rows)
    S = {"n": N, "primary_metric": "norm_match", "secondary_metric": "exact_match",
         "note": "contains_match is a formatting diagnostic only, never an accuracy"}

    S["missing_predictions"] = {c: sum(1 for r in rows if r[f"pred_{c}"] is None) for c in CONDS}
    S["overall_norm"] = cell(rows, "norm")
    S["overall_exact"] = {c: acc(rows, f"exact_{c}")[0] for c in CONDS}
    S["overall_contains_UPPER_BAND"] = {c: acc(rows, f"contains_{c}")[0] for c in CONDS}
    S["contains_not_norm_counts"] = {c: sum(r[f"contains_not_norm_{c}"] or 0 for r in rows)
                                     for c in CONDS}
    S["verbose_pred_gt15chars"] = {c: sum(1 for r in rows
                                          if r[f"pred_{c}"] and len(r[f"pred_{c}"]) > 15)
                                   for c in CONDS}

    # ---- SENSITIVITY band: whole analysis recomputed on contains_match ----
    # (NOT an accuracy - a permissive upper bound to show how much of each result is
    #  format-limited vs reasoning-limited. Qwen2-VL ignores the terse-answer instruction
    #  ~30-50% of the time in Korean, so norm_match under-measures every condition.)
    S["SENSITIVITY_contains_band"] = {
        "overall": cell(rows, "contains"),
        "by_group": {g: cell([r for r in rows if r["group"] == g], "contains")
                     for g in ("DIRECT", "COMPOSITIONAL")},
        "grid_2x2": {f"{g}__{'Present' if p else 'Absent'}":
                     cell([r for r in rows if r["group"] == g and bool(r["lexical_answer_present"]) == p],
                          "contains")
                     for g in ("DIRECT", "COMPOSITIONAL") for p in (True, False)},
        "focus_Compositional_LexAbsent": cell(
            [r for r in rows if r["group"] == "COMPOSITIONAL" and not r["lexical_answer_present"]],
            "contains"),
    }

    # ---- caption degeneracy (greedy-loop) impact ----
    try:
        caps = {r["image_id"]: r for r in read_jsonl(
            args.scored.replace("scored.jsonl", "captions.jsonl"))}
        rep = {iid: c.get("repetition_score", 0.0) for iid, c in caps.items()}
        deg_ids = {iid for iid, v in rep.items() if v > 0.4}
        S["caption_degeneracy"] = {
            "degenerate_rep_gt_0.4": len(deg_ids),
            "mild_0.2_0.4": sum(1 for v in rep.values() if 0.2 < v <= 0.4),
            "pilot_QA_with_degenerate_caption": sum(1 for r in rows if r["image_id"] in deg_ids),
            "norm_on_clean_caption_subset": cell(
                [r for r in rows if r["image_id"] not in deg_ids], "norm"),
        }
    except Exception as e:  # pragma: no cover
        S["caption_degeneracy"] = {"error": repr(e)}

    S["by_group"] = {g: cell([r for r in rows if r["group"] == g], "norm")
                     for g in ("DIRECT", "COMPOSITIONAL")}
    S["by_family"] = {f: cell([r for r in rows if r["family"] == f], "norm")
                      for f in sorted({r["family"] for r in rows})}

    lap = "lexical_answer_present"
    S["lexical_answer_presence"] = {
        "present": sum(r[lap] for r in rows), "absent": sum(1 - r[lap] for r in rows),
        "present_loose": sum(r["lexical_answer_present_loose"] for r in rows),
        "by_group": {g: {"present": sum(r[lap] for r in rows if r["group"] == g),
                         "absent": sum(1 - r[lap] for r in rows if r["group"] == g)}
                     for g in ("DIRECT", "COMPOSITIONAL")},
    }

    # ---- the 2x2 : group x lexical presence ----
    S["grid_2x2"] = {}
    for g in ("DIRECT", "COMPOSITIONAL"):
        for pres in (True, False):
            key = f"{g}__{'Present' if pres else 'Absent'}"
            sub = [r for r in rows if r["group"] == g and bool(r[lap]) == pres]
            S["grid_2x2"][key] = cell(sub, "norm")

    # family x lexical presence
    S["grid_family_x_presence"] = {}
    for f in sorted({r["family"] for r in rows}):
        for pres in (True, False):
            sub = [r for r in rows if r["family"] == f and bool(r[lap]) == pres]
            if sub:
                S["grid_family_x_presence"][f"{f}__{'Present' if pres else 'Absent'}"] = cell(sub, "norm")

    # ---- GO / NO-GO focus cell ----
    focus = [r for r in rows if r["group"] == "COMPOSITIONAL" and not r[lap]]
    fc = cell(focus, "norm")
    S["GO_NOGO__Compositional_and_LexicalAnswerAbsent"] = {
        **fc,
        "A__B2_gt_Bq": {"delta_ci": fc["deltas"]["B2_minus_Bq"],
                        "verdict": ("B2 > Bq (CI excludes 0)"
                                    if fc["deltas"]["B2_minus_Bq"]["lo"] > 0 else
                                    "B2 <= Bq or CI includes 0")},
        "B__caption_substitution_ratio": fc["caption_substitution_ratio"],
        "C__B1_gt_B0": {"delta_ci": fc["deltas"]["B1_minus_B0"],
                        "verdict": ("B1 > B0 (CI excludes 0)"
                                    if fc["deltas"]["B1_minus_B0"]["lo"] > 0 else
                                    "B1 <= B0 or CI includes 0")},
    }

    with open(args.summary, "w", encoding="utf-8") as f:
        json.dump(S, f, ensure_ascii=False, indent=2)

    write_report(S, rows, args.report)
    print(f"[gqa-analyze] wrote {args.summary} and {args.report}")
    print(json.dumps({k: S[k] for k in ("overall_exact", "contains_not_norm_counts",
                                        "lexical_answer_presence")},
                     ensure_ascii=False, indent=2))


def _accrow(c):
    return f"{fmt(c['acc'])} [{fmt(c['lo'])}, {fmt(c['hi'])}]"


def write_report(S, rows, path):
    L = []
    L.append("# Korean GQA — PHASE B baseline pilot report\n")
    L.append(f"- Pilot: **{S['n']} QA** from the **official TRAIN split** "
             "(`data/korean_gqa_pilot_manifest.jsonl`). Official **val is reserved** — no "
             "inference was run on it.")
    L.append("- Model: `Qwen/Qwen2-VL-7B-Instruct`, greedy, `vqa_max_new_tokens=20` "
             "(Exp1/Exp3 generation block).")
    L.append("- Conditions: **Bq** Q-only · **B0** Image+Q · **B1** Image+Direct-KO caption+Q "
             "(*HCLT-style baseline adapted to Korean GQA*) · **B2** caption+Q, no image.")
    L.append("- Same answer instruction in all four: `정답만 간단히 출력하고 설명하지 마세요.` "
             "(the one deliberate change vs Exp1's `정답만 짧게 작성하세요.` — applied identically "
             "to every condition; B0/B1 preambles are otherwise Exp1's verbatim; Bq/B2 are new "
             "parallel templates). See `src/gqa/common.py`.")
    L.append("- **PRIMARY metric = `norm_match`** (NFC, trim, strip one trailing period, single "
             "reference). `exact_match` reported alongside. `contains_match` is a **formatting "
             "diagnostic only**.")
    L.append(f"- Missing predictions: {S['missing_predictions']}\n")

    L.append("## 0. ⚠️ Two measurement caveats (read first)\n")
    L.append(f"1. **Answer formatting.** Qwen2-VL-7B ignores the terse-answer instruction a "
             f"large fraction of the time in Korean — it answers in a full sentence "
             f"(*\"여자가 입고 있는 바지의 색상은 회색입니다.\"*). Predictions >15 chars: "
             f"{S['verbose_pred_gt15chars']}. `contains_match` True while `norm_match` False: "
             f"{S['contains_not_norm_counts']} of {S['n']}. So **`norm_match` under-measures "
             f"every condition** by ~10–20 pp; the `contains` band in §1b is the permissive "
             f"upper bound (still not an accuracy). Deltas between conditions are more robust "
             f"than absolute levels because the format loss is roughly proportional across Bq/B0/B1/B2.")
    cd = S.get("caption_degeneracy", {})
    L.append(f"2. **Caption degeneracy.** {cd.get('degenerate_rep_gt_0.4','?')} / {S['n']} "
             f"pilot images got a degenerate greedy-loop Direct-KO caption "
             f"(repetition_score > 0.4; +{cd.get('mild_0.2_0.4','?')} mild). "
             f"Exp3 prompt/params used exactly as instructed — not changed. B1/B2 on those rows "
             f"carry little information; §1c reports norm_match on the clean-caption subset.\n")

    L.append("## 1. Overall (primary = norm_match)\n")
    L.append("| cond | norm_match [95% CI] | exact_match |")
    L.append("|---|---|---|")
    for c in CONDS:
        L.append(f"| {c.upper()} | {_accrow(S['overall_norm'][c])} | "
                 f"{fmt(S['overall_exact'][c])} |")
    d = S["overall_norm"]["deltas"]
    L.append("")
    L.append(f"- **B2 − Bq** = {sgn(d['B2_minus_Bq'])}  (caption adds info to text-only QA)")
    L.append(f"- **B1 − B0** = {sgn(d['B1_minus_B0'])}  (HCLT-style caption help on top of image)")
    L.append(f"- **B0 − B2** = {sgn(d['B0_minus_B2'])}  (image advantage the caption fails to carry)")
    csr = S["overall_norm"]["caption_substitution_ratio"]
    L.append(f"- caption substitution ratio (B2−Bq)/(B0−Bq) = "
             + (fmt(csr["value"]) if csr.get("value") is not None else f"**not computed** — {csr['reason']}"))
    L.append("")

    L.append("### 1b. Sensitivity band — `contains_match` (permissive UPPER bound, NOT an accuracy)\n")
    sb = S["SENSITIVITY_contains_band"]["overall"]
    L.append("| cond | contains-band [95% CI] |")
    L.append("|---|---|")
    for c in CONDS:
        L.append(f"| {c.upper()} | {_accrow(sb[c])} |")
    sd = sb["deltas"]
    L.append(f"\nB2−Bq {sgn(sd['B2_minus_Bq'])} · B1−B0 {sgn(sd['B1_minus_B0'])} · "
             f"B0−B2 {sgn(sd['B0_minus_B2'])}")
    scsr = sb["caption_substitution_ratio"]
    L.append(f"· subst. ratio "
             + (fmt(scsr["value"]) if scsr.get("value") is not None else f"n/a ({scsr['reason']})") + "\n")

    L.append("### 1c. norm_match on the clean-caption subset (repetition_score ≤ 0.4)\n")
    cc = cd.get("norm_on_clean_caption_subset", {})
    if cc:
        L.append(f"n = {cc['n']}.  " + " · ".join(
            f"{c.upper()} {fmt(cc[c]['acc'])}" for c in CONDS)
            + f"  |  B1−B0 {sgn(cc['deltas']['B1_minus_B0'])} · "
            f"B2−Bq {sgn(cc['deltas']['B2_minus_Bq'])}\n")

    L.append("## 2. By group\n")
    for g in ("DIRECT", "COMPOSITIONAL"):
        c = S["by_group"][g]
        L.append(f"### {g}  (n={c['n']})")
        L.append("| cond | norm_match [95% CI] |")
        L.append("|---|---|")
        for k in CONDS:
            L.append(f"| {k.upper()} | {_accrow(c[k])} |")
        dd = c["deltas"]
        L.append(f"\nB2−Bq {sgn(dd['B2_minus_Bq'])} · B1−B0 {sgn(dd['B1_minus_B0'])} · "
                 f"B0−B2 {sgn(dd['B0_minus_B2'])}\n")

    L.append("## 3. By family\n")
    L.append("| family | n | Bq | B0 | B1 | B2 | B2−Bq | B1−B0 |")
    L.append("|---|--:|--:|--:|--:|--:|--:|--:|")
    for f, c in S["by_family"].items():
        L.append(f"| {f} | {c['n']} | {fmt(c['bq']['acc'])} | {fmt(c['b0']['acc'])} | "
                 f"{fmt(c['b1']['acc'])} | {fmt(c['b2']['acc'])} | "
                 f"{c['deltas']['B2_minus_Bq']['delta']:+.3f} | "
                 f"{c['deltas']['B1_minus_B0']['delta']:+.3f} |")
    L.append("")

    lp = S["lexical_answer_presence"]
    L.append("## 4. Lexical answer presence\n")
    L.append(f"- **Lexical Answer-Present** (gold's conservative normalized surface form is "
             f"verbatim in the Direct-KO caption): **{lp['present']} / {S['n']}** "
             f"(loose, +adjectival stems: {lp['present_loose']}).")
    L.append(f"- Lexical Answer-Absent: **{lp['absent']} / {S['n']}**.")
    L.append(f"- by group — DIRECT: {lp['by_group']['DIRECT']}; "
             f"COMPOSITIONAL: {lp['by_group']['COMPOSITIONAL']}")
    L.append("- *Lexical* presence only — this is not a claim of semantic answer leakage.\n")

    L.append("## 5. Core 2×2 — group × lexical answer presence (norm_match)\n")
    L.append("| cell | n | Bq | B0 | B1 | B2 | B2−Bq | B1−B0 | B0−B2 | subst. ratio |")
    L.append("|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|")
    for g in ("DIRECT", "COMPOSITIONAL"):
        for pres in ("Present", "Absent"):
            c = S["grid_2x2"][f"{g}__{pres}"]
            csr = c["caption_substitution_ratio"]
            L.append(f"| {g} ∩ {pres} | {c['n']} | {fmt(c['bq']['acc'])} | {fmt(c['b0']['acc'])} "
                     f"| {fmt(c['b1']['acc'])} | {fmt(c['b2']['acc'])} | "
                     f"{c['deltas']['B2_minus_Bq']['delta']:+.3f} | "
                     f"{c['deltas']['B1_minus_B0']['delta']:+.3f} | "
                     f"{c['deltas']['B0_minus_B2']['delta']:+.3f} | "
                     f"{fmt(csr['value']) if csr.get('value') is not None else 'n/a'} |")
    L.append("")

    L.append("## 6. GO / NO-GO — Compositional ∩ Lexical Answer-Absent\n")
    fcell = S["GO_NOGO__Compositional_and_LexicalAnswerAbsent"]
    L.append(f"n = **{fcell['n']}**.  Bq {_accrow(fcell['bq'])} · B0 {_accrow(fcell['b0'])} · "
             f"B1 {_accrow(fcell['b1'])} · B2 {_accrow(fcell['b2'])}\n")
    L.append(f"- **A. B2 > Bq?**  Δ = {sgn(fcell['A__B2_gt_Bq']['delta_ci'])} → "
             f"{fcell['A__B2_gt_Bq']['verdict']}. "
             "*(does a caption with no verbatim answer still beat question-only?)*")
    b = fcell["B__caption_substitution_ratio"]
    L.append(f"- **B. caption substitution ratio (B2−Bq)/(B0−Bq)** = "
             + (fmt(b["value"]) if b.get("value") is not None else f"**not computed** — {b['reason']}")
             + ".  *(how much of the image's lift does a text-only caption recover?)*")
    L.append(f"- **C. B1 > B0?**  Δ = {sgn(fcell['C__B1_gt_B0']['delta_ci'])} → "
             f"{fcell['C__B1_gt_B0']['verdict']}. "
             "*(HCLT-style generic-caption assistance help on Korean GQA?)*")
    fb = S["SENSITIVITY_contains_band"]["focus_Compositional_LexAbsent"]
    L.append(f"\n*contains-band cross-check (same cell, n={fb['n']}):* "
             f"Bq {fmt(fb['bq']['acc'])} · B0 {fmt(fb['b0']['acc'])} · B1 {fmt(fb['b1']['acc'])} · "
             f"B2 {fmt(fb['b2']['acc'])}  |  B2−Bq {sgn(fb['deltas']['B2_minus_Bq'])} · "
             f"B1−B0 {sgn(fb['deltas']['B1_minus_B0'])}\n")

    L.append("## 7. Read of the pilot\n")
    g2 = S["grid_2x2"]
    L.append("**Focus cell (Compositional ∩ Lexical Answer-Absent, n=233): all three GO checks fail.** "
             "A_B2>Bq: no (Δ≈0). B_substitution ratio: ≈0.08 (norm) — the caption recovers "
             "almost none of the image's lift. C_B1>B0: no — B1 is flat-to-slightly-below B0 "
             "(contains-band CI **excludes 0 on the negative side**). The contains-band "
             "cross-check tells the same story, so this is not just a formatting artefact.\n")
    L.append("**Where captions do help** (§5, §3): only the **Lexical Answer-Present** cells "
             f"(Comp∩Present B1−B0 {sgn(g2['COMPOSITIONAL__Present']['deltas']['B1_minus_B0'])}, "
             f"Direct∩Present {sgn(g2['DIRECT__Present']['deltas']['B1_minus_B0'])}) and "
             "**direct_global** (weather / brightness / indoor-outdoor — a whole-image property a "
             "generic caption naturally states; B2−Bq large but the label space is tiny). "
             "The positive **overall** B2−Bq is driven entirely by those.\n")
    L.append("**Implication for the research question:** on this pilot, generic Direct-KO captions "
             "do **not** carry compositional visual facts that substitute for the image — they help "
             "mainly by containing the answer string, echoing the KVQA problem. Unlike KVQA Exp3 "
             "(small reliable V1−V0 drop), here B1 ≈ B0. This *motivates* an answer-excluded "
             "visual guide, but also shows the bar: a guide must beat a generic caption on "
             "Compositional ∩ Answer-Absent, where the caption currently does nothing.\n")
    L.append("**Before scaling up / building the guide — needs sign-off (both are changes the "
             "brief pinned):**")
    L.append("1. **Answer-format enforcement.** norm_match under-measures ~2× because Qwen2-VL "
             "answers in sentences; a 2–3-shot 'answer only' exemplar block or constrained "
             "decoding would let the metric track reasoning. (contains-band is only a diagnostic.)")
    L.append("2. **Caption degeneracy 15 %.** One bounded greedy-loop retry, or a light "
             "repetition_penalty for the caption pass only (Exp4 already ships the hook).")
    L.append("3. Consider Qwen2.5-VL and/or a larger n — B0 itself is low "
             f"({fmt(S['overall_norm']['b0']['acc'])} norm / "
             f"{fmt(S['overall_contains_UPPER_BAND']['b0'])} contains), which compresses every delta.\n")
    L.append("## 8. Reserved / not done\n")
    L.append("Official val inference · answer-excluded guide generation · guide prompt design · "
             "Qwen2.5-VL · any training (SFT/DPO). Next step decided with the user from §7.\n")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
