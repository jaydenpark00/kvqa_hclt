# KVQA — FINAL: Generic vs HCLT-style (High-relevance + High-density Top-3) vs Ours-Raw vs Ours-AS  (Text-only + Multimodal)

Fixed KVQA-2K evaluation subset (repo seed-2026 test split, kvqa-source, resolvable image; **not an exact HCLT 2025 reproduction**). N=2000. Backbone Qwen3-VL-8B-Instruct, greedy. One common answer solver (no answer_type hint) for every method/condition. PRIMARY = KVQA soft VQA accuracy over 10 human answers (SKTBrain/BAN-KVQA normalization, verbatim). Paired bootstrap 95% CI, seed 20240301.

## Main table

| Method | Text-only Acc | Multimodal Acc | Image Gain | Novel Gold | Result Prop | Cand Enum | Mean Tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| I+Q | — | 0.3420 | — | — | — | — | — |
| Generic | 0.2727 | 0.3238 | +0.0512 | 0.5540 | 0.0030 | 0.0720 | 244.9 |
| HCLT-style [HighRel + HighDensity] | 0.1997 | 0.3293 | +0.1297 | 0.3735 | 0.0010 | 0.0200 | 76.1 |
| Ours-Raw | 0.2713 | 0.3268 | +0.0555 | 0.5310 | 0.0005 | 0.0445 | 75.3 |
| Ours-AS | 0.0940 | 0.3245 | +0.2305 | 0.2435 | 0.0000 | 0.0195 | 65.1 |

Image Gain = Multimodal Acc − Text-only Acc.

## Representation length (section 16)

| repr | mean sent | median sent | mean chars | mean tok | p25 tok | p75 tok |
|---|---:|---:|---:|---:|---:|---:|
| Generic | 9.32 | 9.0 | 369.8 | 244.9 | 207.0 | 298.0 |
| HCLT-style [HighRel + HighDensity] | 3.0 | 3.0 | 111.6 | 76.1 | 64.0 | 86.0 |
| Ours-Raw | 2.39 | 2.0 | 109.7 | 75.3 | 60.0 | 88.0 |
| Ours-AS | 2.49 | 2.0 | 95.0 | 65.1 | 50.8 | 78.0 |

## Key comparisons (paired bootstrap 95% CI)

**Generic->HCLT**
- Text-only: -0.0730 [-0.0878, -0.0582]  (excludes 0)
- Multimodal: +0.0055 [-0.0042, +0.0153]  (includes 0)
- Novel Gold: -0.1805 [-0.1970, -0.1640]  (excludes 0)
- Result Prop: -0.0020 [-0.0040, -0.0005]  (excludes 0)   ·   Candidate Enum: -0.0520 [-0.0620, -0.0425]  (excludes 0)

**HCLT->Ours-Raw**
- Text-only: +0.0717 [+0.0545, +0.0885]  (excludes 0)
- Multimodal: -0.0025 [-0.0153, +0.0102]  (includes 0)
- Novel Gold: +0.1575 [+0.1375, +0.1780]  (excludes 0)
- Result Prop: -0.0005 [-0.0025, +0.0010]  (includes 0)   ·   Candidate Enum: +0.0245 [+0.0140, +0.0355]  (excludes 0)

**Raw->AS**
- Text-only: -0.1773 [-0.1948, -0.1598]  (excludes 0)
- Multimodal: -0.0023 [-0.0117, +0.0067]  (includes 0)
- Novel Gold: -0.2875 [-0.3075, -0.2675]  (excludes 0)
- Result Prop: -0.0005 [-0.0015, +0.0000]  (includes 0)   ·   Candidate Enum: -0.0250 [-0.0325, -0.0175]  (excludes 0)

**HCLT->Ours-AS**
- Text-only: -0.1057 [-0.1237, -0.0880]  (excludes 0)
- Multimodal: -0.0048 [-0.0168, +0.0073]  (includes 0)
- Novel Gold: -0.1300 [-0.1540, -0.1065]  (excludes 0)
- Result Prop: -0.0010 [-0.0025, +0.0000]  (includes 0)   ·   Candidate Enum: -0.0005 [-0.0090, +0.0080]  (includes 0)

**Image dependence**
- Raw image gain (Raw_MM − Raw_Text): +0.0555 [+0.0443, +0.0672]  (excludes 0)
- AS image gain (AS_MM − AS_Text): +0.2305 [+0.2120, +0.2488]  (excludes 0)
- (AS gain − Raw gain): +0.1750 [+0.1567, +0.1930]  (excludes 0)

## Per-category soft accuracy (secondary)

| answer_type | N | I+Q(mm) | Gen-T | Gen-MM | HCLT-T | HCLT-MM | Raw-T | Raw-MM | AS-T | AS-MM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| yes/no | 103 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| number | 198 | 0.7744 | 0.6364 | 0.7290 | 0.5168 | 0.7458 | 0.6717 | 0.7525 | 0.1515 | 0.7593 |
| other | 1427 | 0.3639 | 0.2899 | 0.3455 | 0.2049 | 0.3504 | 0.2840 | 0.3450 | 0.1093 | 0.3415 |
| unanswerable | 272 | 0.0417 | 0.0208 | 0.0380 | 0.0172 | 0.0404 | 0.0159 | 0.0453 | 0.0074 | 0.0417 |

## Verdict (section 22)

- **A. Structured acquisition (Ours-Raw vs HCLT-style):** PARTIAL  (text +0.0717 [+0.0545, +0.0885]  (excludes 0); mm -0.0025 [-0.0153, +0.0102]  (includes 0))
- **B. Suppression effect (Ours-Raw → Ours-AS):** novel-gold significantly lower (-0.2875 [-0.3075, -0.2675]  (excludes 0)); text-only utility cost -0.1773 [-0.1948, -0.1598]  (excludes 0); mm utility -0.0023 [-0.0117, +0.0067]  (includes 0)
- **C. Final method (Ours-AS vs HCLT-style):** Ours-AS vs HCLT: text significantly lower (-0.1057 [-0.1237, -0.0880]  (excludes 0)), mm not statistically significant (-0.0048 [-0.0168, +0.0073]  (includes 0)), novel-gold significantly lower (-0.1300 [-0.1540, -0.1065]  (excludes 0))

## Limitations

- `NOVEL_GOLD_MENTION` is a lexical gold-surface proxy over the 10 human answers; it can miss semantic synonym / paraphrase leakage and can over-fire on short number/color surfaces.
- HCLT-style here fuses arrangement (high-relevance) and density (Top-3) into one adapted pipeline on Qwen3-VL-8B — not an exact HCLT 2025 reproduction (different backbone, unpublished caption prompt, 6–10-sentence generic target).
- `unanswerable` questions are kept; the common solver has no explicit unanswerable rule.
- Accuracy CIs that include 0 are reported as 'not statistically significant', never as 'better'.
- **yes/no format mismatch:** KVQA yes/no gold answers are the English strings `Yes`/`No`, but the frozen answer_type-agnostic common solver instructs Korean `예`/`아니오`, so **all 103 yes/no questions score 0 for every method including I+Q**. This depresses absolute accuracy uniformly (~5% of the set) and does NOT bias the representation comparison. Not fixed: the solver/evaluator are frozen (section 18); the answer_type-aware solver that would emit `Yes`/`No` is disallowed (section 8/15).
- **text-only VQA on these representations is largely answer-copying:** worked examples (e.g. gold `KURZWEIL`, `3:00`) show Generic/Ours-Raw place the literal answer in the text so text-only succeeds trivially; Ours-AS abstracts it away ("…확인한다") so text-only has nothing to copy and collapses, while Ours-AS *multimodal* reads the image and recovers fully.

