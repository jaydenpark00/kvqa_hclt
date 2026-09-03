# Results — Ours-DirectAS-r2, Yes/No-corrected solver, full split

Companion to `RESULTS.md`. Same backbone (`Qwen/Qwen3-VL-8B-Instruct`, greedy), same frozen
soft VQA evaluator and `NOVEL_GOLD_MENTION` lexical detector. **Lexical Answer Exposure** below
is exactly that frozen metric (≥1 normalized human-answer surface present in the representation
and not already in the question) — a lexical surface proxy, not a semantic leakage measure.

Aggregate numbers only (KVQA data is not redistributed; see `NOTICE.md`).

---

## 1. 500-sample pilot (Ours-DirectAS-r2 vs Ours-Raw / Ours-AS)

Subset: first 500 rows of the frozen 2K manifest, order preserved, no resampling
(`sha256 907c39e1754c89c9577c00d58d4eb534004a29dda133f2232b169b2438f7c24f`). Original frozen
(Korean yes/no) solver. Paired bootstrap: 10 000 resamples, seed 20240301.

| Representation | Text Acc | MM Acc | NOVEL_GOLD_MENTION ↓ | mean tokens |
|---|---:|---:|---:|---:|
| Ours-Raw | 0.2387 | 0.2953 | 20.0% | 72.2 |
| Ours-AS | 0.0953 | 0.2987 | 3.3% | 63.4 |
| **Ours-DirectAS-r2** | 0.0847 | 0.2987 | **0.0%** | 127.0 |
| _I+Q (reference)_ | _—_ | _0.3093_ | _—_ | _—_ |

Paired deltas (MM): DirectAS−Raw `+0.0033 [−0.0200, +0.0273]` (incl 0); DirectAS−AS
`+0.0000 [−0.0240, +0.0240]` (incl 0). Text-only: DirectAS−Raw `−0.1540 [−0.1900, −0.1193]`
(excl 0) — the representation-alone answer shortcut is removed, as intended. Image gain
(MM−Text): Raw `+0.0567`, AS `+0.2033`, DirectAS `+0.2140`; DirectAS gain − Raw gain
`+0.1573 [+0.1207, +0.1953]` (excl 0).

---

## 2. Full-2K four-method comparison (Generic / HCLT / Ours-AS / Ours-DirectAS-r2)

N = 2000, frozen manifest, original frozen solver (so `yes/no` is uninformative here — see §3).
Generic / HCLT / Ours-AS reuse the frozen full-2K outputs; DirectAS-r2 generated from
`configs/kvqa_ours_directas.yaml` revision r2 on the same manifest.

| Method | MM Acc ↑ | Lexical Answer Exposure ↓ | Mean Tokens ↓ | Token Reduction vs Generic ↑ |
|---|---:|---:|---:|---:|
| Generic Caption | 0.3238 [0.3043, 0.3432] | 55.4% | 244.9 | +0.0% |
| HCLT-style | 0.3293 [0.3098, 0.3488] | 37.4% | 76.1 | +68.9% |
| Ours-AS | 0.3245 [0.3053, 0.3442] | 24.3% | 65.1 | +73.4% |
| **Ours-DirectAS-r2** | 0.3297 [0.3102, 0.3492] | **17.4%** | 130.0 | +46.9% |
| _I+Q (reference)_ | _0.3420 [0.3220, 0.3613]_ | _—_ | _—_ | _—_ |

Paired bootstrap (MM): every four-method pairwise CI **includes 0** — HCLT−Generic
`+0.0055 [−0.0042, +0.0153]`, Ours-AS−HCLT `−0.0048 [−0.0168, +0.0073]`, DirectAS−HCLT
`+0.0003 [−0.0118, +0.0123]`, DirectAS−Ours-AS `+0.0052 [−0.0055, +0.0162]`. Multimodal QA
utility is statistically indistinguishable across the four representations.

Exposure (paired, all **exclude 0**): DirectAS−HCLT `−0.1995 [−0.2225, −0.1765]`,
DirectAS−Ours-AS `−0.0695 [−0.0885, −0.0500]`, Ours-AS−HCLT `−0.1300 [−0.1540, −0.1065]`.
Lexical answer exposure falls monotonically Generic → HCLT → Ours-AS → DirectAS-r2, each step
significant.

Secondary (frozen `main2000_diag` defs; DirectAS computed identically): CandEnum
Generic 7.2% / HCLT 2.0% / AS 1.9% / DirectAS 0.5%; ResultProp ≈ 0 for all. Auxiliary
heuristics (not the frozen metric): a Korean state/yes-no-conclusion flag is higher for
DirectAS (2.1%) than AS (0.8%) because its visual descriptions are longer; OCR-literal
heuristic lowest for DirectAS (1.6%).

By official `answer_type`: `number` all high (0.73–0.77, no significant method gap);
`other` (n=1427) all ≈ 0.34–0.35; `unanswerable` DirectAS-r2 modestly but significantly above
AS and HCLT (`+0.013–0.015`, CI excludes 0); `yes/no` = 0 for every method under this solver.

---

## 3. Yes/No output-format correction (`solver_revision: yesno_en_v2`)

The `yes/no` category (555 of 9424 full; 103 of the 2K) scored 0 for **every** method because
the frozen solver instructs Korean `예/아니오` while KVQA gold is English `Yes`/`No`. One rule
line changed in `text_prompt` / `mm_prompt` / `iq_prompt` (English `Yes`/`No`), applied
identically to all five conditions; nothing else touched.

`yes/no` subset of the 2K (N = 103), corrected solver:

| Condition | old (ko) | corrected (en) | Yes preds | No preds | other-format |
|---|---:|---:|---:|---:|---:|
| I+Q (reference) | 0.0000 | 0.9094 | 51 | 52 | 0 |
| Generic-MM | 0.0000 | 0.8932 | 49 | 54 | 0 |
| HCLT-MM | 0.0000 | 0.8738 | 47 | 56 | 0 |
| Ours-AS-MM | 0.0000 | 0.9126 | 52 | 51 | 0 |
| Ours-DirectAS-r2-MM | 0.0000 | 0.8608 | 54 | 49 | 0 |

All 515 predictions are exactly `Yes` or `No` (0 malformed). yes/no soft score ≈ exact-match
rate here (the 10 human answers are unanimous). The artifact is uniform, so it does not bias
the representation comparison; the corrected solver is the appropriate main result and the
original frozen-solver numbers remain the pre-registered run.

---

## 4. Full eligible test split (N = 9424)

`outputs/kvqa_final_full/manifest.jsonl` — all `test_annotation_indices` of the seed-2026
split that are kvqa-source with a resolvable image, 10 answers and an official `answer_type`
(only exclusion: 621 vizwiz-source). `sha256
b1be43f5a60cfbdf83e7d481787cf6a2ff2cc351ae5a6317ca5399c5fa204dec`;
answer_type: other 6574 / unanswerable 1377 / number 918 / yes/no 555; 2000 ⊂ 9424.

The five-condition comparison on this split with the corrected solver
(`final_full_comparison.md`, `*.csv`, `full_summary.json`) is produced by
`kvqa_final_full_analyze.py`. **This run was still in progress at the time of writing** —
numbers to be filled in from `outputs/kvqa_final_full/final_full_comparison.md`.
