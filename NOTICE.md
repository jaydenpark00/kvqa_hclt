# Data, model, and prompt provenance

## KVQA data — NOT redistributed here

This repository contains **no KVQA questions, answers, images, or the sampled manifest**.
KVQA (SKT) is released under the **Korean VQA License**: research/education only, no commercial
use, and no third-party redistribution/sharing or modification without SKT's prior consent.
The VizWiz subset additionally carries CC BY 4.0. Obtain the data from the official sources
(`SKTBrain/KVQA`, `skt/KVQA` on Hugging Face) and follow their license.

What is included, and why it is safe to share:
- `assets/kvqa_8_1_1_seed2026.json` — an 80/10/remainder split expressed **only as integer
  annotation indices** + the seed (`random.Random(2026)`). No KVQA content.
- `assets/kvqa_final_2k_annotation_indices.json` — the 2,000 integer annotation indices used in
  the paper. No KVQA content.
- `assets/kvqa_final_sampling_stats.json`, `assets/kvqa_final_summary.json`,
  `assets/kvqa_final_representative_examples.json`, `RESULTS.md` — aggregate statistics /
  example IDs only.

The manifest with questions and the 10 gold answers is produced locally by `./run.sh build`
from your own KVQA download and is **git-ignored**.

## Evaluator

`src/gqa/kvqa_softscore.py` `normalize_answer` / `soft_accuracy` are a **verbatim copy** of the
KVQA soft-accuracy implementation used by the authors' KVQA structured-caption code, which cites
`SKTBrain/BAN-KVQA` `tools/compute_softscore.py` for the punctuation/whitespace normalization
(no semantic Korean rewrites). The unit test (`./run.sh softtest`) verifies the k→score table
and, if `HCLT_REPO` points at that repo, cross-checks 200 random cases against it.

## Prompts

The `Ours` decomposition and raw-evidence prompts
(`configs/gqa_final_hclt_vs_ours_as_2k.yaml : ours_decomposition / ours_raw_evidence`) are
byte-identical to `revised_decomposition_prompt()` / `evidence_preservation_caption_prompt()`
from the authors' earlier KVQA structured-caption repository. The gold-free suppressor prompt is
that repository-independent `suppressor_instruction` (from the Korean-GQA answer-suppression
config) with a mechanical *step → sentence* wording adaptation; both the original and adapted
texts are stored in `configs/gqa_final_hclt_vs_ours_as_2k.yaml : suppressor`.

The generic-caption prompt is a **paper-faithful approximation** — HCLT 2025's caption-generation
prompt is unpublished. The "6–10 short sentences" target is an implementation choice so that
Top-3 relevance selection is non-degenerate. This is a **HCLT-style adaptation**, not an exact
HCLT reproduction: different backbone (Qwen3-VL-8B vs Qwen2-VL-7B) and arrangement (high-
relevance) + density (Top-3) are fused into a single pipeline.

## Model

`Qwen/Qwen3-VL-8B-Instruct` — obtain from Hugging Face under its own license. Not included.

## Code license

The code in `src/` and `configs/` (excluding the noted verbatim/derived prompt and evaluator
text) is released under the MIT License — see `LICENSE`.
