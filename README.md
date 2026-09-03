# KVQA: Generic vs HCLT-style vs Structured Visual Evidence (Ours-Raw / Ours-AS)

Inference-only comparison of four **intermediate visual-text representations** for Korean VQA,
evaluated **text-only and multimodal** on a fixed 2,000-question KVQA subset with a single
frozen answer solver and the official-derived soft VQA accuracy.

```
Question ─► [Ours] question-only structured decomposition (no image, no gold)
Image, decomposition ─► raw structured visual evidence  ........................  Ours-Raw
Question, raw evidence ─► gold-free answer suppressor (no image/gold/prediction)  Ours-AS

Image ─► generic caption (question-agnostic) ................................... Generic
generic caption ─► sentence split ─► MiniLM(question, sentence) cosine
                 ─► sort by relevance ─► keep Top-3 ........................... HCLT-style
                                                                (High-relevance + High-density)
```

Every representation is answered by the **same** frozen solver, text-only and with the image,
plus an `I+Q` (image+question) baseline. Leakage of the intermediate text is measured with a
frozen lexical detector (NOVEL_GOLD_MENTION / RESULT_PROPOSITION / CANDIDATE_ENUMERATION).

- Results: **[`RESULTS.md`](RESULTS.md)**, `assets/kvqa_final_summary.json`
- Backbone: `Qwen/Qwen3-VL-8B-Instruct`, greedy decoding.
- This is a **HCLT-style adaptation**, not an exact HCLT 2025 reproduction (different backbone,
  unpublished caption prompt, and arrangement+density fused into one pipeline). See `NOTICE.md`.

## 1. Environment

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt          # torch/torchvision per your CUDA — see requirements.txt
```
Tested: Python 3.11, `torch 2.6 cu124`, `transformers 5.16`, 10× RTX A6000 (48 GB).
The MiniLM sentence encoder (`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`) is
loaded through `transformers` (mean pooling + L2 normalize); the `sentence-transformers`
package is **not** required.

## 2. Data (not redistributed — see `NOTICE.md`)

Obtain the official SKT/KVQA release yourself:
- `kvqa_annotations.json` (unified HF annotation, 100,445 records, 10 human answers each)
- the KVQA image archives, extracted so files sit under one directory.

Then point the build step at them:
```bash
export KVQA_ANNOTATIONS=/path/to/kvqa_annotations.json
export KVQA_IMAGE_GLOB='/path/to/kvqa_images/**/*.jpg'   # recursive glob to your extracted images
export KVQA_SPLIT=assets/kvqa_8_1_1_seed2026.json        # bundled: 80/10/remainder, random.Random(2026)
```

The exact 2,000 annotation indices used in the paper are in
`assets/kvqa_final_2k_annotation_indices.json` (integers only). `kvqa_final_build.py`
regenerates the same manifest deterministically from the split file + `random.Random(20260903)`.

## 3. Run

`run.sh <step> [args]` — `PYTHON` and GPU pinning via env.

```bash
# fixed 2K manifest (writes outputs/kvqa_final_hclt_vs_ours_as_2k/manifest.jsonl)
./run.sh build

# evaluator unit test (section 11) — must pass before scoring
./run.sh softtest

# generation (sharded, resumable). One worker per idle GPU:
#   GQA_FORCE_GPU=<id> ./run.sh gen --stage generic,decomp,raw,supp --shard <i>/<N>
./run.sh gen --stage generic,decomp,raw,supp

# HCLT-style High-relevance + High-density Top-3 selection (MiniLM, no VLM, no gold)
./run.sh hclt

# 9 answer conditions: I+Q + {text-only, multimodal} x {Generic, HCLT, Ours-Raw, Ours-AS}
./run.sh solve

# frozen leakage detector on each representation (gold used for diagnostics only)
./run.sh diag

# soft-accuracy scoring, paired bootstrap 95% CI, length analysis, console block + RESULTS.md
./run.sh analyze

# qualitative HTML
./run.sh review --mode sanity30
./run.sh review --mode full
```

Sharding: pass `--shard i/N` and pin `GQA_FORCE_GPU=<physical id>` per worker (helper pattern
in `run.sh`). All stages are per-id resumable; workers write their own shard file and the merge
is deterministic.

## 4. What is frozen

| component | file |
|---|---|
| generic-caption prompt | `configs/gqa_final_hclt_vs_ours_as_2k.yaml : generic_caption.instruction` |
| Ours decomposition prompt (question only) | `… : ours_decomposition.instruction` |
| Ours raw-evidence prompt (image + decomposition) | `… : ours_raw_evidence.instruction` |
| gold-free suppressor prompt | `… : suppressor.adapted_evidence_suppressor_instruction` (+ original recorded) |
| HCLT-style pipeline (embedding model, top-k=3, ordering, tie-break) | `configs/kvqa_final_hclt_vs_ours_as_2k.yaml : hclt_style_combination` |
| common answer solver (text / multimodal / I+Q), max_new_tokens=20 | `configs/kvqa_final_hclt_vs_ours_as_2k.yaml : solver` |
| soft VQA evaluator (BAN-KVQA punctuation normalization, min(match/3,1)) | `src/gqa/kvqa_softscore.py` (verbatim; unit-tested) |
| leakage detector regexes | `src/gqa/main2000_diag.py` (`_cand_enum`, `_result_prop`) |
| 2K sample | `assets/kvqa_8_1_1_seed2026.json` + `random.Random(20260903)` |

Nothing above is changed based on results.

## 5. Outputs

`outputs/kvqa_final_hclt_vs_ours_as_2k/`:
`manifest.jsonl`, `generic_captions.jsonl`, `generic_sentences.jsonl`,
`hclt_high_density.jsonl` (ranked + top-3 + cosine sims), `ours_decompositions.jsonl`,
`ours_raw_evidence.jsonl`, `ours_suppressed_evidence.jsonl`,
`predictions_{iq,generic_text,generic_mm,hclt_text,hclt_mm,raw_text,raw_mm,as_text,as_mm}.jsonl`,
`scored_*.jsonl`, `leakage_{generic,hclt,raw,as}.jsonl`, `summary.json`,
`sanity30_review.html`, `review.html`.
