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

---

## 6. Extensions in this revision

Three additive changes. **Nothing above is modified**: the frozen 2K prompts, solver,
evaluator, leakage detector and split are byte-for-byte unchanged, and the frozen outputs
are not regenerated.

### 6a. Ours-DirectAS-r2 — acquisition-time answer protection

Instead of generating raw answer-bearing evidence and then removing the answer with a Stage-3
suppressor (Ours-AS), DirectAS folds suppression into the acquisition step:

```
Question ─► 9-key PROTECTED decomposition   (question only; no image, no gold)
            = frozen 6 keys + observation_target + protected_answer_value + evidence_to_preserve
Image, protected decomposition ─► directly answer-suppressed visual guidance ......... Ours-DirectAS-r2
                                  (NO Stage-3 suppressor)
```

`protected_answer_value` records the *meaning* of the answer variable, never a value
(`"액정에 표시된 숫자 문자열"`, not `"3:00"`); Stage 2 must not emit it, while keeping
localization / reference object / region / shape cues. Prompts are frozen at revision **r2**
(one rule-level re-freeze after a 30-sample wiring check; see the config `revision:` field and
its `changelog`). No per-sample tuning.

```
configs/kvqa_ours_directas.yaml          # protected_decomposition + protected_evidence (r2)
src/gqa/kvqa_directas_gen.py             # 2 stages: pdecomp (text) -> pevidence (image)
src/gqa/kvqa_directas_sanity.py          # 30-sample diagnostic HTML/JSONL (fixed sanity ids)
src/gqa/kvqa_directas_solve.py           # frozen common solver applied to DirectAS evidence
src/gqa/kvqa_directas_pilot_build.py     # deterministic 500-sample pilot subset (first 500)
src/gqa/kvqa_directas_pilot_analyze.py   # pilot scoring + paired bootstrap + leakage + length
src/gqa/kvqa_directas_final_compare.py   # 2K four-method comparison (MM utility / exposure / tokens / q-type)
```

```bash
GQA_FORCE_GPU=<id> ./run.sh directas-gen --stage pdecomp,pevidence --shard <i>/<N>
./run.sh directas-sanity
./run.sh directas-pilot-build && ./run.sh directas-fanout <N> --manifest <pilot500> --outdir <pilotdir> && ./run.sh directas-pilot-analyze
./run.sh directas-compare
```

### 6b. Yes/No output-format-corrected common solver

KVQA stores `yes/no` gold as English `Yes` / `No`; the frozen solver instructs Korean
`예` / `아니오`, so every `yes/no` question scored 0 for **all** methods (a pure output-format
artifact, uniform, non-biasing). `configs/kvqa_final_solver_yesno_fixed.yaml`
(`solver_revision: yesno_en_v2`) changes exactly one rule line in `text_prompt` / `mm_prompt` /
`iq_prompt`:

```
- 예/아니오 질문이면 예 또는 아니오로 답하세요.
+ 예/아니오로 답할 수 있는 질문이면 반드시 영어로 Yes 또는 No 중 하나만 출력하세요.
```

Everything else (other rules, greedy decoding, `max_new_tokens=20`, no answer_type hint, no
method-specific prompt) is unchanged, applied identically to all five conditions. The original
frozen-solver numbers remain the pre-registered run.

```
src/gqa/kvqa_yesno_fixed_solve.py     # re-solve I+Q + {Generic,HCLT,Ours-AS,DirectAS-r2}-MM
src/gqa/kvqa_yesno_fixed_analyze.py   # yes/no subset table + non-yes/no stability + corrected tables
```

```bash
./run.sh yesno-solve <N> --config configs/kvqa_final_solver_yesno_fixed.yaml \
         --repr-dir <reprdir> --directas-dir <dadir> --outdir <outdir>
./run.sh yesno-analyze
```

### 6c. Full eligible test split (N = 9424)

`src/gqa/kvqa_full_build.py` is `kvqa_final_build.py` with the `random.Random(20260903)
.sample(resolvable, 2000)` subsample removed — same source (`test_annotation_indices`),
eligibility (kvqa-source, resolvable image, 10 answers, official `answer_type`) and ordering
(annotation index ascending). Path constants read `KVQA_ANNOTATIONS` / `KVQA_SPLIT` /
`KVQA_IMAGE_GLOB` like the 2K build. `src/gqa/kvqa_final_full_analyze.py` merges shards, scores
the five MM conditions with the corrected solver, and emits the full comparison
(`final_full_comparison.md` + `*.csv` + `full_summary.json`). The 2000 frozen-subset
representations are reused verbatim for the overlapping ids (identical prompts, deterministic
greedy); the remaining 7424 use the same pipelines.

```bash
./run.sh build-full
GQA_FORCE_GPU=<id> ./run.sh full-fanout src.gqa.kvqa_final_gen <N> \
   --manifest outputs/kvqa_final_full/manifest.jsonl --outdir outputs/kvqa_final_full --stage decomp,raw,supp
# ... likewise kvqa_directas_gen (pdecomp,pevidence) and kvqa_final_gen (generic); then hclt; then:
./run.sh yesno-solve <N> --manifest outputs/kvqa_final_full/manifest.jsonl \
   --repr-dir outputs/kvqa_final_full --directas-dir outputs/kvqa_final_full --outdir outputs/kvqa_final_full
./run.sh full-analyze
```

Results for 6a–6c: **[`RESULTS_DIRECTAS.md`](RESULTS_DIRECTAS.md)**.
