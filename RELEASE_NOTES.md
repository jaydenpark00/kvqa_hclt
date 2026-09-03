# Release notes — relation to the exact run

The scripts under `src/` and the configs under `configs/` are the **exact code and frozen
prompts used to produce `RESULTS.md`**, with two mechanical portability edits so the package
runs outside the original cluster:

1. `src/gqa/kvqa_final_build.py` — the three data-path constants (`ANN`, `SPLIT`, `IMG_GLOB`)
   now read from env vars `KVQA_ANNOTATIONS` / `KVQA_SPLIT` / `KVQA_IMAGE_GLOB`, defaulting to
   the original relative paths (`KVQA_SPLIT` default points at the bundled
   `assets/kvqa_8_1_1_seed2026.json`). Selection logic, seed (`random.Random(20260903)`),
   dedup, and ordering are unchanged.
2. `src/gqa/kvqa_softscore.py` — the optional evaluator cross-check path is now `HCLT_REPO`
   (env), empty by default. The soft-accuracy and normalization functions themselves are
   untouched and remain a verbatim copy (see `NOTICE.md`).

`scripts/run_gqa_kvqa_final.sh` is the original cluster runner (hard-coded interpreter path);
prefer the portable `run.sh` at the repo root.

Everything else — the four generation prompts, the HCLT-style pipeline parameters, the common
answer solver, the leakage-detector regexes, `max_new_tokens`, greedy decoding, and the
evaluator — is byte-for-byte as run and is not changed based on results.

## Known limitations carried from the run (see `RESULTS.md` for detail)

- KVQA `yes/no` gold answers are the English strings `Yes` / `No`, while the frozen
  answer_type-agnostic solver instructs Korean `예` / `아니오`; all `yes/no` questions
  (~5% of the 2K) score 0 for **every** method including `I+Q`. Uniform, so it does not bias
  the representation comparison; not fixed because the solver/evaluator are frozen.
- `NOVEL_GOLD_MENTION` is a lexical gold-surface proxy over the 10 human answers.
- Text-only VQA over these representations is largely an answer-copying task; suppression
  removes the copyable answer (text-only collapses) while the multimodal path recovers.

---

## Revision — Ours-DirectAS-r2, Yes/No-corrected solver, full split

Additive only. The frozen 2K prompts, common solver, evaluator, `NOVEL_GOLD_MENTION` detector,
split file and the `outputs/kvqa_final_hclt_vs_ours_as_2k/` results are **unchanged and not
regenerated**. New code and configs:

- `configs/kvqa_ours_directas.yaml` — `protected_decomposition` (9-key) + `protected_evidence`
  prompts, frozen at `revision: r2` (one rule-level re-freeze after a 30-sample wiring check;
  the `changelog:` field records the exact two category-rule edits). No per-sample tuning.
- `configs/kvqa_final_solver_yesno_fixed.yaml` — the frozen `solver` block with **one** rule
  line changed to English `Yes`/`No` in `text_prompt` / `mm_prompt` / `iq_prompt`
  (`solver_revision: yesno_en_v2`); `max_new_tokens`, decoding and every other line identical.
- `src/gqa/kvqa_directas_gen.py`, `kvqa_directas_sanity.py`, `kvqa_directas_solve.py`,
  `kvqa_directas_pilot_build.py`, `kvqa_directas_pilot_analyze.py`,
  `kvqa_directas_final_compare.py` — DirectAS generation, 30-sample sanity, solving, 500-pilot
  and 2K four-method comparison. These reuse `kvqa_final_solve.fill/_shard/_load`,
  `kvqa_final_analyze.bmean/dpair/_tok_counter`, `kvqa_softscore`, and the frozen
  `main2000_diag` regexes verbatim.
- `src/gqa/kvqa_yesno_fixed_solve.py`, `kvqa_yesno_fixed_analyze.py` — re-solve the five MM
  conditions with the corrected solver and report the yes/no subset + non-yes/no stability.
- `src/gqa/kvqa_full_build.py` — `kvqa_final_build.py` with the 2000-subsample removed; the
  three data-path constants read `KVQA_ANNOTATIONS` / `KVQA_SPLIT` / `KVQA_IMAGE_GLOB` (same
  portability edit already applied to `kvqa_final_build.py`), default `KVQA_SPLIT` →
  `assets/kvqa_8_1_1_seed2026.json`. Eligibility / order / schema are otherwise identical.
- `src/gqa/kvqa_final_full_analyze.py` — merge + score + full comparison on the 9424 split.
- `run.sh` — new steps `directas-*`, `yesno-*`, `build-full`, `full-analyze`, `*-fanout`.

Auxiliary diagnostics introduced for DirectAS review (a Korean state/yes-no-conclusion regex
and an OCR-literal-transcription heuristic) are **not** frozen metrics and are reported
separately from `NOVEL_GOLD_MENTION` / `RESULT_PROPOSITION` / `CANDIDATE_ENUMERATION`.

Results: `RESULTS_DIRECTAS.md`.
