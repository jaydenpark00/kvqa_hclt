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
