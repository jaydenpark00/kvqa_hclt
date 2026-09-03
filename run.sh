#!/usr/bin/env bash
# Portable entrypoint for the KVQA representation comparison.
#   PYTHON     : python interpreter (default: python)
#   GQA_FORCE_GPU : physical GPU id to pin for GPU steps (gen / hclt / solve / directas / yesno)
#   KVQA_ANNOTATIONS / KVQA_IMAGE_GLOB / KVQA_SPLIT : data locations (see README / NOTICE)
#   HCLT_REPO  : optional path to the KVQA structured-caption repo for evaluator cross-check
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
PY="${PYTHON:-python}"

# fan a sharded module over GPUs 0..N-1
_fanout () { local mod="$1" n="$2"; shift 2
  local pids=(); for ((i=0;i<n;i++)); do
    GQA_FORCE_GPU=$i "$PY" -m "$mod" --shard "$i/$n" "$@" & pids+=($!)
  done
  local rc=0; for p in "${pids[@]}"; do wait "$p" || rc=1; done; return $rc ; }

STEP="${1:-}"; shift || true
case "$STEP" in
  # --- frozen 2K representation comparison (Generic / HCLT / Ours-Raw / Ours-AS) -----
  build)    exec "$PY" -m src.gqa.kvqa_final_build "$@" ;;
  softtest) exec "$PY" -m src.gqa.kvqa_softscore "$@" ;;
  gen)      exec "$PY" -m src.gqa.kvqa_final_gen "$@" ;;
  hclt)     exec "$PY" -m src.gqa.kvqa_final_hclt "$@" ;;
  solve)    exec "$PY" -m src.gqa.kvqa_final_solve "$@" ;;
  diag)     exec "$PY" -m src.gqa.kvqa_final_diag "$@" ;;
  analyze)  exec "$PY" -m src.gqa.kvqa_final_analyze "$@" ;;
  review)   exec "$PY" -m src.gqa.kvqa_final_review "$@" ;;
  fanout)   s="$1"; n="$2"; shift 2; _fanout "src.gqa.kvqa_final_${s}" "$n" "$@"; exit $? ;;

  # --- Ours-DirectAS-r2 : acquisition-time answer protection, no Stage-3 suppressor --
  directas-gen)      exec "$PY" -m src.gqa.kvqa_directas_gen "$@" ;;          # --stage pdecomp,pevidence
  directas-sanity)   exec "$PY" -m src.gqa.kvqa_directas_sanity "$@" ;;       # 30-sample wiring check
  directas-solve)    exec "$PY" -m src.gqa.kvqa_directas_solve "$@" ;;
  directas-pilot-build)   exec "$PY" -m src.gqa.kvqa_directas_pilot_build "$@" ;;
  directas-pilot-analyze) exec "$PY" -m src.gqa.kvqa_directas_pilot_analyze "$@" ;;
  directas-compare)  exec "$PY" -m src.gqa.kvqa_directas_final_compare "$@" ;;
  directas-fanout)   n="$1"; shift; _fanout src.gqa.kvqa_directas_gen "$n" "$@"; exit $? ;;

  # --- Yes/No output-format-corrected common solver (English Yes/No; one-line change) -
  yesno-solve)    n="${1:-10}"; shift || true; _fanout src.gqa.kvqa_yesno_fixed_solve "$n" "$@"; exit $? ;;
  yesno-analyze)  exec "$PY" -m src.gqa.kvqa_yesno_fixed_analyze "$@" ;;

  # --- full eligible test split (9424) ----------------------------------------------
  build-full)     exec "$PY" -m src.gqa.kvqa_full_build "$@" ;;
  full-analyze)   exec "$PY" -m src.gqa.kvqa_final_full_analyze "$@" ;;
  full-fanout)    mod="$1"; n="$2"; shift 2; _fanout "$mod" "$n" "$@"; exit $? ;;

  *) echo "usage: $0 {build|softtest|gen|hclt|solve|diag|analyze|review|fanout" >&2
     echo "        |directas-gen|directas-sanity|directas-solve|directas-pilot-build" >&2
     echo "        |directas-pilot-analyze|directas-compare|directas-fanout" >&2
     echo "        |yesno-solve|yesno-analyze|build-full|full-analyze|full-fanout} [args...]" >&2
     exit 1 ;;
esac
