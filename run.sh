#!/usr/bin/env bash
# Portable entrypoint for the KVQA representation comparison.
#   PYTHON     : python interpreter (default: python)
#   GQA_FORCE_GPU : physical GPU id to pin for GPU steps (gen / hclt / solve)
#   KVQA_ANNOTATIONS / KVQA_IMAGE_GLOB / KVQA_SPLIT : data locations (see README / NOTICE)
#   HCLT_REPO  : optional path to the KVQA structured-caption repo for evaluator cross-check
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
PY="${PYTHON:-python}"

STEP="${1:-}"; shift || true
case "$STEP" in
  build)    exec "$PY" -m src.gqa.kvqa_final_build "$@" ;;
  softtest) exec "$PY" -m src.gqa.kvqa_softscore "$@" ;;
  gen)      exec "$PY" -m src.gqa.kvqa_final_gen "$@" ;;
  hclt)     exec "$PY" -m src.gqa.kvqa_final_hclt "$@" ;;
  solve)    exec "$PY" -m src.gqa.kvqa_final_solve "$@" ;;
  diag)     exec "$PY" -m src.gqa.kvqa_final_diag "$@" ;;
  analyze)  exec "$PY" -m src.gqa.kvqa_final_analyze "$@" ;;
  review)   exec "$PY" -m src.gqa.kvqa_final_review "$@" ;;
  fanout)   # fanout <step> <N> [args...] : one worker per GPU 0..N-1
            s="$1"; n="$2"; shift 2
            pids=()
            for ((i=0;i<n;i++)); do
              GQA_FORCE_GPU=$i "$PY" -m "src.gqa.kvqa_final_${s}" --shard "$i/$n" "$@" &
              pids+=($!)
            done
            rc=0; for p in "${pids[@]}"; do wait "$p" || rc=1; done; exit $rc ;;
  *) echo "usage: $0 {build|softtest|gen|hclt|solve|diag|analyze|review|fanout} [args...]" >&2; exit 1 ;;
esac
