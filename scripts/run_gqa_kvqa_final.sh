#!/usr/bin/env bash
# KVQA FINAL — Generic / HCLT-style / Ours-Raw / Ours-AS  (Text-only + Multimodal) on the fixed 2K.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=/ceph_data/jaydenpark/korean_caption_bottleneck
export HF_HOME=/ceph_data/jaydenpark/korean_caption_bottleneck/cache/hf
export TOKENIZERS_PARALLELISM=false
PYBIN=/ceph_data/jaydenpark/envs/qwen3vl/bin/python

STEP="${1:-}"; shift || true
case "$STEP" in
  build)      exec "$PYBIN" -m src.gqa.kvqa_final_build "$@" ;;
  softtest)   exec "$PYBIN" -m src.gqa.kvqa_softscore "$@" ;;
  gen)        exec "$PYBIN" -m src.gqa.kvqa_final_gen "$@" ;;
  hclt)       exec "$PYBIN" -m src.gqa.kvqa_final_hclt "$@" ;;
  solve)      exec "$PYBIN" -m src.gqa.kvqa_final_solve "$@" ;;
  diag)       exec "$PYBIN" -m src.gqa.kvqa_final_diag "$@" ;;
  analyze)    exec "$PYBIN" -m src.gqa.kvqa_final_analyze "$@" ;;
  review)     exec "$PYBIN" -m src.gqa.kvqa_final_review "$@" ;;
  *) echo "usage: $0 {build|softtest|gen|hclt|solve|diag|analyze|review} [args...]" >&2; exit 1 ;;
esac
