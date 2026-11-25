#!/usr/bin/env bash
# Source from a Bash shell before running the experiments.
export GAVR_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export HF_CACHE="${HF_CACHE:-${HF_HUB_CACHE:-$HOME/.cache/huggingface/hub}}"
export HF_HUB_CACHE="$HF_CACHE"
export DATA_ROOT="${DATA_ROOT:-$GAVR_ROOT/data/visual_haystacks}"
export COCO_ROOT="${COCO_ROOT:-$GAVR_ROOT/data/COCO/images}"
export ICIR_MCQ_ROOT="${ICIR_MCQ_ROOT:-$GAVR_ROOT/data/icir_mcq}"
export ATTN_ROOT="${ATTN_ROOT:-$GAVR_ROOT/outputs}"
export ATTN_IMPL="${ATTN_IMPL:-sdpa}"
export PYTHONPATH="$GAVR_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
