#!/usr/bin/env bash
# Set MODELS_ONLY=1 to skip the benchmark datasets.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/env.sh

HF=${HF:-hf}
command -v "$HF" >/dev/null || { echo 'The hf command is provided by huggingface_hub.' >&2; exit 1; }

models=(
  'OpenGVLab/InternVL3_5-8B|9bb6a56ad9cc69db95e2d4eeb15a52bbcac4ef79'
  'OpenGVLab/InternVL3_5-4B|481f6e32467eab4e922ccd7fd6cf420441a62331'
  'OpenGVLab/InternVL3_5-14B|a1e37197b393ce9eec9df700fef65c11f4a6ffbd'
  'Qwen/Qwen3-VL-8B-Instruct|0c351dd01ed87e9c1b53cbc748cba10e6187ff3b'
  'Qwen/Qwen3-VL-4B-Instruct|ebb281ec70b05090aa6165b016eac8ec08e71b17'
)
for entry in "${models[@]}"; do
  "$HF" download "${entry%%|*}" --revision "${entry##*|}"
done

if [[ ${MODELS_ONLY:-0} == 1 ]]; then
  exit 0
fi

vhs_files=()
for size in 2 3 5 10 20 50; do
  vhs_files+=("single_needle/VHs_large/visual_haystack_${size}.json")
done
"$HF" download tsunghanwu/visual_haystacks "${vhs_files[@]}" \
  --repo-type dataset --local-dir "$DATA_ROOT"

for split in train2017 val2017 test2017; do
  if [[ ! -d "$COCO_ROOT/$split" ]]; then
    mkdir -p "$COCO_ROOT/downloads"
    archive="$COCO_ROOT/downloads/$split.zip"
    if [[ ! -s "$archive" ]]; then
      curl -fL --retry 3 "https://images.cocodataset.org/zips/$split.zip" -o "$archive.part"
      mv "$archive.part" "$archive"
    fi
    unzip -qn "$archive" -d "$COCO_ROOT"
  fi
done

if [[ -n ${MCQ_DATASET_ID:-} ]]; then
  "$HF" download "$MCQ_DATASET_ID" --repo-type dataset --local-dir "$ICIR_MCQ_ROOT"
fi
