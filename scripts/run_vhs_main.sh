#!/usr/bin/env bash
# Run the VHS grid sequentially so each result can be resumed independently.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/env.sh

read -r -a models <<< "${MODELS:-internvl-8b internvl-4b qwen3vl-8b qwen3vl-4b}"
read -r -a sizes <<< "${HS_LIST:-2 3 5 10 20 50}"
read -r -a modes <<< "${MODE_SET:-base top1 gavr oracle}"
start=${START:-0}
end=${END:-1000}
shards=${SHARDS:-1}
k=${K:-5}
py=${PY:-python}
if (( shards < 1 || start < 0 || end <= start || shards > end - start )); then
  echo 'Use 0 <= START < END and 1 <= SHARDS <= END - START.' >&2
  exit 2
fi

for model in "${models[@]}"; do
  for size in "${sizes[@]}"; do
    for mode in "${modes[@]}"; do
      for ((shard=0; shard<shards; shard++)); do
        lo=$((start + (end - start) * shard / shards))
        hi=$((start + (end - start) * (shard + 1) / shards))
        "$py" experiments/vhs_gavr.py --model "$model" --hs "$size" \
          --mode "$mode" --k "$k" --start "$lo" --end "$hi"
      done
    done
  done
done
