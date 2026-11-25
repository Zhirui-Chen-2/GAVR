#!/usr/bin/env bash
# Build the VHS attention caches, run evaluations, then aggregate the results.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/env.sh

read -r -a models <<< "${MODELS:-internvl-8b internvl-4b qwen3vl-8b qwen3vl-4b}"
read -r -a sizes <<< "${HS_LIST:-2 3 5 10 20 50}"
py=${PY:-python}
for model in "${models[@]}"; do
  "$py" data/save_vhs_head_outputs.py --model "$model" --hs "${sizes[@]}" \
    --start "${START:-0}" --end "${END:-1000}"
done
bash scripts/run_vhs_main.sh
"$py" experiments/agg_main.py --models "${models[@]}" --hs "${sizes[@]}" --k "${K:-5}"
