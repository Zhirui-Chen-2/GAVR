# GAVR: Grounded Anchor-based Visual Retrieval

GAVR is a training-free method for multi-image visual question answering. It shortlists images using a vision-language model's pointer head, verifies which candidates contain the object named in the question, and answers from the selected image.

This repository contains the evaluation code, attention-cache extractors, experiment scripts, and [head rankings](assets/head_rankings/SUMMARY.md).

## Setup

Use Python 3.11, Bash, and a CUDA-capable GPU for model inference. From the repository root:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
source scripts/env.sh
```

Choose the PyTorch CUDA wheel index that matches your host and GPU. SDPA is the default attention backend. If you have a compatible FlashAttention 2 installation, set `ATTN_IMPL=flash_attention_2` before starting an experiment.

## Data and model checkpoints

```bash
bash scripts/download_assets.sh
```

The downloader fetches the five model revisions pinned in `utils/model_utils.py`, the [Visual Haystacks](https://huggingface.co/datasets/tsunghanwu/visual_haystacks) single-needle annotations, and the COCO 2017 image splits. Enable `MODELS_ONLY=1` to download only checkpoints. The InternVL loader executes the custom model code from the pinned Hugging Face snapshots.

Override the following paths in your shell before sourcing `scripts/env.sh` to reuse existing downloads or store experiment outputs elsewhere.

| Variable | Default | Expected contents |
| --- | --- | --- |
| `HF_CACHE` | `HF_HUB_CACHE` or `~/.cache/huggingface/hub` | Hugging Face model snapshots |
| `DATA_ROOT` | `data/visual_haystacks` | `single_needle/VHs_large/visual_haystack_<N>.json` |
| `COCO_ROOT` | `data/COCO/images` | COCO split directories such as `val2017` |
| `ICIR_MCQ_ROOT` | `data/icir_mcq` | Optional ICIR-MCQ JSONL files |
| `ATTN_ROOT` | `outputs` | Attention caches and per-sample results |

## Visual Haystacks

Start with five samples at haystack size 10. Use a separate output directory for the smoke run.

```bash
export ATTN_ROOT="$PWD/outputs/smoke"
python data/save_vhs_head_outputs.py --model internvl-8b --hs 10 --limit 5
MODELS=internvl-8b HS_LIST=10 END=5 bash scripts/run_vhs_main.sh
python experiments/agg_main.py --models internvl-8b --hs 10
```

For the full four-model grid, choose an output directory for the run and launch extraction, evaluation, and aggregation together.

```bash
export ATTN_ROOT="$PWD/outputs/vhs"
bash scripts/run_full_pipeline.sh
```

The grid runner evaluates Base Joint, pointer top-1, GAVR, and the single-image oracle. Configure it with `MODELS`, `HS_LIST`, `MODE_SET`, `START`, `END`, `K`, and `SHARDS`. Use a separate `ATTN_ROOT` for runs with different settings. Keep one writer per HDF5 cache and JSONL file. Large joint-inference runs may need multiple GPUs. Set `MULTI_GPU=1` and select the available devices with `CUDA_VISIBLE_DEVICES`.

## ICIR-MCQ

Place a compatible JSONL export under `ICIR_MCQ_ROOT`. To request the dataset in the asset downloader, set `MCQ_DATASET_ID=jerryray/ICIR-MCQ`. Alternatively, build the benchmark from the COCO and MM-NIAH source files using `python data/build_coco_retrieval_dataset_mcq.py --help`. Pass both JSONL files to `--datasets` if you have separate exports for each candidate set size. See the [ICIR-MCQ dataset card](https://huggingface.co/datasets/jerryray/ICIR-MCQ).

```bash
mcq="$ICIR_MCQ_ROOT/coco_retrieval_n5n6_mcq.jsonl"
python data/save_coco_head_outputs_mcq.py --model qwen3vl-4b --datasets "$mcq"
python experiments/mcq_gavr.py --model qwen3vl-4b --mode base --from-h5
python experiments/mcq_gavr.py --model qwen3vl-4b --datasets "$mcq" --mode gavr --k 5
```

## Repository layout

- `utils/` contains the GAVR algorithm, model backends, and shared helpers.
- `data/` contains dataset preparation and attention-cache extraction.
- `experiments/` contains benchmark runners, controls, aggregation, and statistics.
- `training/` contains the offline head-ranking and retrieval-probe experiments.
- `eval/` contains ablations and external retriever comparisons.
- `assets/head_rankings/` contains the released ranking CSVs.

## License and attribution

See [LICENSE](LICENSE) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for code attribution and the terms of the external models and datasets.
