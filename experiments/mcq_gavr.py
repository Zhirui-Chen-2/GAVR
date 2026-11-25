"""
ICIR-MCQ GAVR evaluation (the COCO side of the paper table).

For every sample, the paper protocol is 4 option rotations (A/B/C/D as the
correct letter). The base accuracy can be read offline from the extraction h5
(each `ans{L}` group stores response + correct). GAVR needs the model:

    pointer scores (attn_score/scores_sum at the category anchor)
    -> top-k candidate images
    -> "Is there a <target_category> in this image?" P(yes) verified
    -> re-run the MCQ with the rotated options on the chosen image only

    experiments/mcq_gavr.py --model qwen3vl-4b \
        --datasets data/icir_mcq/coco_retrieval_n5n6_mcq.jsonl \
        --mode gavr --k 5 --start 0 --end 1500

--mode base --from-h5 aggregates the cached base predictions without GPU.
"""
import argparse
import json
import os
import sys

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import gavr_core as G
from utils.model_utils import ATTN_ROOT, MODEL_CONFIGS

MCQ_LABELS = ['A', 'B', 'C', 'D']


def rotate_options(options, original_answer, new_answer):
    if original_answer == new_answer:
        return dict(options)
    o = dict(options)
    o[new_answer] = options[original_answer]
    o[original_answer] = options[new_answer]
    return o


def build_question(category, options):
    opts = '\n'.join(f'{L}) {options[L]}' for L in MCQ_LABELS)
    return (f'Which caption best describes the image above that contains '
            f'{category}? Answer with a single letter (A, B, C, or D).\n{opts}')


def parse_letter(resp):
    import re
    if not resp:
        return None
    m = re.search(r'\b([ABCD])\b', resp.strip().upper()[:40])
    return m.group(1) if m else None


def single_image_context(context, chosen):
    """Keep the chosen image's <image> slot in the context, drop the others."""
    parts = context.split('<image>')
    out = parts[0]
    for i in range(1, len(parts)):
        out += ('<image>' if i - 1 == chosen else '') + parts[i]
    return out


def mcq_answer(backend, image_rels, category, context, options, max_new_tokens=8):
    """Answer one rotated MCQ on the given images; letter from constrained argmax."""
    pils = [G.load_pil(r) for r in image_rels]
    prompt = context + '\n' + build_question(category, options)
    inputs = backend.build_inputs(pils, prompt, interleave_marker='<image>')
    gen = backend.generate(inputs, max_new_tokens=max_new_tokens,
                           capture_first_logits=True)
    letter = parse_letter(gen['response'])
    if letter is None and gen.get('first_logits') is not None:
        ids = backend.get_letter_ids(MCQ_LABELS)
        logits = gen['first_logits']
        letter = max(MCQ_LABELS, key=lambda L: float(logits[ids[L]]))
    return letter, gen['response']


def h5_path(model, outputs_dir=None):
    d = outputs_dir or os.path.join(ATTN_ROOT,
                                    MODEL_CONFIGS[model]['model_dir'], 'coco_mcq')
    if os.path.exists(os.path.join(d, 'head_outputs.h5')):
        return os.path.join(d, 'head_outputs.h5')
    # sharded extraction: merge happens externally; use worker 0 by default
    return os.path.join(d, 'head_outputs_w0.h5')


def scores_from_h5(path, sid, layer, head, kind='scores_sum'):
    with h5py.File(path, 'r') as f:
        s = f[f'sample_{sid}']['attn_score'][kind][layer, head]
        return np.asarray(s)


def run_base_offline(args):
    path = args.h5 or h5_path(args.model, args.outputs_dir)
    n = ok = 0
    per_sample = []
    with h5py.File(path, 'r') as f:
        for sid_key in f.keys():
            grp = f[sid_key]
            corrs = []
            for L in MCQ_LABELS:
                if f'ans{L}' in grp:
                    corrs.append(int(grp[f'ans{L}'].attrs['correct']))
            if not corrs:
                continue
            n += 1
            ok += sum(corrs) / len(corrs)
            per_sample.append({'sid': sid_key, 'base_acc': sum(corrs) / len(corrs)})
    print(json.dumps({'model': args.model, 'n': n,
                      'base_acc_4rot': ok / max(n, 1)}, indent=2))
    if args.out:
        with open(args.out, 'w') as fo:
            for r in per_sample:
                fo.write(json.dumps(r) + '\n')


def run_gavr(args, backend):
    samples = []
    for p in args.datasets:
        samples.extend(json.loads(l) for l in open(p))
    if args.limit:
        samples = samples[:args.limit]
    if args.start is not None or args.end is not None:
        samples = samples[args.start:args.end]

    layer, head = G.POINTER_HEADS[args.model]
    if args.pointer_layer is not None:
        layer = args.pointer_layer
    if args.pointer_head is not None:
        head = args.pointer_head
    path = args.h5 or h5_path(args.model, args.outputs_dir)

    done = set()
    if os.path.exists(args.out):
        with open(args.out) as fh:
            for line in fh:
                try:
                    done.add(json.loads(line)['sid'])
                except Exception:
                    continue

    n = ok = n_r1 = 0
    with open(args.out, 'a') as fo:
        for s in samples:
            sid = str(s['id'])
            if sid in done:
                continue
            try:
                scores = scores_from_h5(path, sid, layer, head)
            except KeyError:
                continue
            cat = s['target_category']
            target = int(s['target_index'])
            cand = G.pointer_topk(scores, min(args.k, len(s['images_list'])))
            pvals = [G.p_yes(backend, s['images_list'][i], cat) for i in cand]
            chosen = cand[int(np.argmax(pvals))]

            corrs, preds = [], []
            ctx_single = single_image_context(s['context'], chosen)
            for L in MCQ_LABELS:
                opts = rotate_options(s['mcq_options'], s['mcq_answer'], L)
                letter, _ = mcq_answer(backend, [s['images_list'][chosen]],
                                       cat, ctx_single, opts)
                preds.append(letter)
                corrs.append(int(letter == L))
            acc = sum(corrs) / len(corrs)
            n += 1
            ok += acc
            r1 = int(chosen == target)
            n_r1 += r1
            rec = {'sid': sid, 'n_images': s['n_images'], 'target': target,
                   'chosen': int(chosen), 'retr_correct': r1,
                   'p_yes': pvals, 'preds': preds, 'correct': corrs,
                   'acc': acc, 'model': args.model}
            fo.write(json.dumps(rec) + '\n')
            fo.flush()
            if n % 20 == 0:
                print(f'[{args.model} mcq gavr] {n}  gavr_acc={ok / n:.3f} '
                      f'retrR@1={n_r1 / n:.3f}', flush=True)
    print(f'done -> {args.out}  gavr_acc={ok / max(n, 1):.4f}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, choices=list(MODEL_CONFIGS.keys()))
    ap.add_argument('--datasets', nargs='+',
                    default=['data/icir_mcq/coco_retrieval_n5n6_mcq.jsonl'])
    ap.add_argument('--mode', default='gavr', choices=['base', 'gavr'])
    ap.add_argument('--from-h5', action='store_true',
                    help='base mode only: read cached predictions (no GPU)')
    ap.add_argument('--k', type=int, default=5)
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--start', type=int, default=None)
    ap.add_argument('--end', type=int, default=None)
    ap.add_argument('--pointer-layer', type=int, default=None)
    ap.add_argument('--pointer-head', type=int, default=None)
    ap.add_argument('--h5', default=None)
    ap.add_argument('--outputs-dir', default=None)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    if args.mode == 'base':
        run_base_offline(args)
        return

    if not args.out:
        args.out = os.path.join(
            ATTN_ROOT, 'results',
            f"mcq_{args.model}_gavr_k{args.k}_s{args.start or 0}.jsonl")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    from utils.model_backend import ModelBackend
    backend = ModelBackend(args.model)
    run_gavr(args, backend)


if __name__ == '__main__':
    main()