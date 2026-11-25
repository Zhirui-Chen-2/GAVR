"""
VHS main evaluation: Base-Joint / Pointer top-1 / GAVR / Oracle / Fusion.

One JSONL record per sample, flushed immediately so shards can be killed,
resumed and aggregated independently:

    experiments/vhs_gavr.py --model internvl-8b --hs 10 --mode gavr --k 5 \
        --start 0 --end 200 --out outputs/results/vhs_internvl-8b_gavr_hs10_k5_s0.jsonl

Modes
    base   : all N images jointly (the "Base Joint" baseline)
    top1   : pointer top-1 image only (the paper's retrieval baseline)
    gavr   : top-k candidates -> anchor P(yes) re-rank -> single-image answer
    oracle : ground-truth image only (retrieval upper bound)
    fusion : base + gavr with the 'base if mj > k*ms else GAVR' gate
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.model_backend import ModelBackend
from utils import gavr_core as G
from utils.model_utils import MODEL_CONFIGS, ATTN_ROOT


def default_out(args):
    layer, head = G.POINTER_HEADS[args.model]
    k = f'_k{args.k}' if args.mode in ('gavr', 'fusion') else ''
    s = f'_s{args.start}' if args.start is not None else ''
    name = f"vhs_{args.model}_{args.mode}_hs{args.hs}{k}{s}.jsonl"
    return os.path.join(ATTN_ROOT, 'results', name)


def existing_ids(path):
    ids = set()
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                try:
                    record = json.loads(line)
                    if 'error' not in record:
                        ids.add(record['sid'])
                except (ValueError, KeyError, TypeError):
                    continue
    return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, choices=list(MODEL_CONFIGS.keys()))
    ap.add_argument('--hs', type=int, required=True, help='haystack size N')
    ap.add_argument('--mode', default='gavr',
                    choices=['base', 'top1', 'gavr', 'oracle', 'fusion'])
    ap.add_argument('--k', type=int, default=5, help='number of verified candidates')
    ap.add_argument('--fusion-k', type=float, default=10.0)
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--start', type=int, default=None)
    ap.add_argument('--end', type=int, default=None)
    ap.add_argument('--pointer-layer', type=int, default=None)
    ap.add_argument('--pointer-head', type=int, default=None)
    ap.add_argument('--phase', default='cattok',
                    choices=['cattok', 'generation'])
    ap.add_argument('--res', default='448', choices=['448', 'native'],
                    help='MM_NATIVE_RES for Qwen3-VL / InternVL tiling')
    ap.add_argument('--data-root', default=G.VHS_DATA_ROOT)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    def answer_final(haystack_rels, question):
        """Final answer on the selected image(s); honors --res without leaking
        MM_NATIVE_RES into verification (protocol: verify at 448)."""
        if args.res == 'native':
            os.environ['MM_NATIVE_RES'] = '1'
        try:
            return G.answer(backend, haystack_rels, question)
        finally:
            if args.res == 'native':
                os.environ.pop('MM_NATIVE_RES', None)

    layer, head = G.POINTER_HEADS[args.model]
    if args.pointer_layer is not None:
        layer = args.pointer_layer
    if args.pointer_head is not None:
        head = args.pointer_head

    backend = ModelBackend(args.model)
    samples, hmap = G.load_vhs_samples(args.hs, args.data_root)
    if args.limit:
        samples = samples[:args.limit]
    if args.start is not None or args.end is not None:
        samples = samples[args.start:args.end]

    h5 = G.h5_path(args.model, args.hs)
    out = args.out or default_out(args)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    done = existing_ids(out)

    n_correct = n_r1 = n_seen = 0
    with open(out, 'a') as fo:
        for s in samples:
            sid = s['id']
            if sid in done:
                continue
            haystack = hmap[sid]
            pos = G.pos_indices(s, haystack)[0]
            question = s['conversations'][0]['value']
            gt = G.sample_gt(s)

            t0 = time.time()
            rec = {'sid': sid, 'hs': args.hs, 'n': len(haystack),
                   'model': args.model, 'mode': args.mode,
                   'gt': gt, 'pos_idx': pos}

            scores = None
            if os.path.exists(h5):
                try:
                    scores = G.pointer_scores(h5, sid, layer, head, args.phase)
                except KeyError:
                    scores = None

            if scores is not None:
                rec['ptr_idx'] = int(np.argmax(scores))
                rec['retr_correct'] = int(rec['ptr_idx'] == pos)

            base = None
            if args.mode in ('base', 'fusion'):
                base = G.answer(backend, haystack, question)
                rec['base_pred'] = base['pred']
                rec['base_correct'] = int(base['pred'] == gt)
                rec['base_margin'] = base['margin']

            if args.mode == 'top1':
                if scores is None:
                    rec['error'] = 'no attention scores (run the h5 extraction)'
                    fo.write(json.dumps(rec) + '\n'); fo.flush()
                    continue
                single = answer_final([haystack[rec['ptr_idx']]], question)
                rec.update(pred=single['pred'],
                           correct=int(single['pred'] == gt),
                           margin=single['margin'],
                           response=single['response'])
            elif args.mode == 'oracle':
                single = answer_final([haystack[pos]], question)
                rec.update(pred=single['pred'],
                           correct=int(single['pred'] == gt),
                           margin=single['margin'],
                           response=single['response'])
            elif args.mode in ('gavr', 'fusion'):
                if scores is None:
                    rec['error'] = 'no attention scores (run the h5 extraction)'
                    fo.write(json.dumps(rec) + '\n'); fo.flush()
                    continue
                anchor = G.parse_anchor(question) or G.parse_needle(question) \
                    or s.get('needle', '')
                cand = G.pointer_topk(scores, min(args.k, len(haystack)))
                ps = [float(x) for x in G.verify_candidates(backend, haystack, cand, anchor)]
                gavr_idx = int(cand[int(np.argmax(ps))])
                gav = answer_final([haystack[gavr_idx]], question)
                rec.update(gavr_idx=gavr_idx,
                           gavr_retr_correct=int(gavr_idx == pos),
                           gavr_scores=ps, anchor=anchor,
                           gavr_pred=gav['pred'],
                           gavr_correct=int(gav['pred'] == gt),
                           gavr_margin=gav['margin'])
                if args.mode == 'fusion':
                    chosen = G.fuse_answer(base, gav, args.fusion_k)
                    rec.update(pred=chosen['pred'],
                               correct=int(chosen['pred'] == gt),
                               margin=chosen['margin'],
                               fused_from=('base' if chosen is base else 'gavr'))
                else:
                    rec.update(pred=gav['pred'],
                               correct=int(gav['pred'] == gt),
                               margin=gav['margin'],
                               response=gav['response'])
            elif args.mode == 'base':
                rec.update(pred=base['pred'],
                           correct=int(base['pred'] == gt),
                           margin=base['margin'],
                           response=base['response'])

            rec['elapsed_s'] = round(time.time() - t0, 2)
            fo.write(json.dumps(rec) + '\n')
            fo.flush()

            n_seen += 1
            n_correct += rec.get('correct', 0)
            if 'retr_correct' in rec:
                n_r1 += rec['retr_correct']
            if n_seen % 20 == 0:
                msg = f"[{args.model} hs={args.hs} {args.mode}] {n_seen} samples"
                if 'correct' in rec:
                    msg += f"  acc={n_correct / n_seen:.3f}"
                if scores is not None:
                    msg += f"  retrR@1={n_r1 / n_seen:.3f}"
                print(msg, flush=True)

    print(f"done -> {out}", flush=True)


if __name__ == '__main__':
    main()
