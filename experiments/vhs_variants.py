"""
VHS variant / control experiments (reconstructed from the session logs).

Variants
    ksweep        GAVR with several candidate counts k (--ks 1 3 5 10)
    noise         replace candidates with random images w.p. p (--ps .25 .5 1)
                  --scope top1 (only the pointer top-1) or all (whole top-k set)
    noise_all     alias of --variant noise --scope all
    text_only     answer without any image (language-prior control)
    order         shuffle the haystack order (position-bias control)
    first_image   always answer the first image
    last_image    always answer the last image
    topk_vote     answer each of the pointer top-k images, majority vote
    selfcons      answer ALL N images, majority vote (image self-consistency)
    margin_grid   offline gate sweep from previously saved GAVR+base JSONLs
    timing        per-sample latency for base / top1 / gavr

Each variant writes one JSONL per (variant, model, hs) under outputs/results/.

Examples
    experiments/vhs_variants.py --variant ksweep --model internvl-4b --hs 10 --ks 1 3 5
    experiments/vhs_variants.py --variant noise --model internvl-4b --hs 10 \
        --ps 0 .25 .5 1 --scope all
    experiments/vhs_variants.py --variant selfcons --model internvl-8b --hs 10 --limit 1000
"""
import argparse
import json
import os
import random
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.model_backend import ModelBackend
from utils import gavr_core as G
from utils.model_utils import MODEL_CONFIGS, ATTN_ROOT

RESULT_DIR = os.path.join(ATTN_ROOT, 'results')


def majority(preds):
    yes = sum(1 for p in preds if p == 'yes')
    no = sum(1 for p in preds if p == 'no')
    if yes == no:
        return None
    return 'yes' if yes > no else 'no'


def out_path(variant, args, extra=''):
    s = f'_s{args.start}' if args.start else ''
    return os.path.join(
        RESULT_DIR,
        f'var_{variant}_{args.model}_hs{args.hs}{extra}{s}.jsonl')


def open_resumed(path):
    done = set()
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                try:
                    done.add(json.loads(line)['sid'])
                except Exception:
                    continue
    return done


def iter_samples(args):
    samples, hmap = G.load_vhs_samples(args.hs, args.data_root)
    if args.limit:
        samples = samples[:args.limit]
    if args.start is not None or args.end is not None:
        samples = samples[args.start:args.end]
    return samples, hmap


def load_pointer(args):
    layer, head = G.POINTER_HEADS[args.model]
    if args.pointer_layer is not None:
        layer = args.pointer_layer
    if args.pointer_head is not None:
        head = args.pointer_head
    h5 = G.h5_path(args.model, args.hs)
    return lambda sid: G.pointer_scores(h5, sid, layer, head, args.phase)


# ── GPU variants ──────────────────────────────────────────────────────────────

def run_gavr_variant(backend, samples, hmap, args, variant, scores_fn,
                     ks=None, ps=None, scope='top1', seed=None):
    path = out_path(variant, args,
                    f"_k{ks[0]}" if ks else '')
    done = open_resumed(path)
    confirmed_native = os.environ.get('MM_NATIVE_RES', '0')
    os.environ['MM_NATIVE_RES'] = '0'  # verification always at 448 (log bugfix)

    with open(path, 'a') as fo:
        for s in samples:
            sid = s['id']
            if sid in done:
                continue
            haystack = hmap[sid]
            pos = G.pos_indices(s, haystack)[0]
            question = s['conversations'][0]['value']
            gt = G.sample_gt(s)
            anchor = G.parse_anchor(question) or G.parse_needle(question) \
                or s.get('needle', '')
            scores = scores_fn(sid)
            t0 = time.time()

            if variant == 'ksweep':
                row = {'ks': {}}
                for k in ks:
                    cand = G.pointer_topk(scores, min(k, len(haystack)))
                    pvals = G.verify_candidates(backend, haystack, cand, anchor)
                    idx = cand[int(np.argmax(pvals))]
                    a = G.answer(backend, [haystack[idx]], question)
                    row['ks'][str(k)] = {'idx': int(idx),
                                         'pred': a['pred'],
                                         'correct': int(a['pred'] == gt),
                                         'retr_correct': int(idx == pos)}
                rec = {'sid': sid, 'hs': args.hs, 'gt': gt, 'pos_idx': pos, **row}
            else:  # noise / noise_all
                row = {'ps': {}}
                for p in ps:
                    # scope=top1 pollutes only the pointer top-1 (first design:
                    # the verifier can still rescue from the rest of top-k);
                    # scope=all pollutes the whole candidate set (main setting).
                    n_cand = 1 if scope == 'top1' else min(args.k, len(haystack))
                    cand = G.pointer_topk(scores, n_cand)
                    rng = random.Random(hash((sid, p)) & 0xffffffff)
                    polluted = []
                    for c in cand:
                        polluted.append(c if rng.random() >= p
                                        else rng.randrange(len(haystack)))
                    pvals = G.verify_candidates(backend, haystack, polluted, anchor)
                    idx = polluted[int(np.argmax(pvals))]
                    a = G.answer(backend, [haystack[idx]], question)
                    row['ps'][str(p)] = {'idx': int(idx), 'pred': a['pred'],
                                         'correct': int(a['pred'] == gt),
                                         'retr_correct': int(idx == pos)}
                rec = {'sid': sid, 'hs': args.hs, 'gt': gt, 'pos_idx': pos, **row}

            rec['elapsed_s'] = round(time.time() - t0, 2)
            fo.write(json.dumps(rec) + '\n')
            fo.flush()
    os.environ['MM_NATIVE_RES'] = confirmed_native
    print(f'done -> {path}')


def run_single_answer(backend, samples, hmap, args, variant, pick):
    path = out_path(variant, args,
                    f"_seed{args.seed}" if variant == 'order' else '')
    done = open_resumed(path)
    confirmed_native = os.environ.get('MM_NATIVE_RES', '0')
    os.environ['MM_NATIVE_RES'] = '0'
    with open(path, 'a') as fo:
        for s in samples:
            sid = s['id']
            if sid in done:
                continue
            haystack = hmap[sid]
            pos = G.pos_indices(s, haystack)[0]
            question = s['conversations'][0]['value']
            gt = G.sample_gt(s)
            t0 = time.time()
            if variant == 'text_only':
                a = G.answer(backend, [], question)
                idx = None
            elif variant == 'order':
                order = list(haystack)
                random.Random(hash((sid, args.seed)) & 0xffffffff).shuffle(order)
                a = G.answer(backend, order, question)
                idx = None
            else:  # first_image / last_image
                idx = 0 if variant == 'first_image' else len(haystack) - 1
                a = G.answer(backend, [haystack[idx]], question)
            rec = {'sid': sid, 'hs': args.hs, 'gt': gt, 'pos_idx': pos,
                   'idx': idx, 'pred': a['pred'],
                   'correct': int(a['pred'] == gt), 'margin': a['margin'],
                   'elapsed_s': round(time.time() - t0, 2)}
            fo.write(json.dumps(rec) + '\n')
            fo.flush()
    os.environ['MM_NATIVE_RES'] = confirmed_native
    print(f'done -> {path}')


def run_vote(backend, samples, hmap, args, variant, scores_fn, k=None):
    path = out_path(variant, args, f"_k{k}" if k else '')
    done = open_resumed(path)
    confirmed_native = os.environ.get('MM_NATIVE_RES', '0')
    os.environ['MM_NATIVE_RES'] = '0'
    with open(path, 'a') as fo:
        for s in samples:
            sid = s['id']
            if sid in done:
                continue
            haystack = hmap[sid]
            pos = G.pos_indices(s, haystack)[0]
            question = s['conversations'][0]['value']
            gt = G.sample_gt(s)
            t0 = time.time()
            if variant == 'topk_vote':
                idxs = G.pointer_topk(scores_fn(sid), min(k, len(haystack)))
            else:  # selfcons
                idxs = list(range(len(haystack)))
            preds = []
            for i in idxs:
                a = G.answer(backend, [haystack[i]], question)
                preds.append(a['pred'])
            pred = majority(preds)
            rec = {'sid': sid, 'hs': args.hs, 'gt': gt, 'pos_idx': pos,
                   'idxs': [int(i) for i in idxs], 'preds': preds,
                   'pred': pred, 'correct': int(pred == gt),
                   'n_yes': sum(1 for p in preds if p == 'yes'),
                   'elapsed_s': round(time.time() - t0, 2)}
            fo.write(json.dumps(rec) + '\n')
            fo.flush()
    os.environ['MM_NATIVE_RES'] = confirmed_native
    print(f'done -> {path}')


def run_timing(backend, samples, hmap, args, scores_fn):
    path = out_path('timing', args)
    done = open_resumed(path)
    with open(path, 'a') as fo:
        for s in samples:
            sid = s['id']
            if sid in done:
                continue
            haystack = hmap[sid]
            question = s['conversations'][0]['value']
            anchor = G.parse_anchor(question) or parse_needle_safe(question)
            scores = scores_fn(sid)
            rec = {'sid': sid, 'hs': args.hs, 'n': len(haystack)}
            t0 = time.time(); G.answer(backend, haystack, question)
            rec['base_s'] = round(time.time() - t0, 2)
            t0 = time.time()
            G.answer(backend, [haystack[int(np.argmax(scores))]], question)
            rec['top1_s'] = round(time.time() - t0, 2)
            t0 = time.time()
            cand = G.pointer_topk(scores, min(args.k, len(haystack)))
            G.verify_candidates(backend, haystack, cand, anchor)
            G.answer(backend, [haystack[cand[0]]], question)
            rec['gavr_s'] = round(time.time() - t0, 2)
            fo.write(json.dumps(rec) + '\n')
            fo.flush()
    print(f'done -> {path}')


def parse_needle_safe(q):
    return G.parse_needle(q) or ''


# ── Offline margin gate sweep ────────────────────────────────────────────────

def margin_grid(args):
    """Sweep the gate factor on saved base+gavr JSONLs (no GPU).

    Accepts either one file per mode (--base-jsonl/--gavr-jsonl) or the
    fusion JSONL produced by vhs_gavr.py --mode fusion.
    """
    base, gavr = {}, {}
    if args.base_jsonl and args.gavr_jsonl:
        for path, dest in ((args.base_jsonl, base), (args.gavr_jsonl, gavr)):
            with open(path) as f:
                for line in f:
                    r = json.loads(line)
                    dest[r['sid']] = r
    else:
        with open(args.fusion_jsonl) as f:
            for line in f:
                r = json.loads(line)
                base[r['sid']] = {'pred': r['base_pred'],
                                  'correct': r['base_correct'],
                                  'margin': r['base_margin']}
                gavr[r['sid']] = {'pred': r['gavr_pred'],
                                  'correct': r['gavr_correct'],
                                  'margin': r.get('gavr_margin', 0.0)}
    sids = sorted(set(base) & set(gavr))
    rows = []
    for k in [0.25, 0.5, 1, 2, 5, 10, 20, 1e9]:
        ok = 0
        for sid in sids:
            b, g = base[sid], gavr[sid]
            use_base = b['margin'] > k * max(g['margin'], 1e-6)
            ok += b['correct'] if use_base else g['correct']
        rows.append({'gate_k': k, 'acc': ok / max(len(sids), 1)})
    print(json.dumps({'n': len(sids), 'grid': rows}, indent=2))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variant', required=True,
                    choices=['ksweep', 'noise', 'noise_all', 'text_only',
                             'order', 'first_image', 'last_image', 'topk_vote',
                             'selfcons', 'margin_grid', 'timing'])
    ap.add_argument('--model', default='internvl-8b',
                    choices=list(MODEL_CONFIGS.keys()))
    ap.add_argument('--hs', type=int, default=10)
    ap.add_argument('--k', type=int, default=5)
    ap.add_argument('--ks', nargs='+', type=int, default=[1, 3, 5, 10])
    ap.add_argument('--ps', nargs='+', type=float, default=[0.0, 0.25, 0.5, 1.0])
    ap.add_argument('--scope', default='top1', choices=['top1', 'all'])
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--start', type=int, default=None)
    ap.add_argument('--end', type=int, default=None)
    ap.add_argument('--pointer-layer', type=int, default=None)
    ap.add_argument('--pointer-head', type=int, default=None)
    ap.add_argument('--phase', default='cattok')
    ap.add_argument('--res', default='448', choices=['448', 'native'])
    ap.add_argument('--data-root', default=G.VHS_DATA_ROOT)
    ap.add_argument('--base-jsonl', default=None)
    ap.add_argument('--gavr-jsonl', default=None)
    ap.add_argument('--fusion-jsonl', default=None)
    args = ap.parse_args()

    if args.variant == 'margin_grid':
        margin_grid(args)
        return

    if args.res == 'native':
        os.environ['MM_NATIVE_RES'] = '1'

    samples, hmap = iter_samples(args)
    backend = ModelBackend(args.model)
    scores_fn = load_pointer(args)
    os.makedirs(RESULT_DIR, exist_ok=True)

    v = args.variant
    if v == 'ksweep':
        run_gavr_variant(backend, samples, hmap, args, v, scores_fn, ks=args.ks)
    elif v in ('noise', 'noise_all'):
        run_gavr_variant(backend, samples, hmap, args, v, scores_fn,
                         ps=args.ps,
                         scope='all' if v == 'noise_all' else args.scope)
    elif v in ('text_only', 'order', 'first_image', 'last_image'):
        run_single_answer(backend, samples, hmap, args, v, None)
    elif v == 'topk_vote':
        run_vote(backend, samples, hmap, args, v, scores_fn, k=args.k)
    elif v == 'selfcons':
        run_vote(backend, samples, hmap, args, v, scores_fn)
    elif v == 'timing':
        run_timing(backend, samples, hmap, args, scores_fn)


if __name__ == '__main__':
    main()
