"""
Offline pointer-head R@1 scan from cached h5 files (no GPU).

For every (layer, head) it computes top-1 retrieval accuracy on the VHS h5:
argmax_head(image attention at the anchor token) == pos_idx.
This is the offline reproduction of the paper's Table 5 and the tool used to
diagnose the InternVL-14B bad pointer head (L26H36 .677 -> L24H0 .843).

    experiments/agg_pointer_r1.py --model internvl-8b --hs 2 3 5 10 20 50
    experiments/agg_pointer_r1.py --model internvl-14b --hs 10 --topk 10
"""
import argparse
import os
import sys

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import gavr_core as G
from utils.model_utils import MODEL_CONFIGS


def scan(path, topk=5, layer=None):
    per_head = {}
    n = 0
    with h5py.File(path, 'r') as f:
        keys = [k for k in f.keys() if k.startswith('sample_')]
        n = len(keys)
        for key in keys:
            grp = f[key]
            if 'cattok_attn_scores' not in grp:
                continue
            pos = int(grp.attrs['pos_idx'])
            if 'pos_idxs' in grp.attrs:
                pos_set = {int(p) for p in grp.attrs['pos_idxs']}
            else:
                pos_set = {pos}
            layers = list(grp['cattok_attn_scores'].keys())
            if layer is not None:
                layers = [k for k in layers if k == f'layer_{layer}']
            for lk in layers:
                s = grp['cattok_attn_scores'][lk][:]  # (H, n_imgs)
                hit = (s.argmax(axis=1)[:, None] ==
                       np.array(sorted(pos_set))[None, :]).any(axis=1)
                li = int(lk.split('_')[1])
                for hi in range(s.shape[0]):
                    per_head.setdefault((li, hi), 0)
                    per_head[(li, hi)] += int(hit[hi])
    return {k: v / max(n, 1) for k, v in per_head.items()}, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, choices=list(MODEL_CONFIGS.keys()))
    ap.add_argument('--hs', nargs='+', type=int, default=[10])
    ap.add_argument('--topk', type=int, default=5)
    ap.add_argument('--layer', type=int, default=None,
                    help='only scan this layer')
    ap.add_argument('--attn-root', default=None)
    args = ap.parse_args()

    default_head = G.POINTER_HEADS[args.model]
    print(f'model={args.model}  default head L{default_head[0]}H{default_head[1]}')
    for hs in args.hs:
        path = G.h5_path(args.model, hs, args.attn_root)
        if not os.path.exists(path):
            print(f'  hs={hs}: missing {path}')
            continue
        per_head, n = scan(path, args.topk, args.layer)
        if not per_head:
            print(f'  hs={hs}: no attention scores in {path}')
            continue
        best = sorted(per_head.items(), key=lambda kv: -kv[1])
        d = per_head.get(default_head)
        print(f'\n  hs={hs}  n={n}  samples')
        print(f'    default L{default_head[0]}H{default_head[1]}: '
              f'R@1={d:.3f}' if d is not None else '    default head: n/a')
        print(f'    best {args.topk} heads:')
        for (li, hi), v in best[:args.topk]:
            print(f'      L{li}H{hi}: {v:.3f}')


if __name__ == '__main__':
    main()
