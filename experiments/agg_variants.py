"""
Aggregate variant / control experiments from outputs/results/var_*.jsonl.

    experiments/agg_variants.py --variant ksweep --model internvl-4b --hs 10
    experiments/agg_variants.py --variant noise --model internvl-4b --hs 10 --scope all
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.model_utils import ATTN_ROOT

RESULT_DIR = os.path.join(ATTN_ROOT, 'results')


def load(pattern):
    rows = []
    for p in sorted(glob.glob(pattern)):
        with open(p) as f:
            for line in f:
                try:
                    rows.append(json.loads(line))
                except Exception:
                    continue
    return rows


def summarize_curve(rows, field):
    keys = set()
    for r in rows:
        keys |= set(r.get(field, {}).keys())
    curve = {}
    for k in sorted(keys, key=lambda x: float(x)):
        vals = [r[field][k] for r in rows if k in r.get(field, {})]
        if not vals:
            continue
        curve[k] = {
            'n': len(vals),
            'acc': sum(v['correct'] for v in vals) / len(vals),
            'retr_r1': sum(v['retr_correct'] for v in vals) / len(vals),
        }
    return curve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variant', required=True)
    ap.add_argument('--model', default='internvl-8b')
    ap.add_argument('--hs', type=int, default=10)
    args = ap.parse_args()

    pat = os.path.join(RESULT_DIR, f'var_{args.variant}_{args.model}_hs{args.hs}*.jsonl')
    rows = load(pat)
    if not rows:
        print(f'no results for {pat}')
        return
    field = 'ks' if args.variant == 'ksweep' else 'ps'
    if args.variant in ('ksweep', 'noise', 'noise_all'):
        print(json.dumps({'variant': args.variant, 'model': args.model,
                          'hs': args.hs, 'n': len(rows),
                          'curve': summarize_curve(rows, field)}, indent=2))
    else:
        n = len(rows)
        acc = sum(r.get('correct', 0) for r in rows) / max(n, 1)
        print(json.dumps({'variant': args.variant, 'model': args.model,
                          'hs': args.hs, 'n': n, 'acc': round(acc, 4)},
                         indent=2))


if __name__ == '__main__':
    main()
