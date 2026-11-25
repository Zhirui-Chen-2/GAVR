"""
Significance tests for GAVR vs baselines: paired bootstrap CI + McNemar.

Inputs are the per-sample JSONLs from experiments/vhs_gavr.py (same sids).
Typically base vs gavr (main claim) and top1 vs gavr (paper-baseline claim).

    experiments/agg_stats.py --model internvl-8b --hs 10 \
        --a-jsonl outputs/results/vhs_internvl-8b_base_hs10.jsonl \
        --b-jsonl outputs/results/vhs_internvl-8b_gavr_hs10_k5.jsonl
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load(path):
    rows = {}
    with open(path) as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            if 'error' not in r:
                rows[r['sid']] = r
    return rows


def correctness(rows, keys):
    for k in keys:
        if k in next(iter(rows.values())):
            return {sid: int(r.get(k) == 1) for sid, r in rows.items()}
    raise KeyError(f'no correctness field among {keys}')


def bootstrap_ci(a, b, n_boot=10000, seed=0):
    """CI of mean(b) - mean(a) over paired samples."""
    rng = np.random.default_rng(seed)
    sids = sorted(set(a) & set(b))
    x = np.array([b[s] - a[s] for s in sids], dtype=float)
    means = np.empty(n_boot)
    n = len(x)
    for i in range(n_boot):
        means[i] = x[rng.integers(0, n, n)].mean()
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def mcnemar(a, b):
    from scipy.stats import binomtest
    sids = sorted(set(a) & set(b))
    b01 = sum(1 for s in sids if a[s] == 0 and b[s] == 1)  # a wrong, b right
    b10 = sum(1 for s in sids if a[s] == 1 and b[s] == 0)
    n = b01 + b10
    p = binomtest(b01, n, 0.5).pvalue if n else 1.0
    return b01, b10, float(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--a-jsonl', required=True, help='baseline JSONL')
    ap.add_argument('--b-jsonl', required=True, help='treatment JSONL (GAVR)')
    ap.add_argument('--a-keys', nargs='+', default=['base_correct', 'correct'])
    ap.add_argument('--b-keys', nargs='+', default=['gavr_correct', 'correct'])
    ap.add_argument('--n-boot', type=int, default=10000)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    a_rows, b_rows = load(args.a_jsonl), load(args.b_jsonl)
    a = correctness(a_rows, args.a_keys)
    b = correctness(b_rows, args.b_keys)
    sids = sorted(set(a) & set(b))
    if not sids:
        raise SystemExit('no overlapping sids between the two JSONLs')

    acc_a = np.mean([a[s] for s in sids])
    acc_b = np.mean([b[s] for s in sids])
    lo, hi = bootstrap_ci(a, b, args.n_boot, args.seed)
    b01, b10, p = mcnemar(a, b)
    print(json.dumps({
        'n': len(sids),
        'acc_a': round(float(acc_a), 4),
        'acc_b': round(float(acc_b), 4),
        'delta': round(float(acc_b - acc_a), 4),
        'delta_ci95': [round(lo, 4), round(hi, 4)],
        'mcnemar': {'a_wrong_b_right': b01, 'a_right_b_wrong': b10,
                    'p_value': p},
    }, indent=2))


if __name__ == '__main__':
    main()
