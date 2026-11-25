"""
Offline analyses on saved per-sample results (no GPU).

    experiments/offline_analysis.py --jsonl outputs/results/vhs_internvl-8b_gavr_hs10_k5.jsonl \
        [--base-jsonl ...] [--oracle-jsonl ...]

Reports
    * error cascade : P(GAVR correct | base wrong), pointer-failure recovery
    * margin calibration : GAVR accuracy per |margin| bucket
    * oracle coverage : GAVR acc / oracle acc (upper-bound utilisation)
"""
import argparse
import json
import sys
import os

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--jsonl', required=True, help='GAVR/fusion JSONL')
    ap.add_argument('--base-jsonl', default=None)
    ap.add_argument('--oracle-jsonl', default=None)
    ap.add_argument('--buckets', type=int, default=5)
    args = ap.parse_args()

    gavr = load(args.jsonl)
    out = {'n': len(gavr)}

    # GAVR accuracy / retrieval accuracy
    out['gavr_acc'] = float(np.mean([r.get('gavr_correct', r.get('correct', 0))
                                     for r in gavr.values()]))
    if any('gavr_retr_correct' in r for r in gavr.values()):
        out['gavr_r1'] = float(np.mean([r.get('gavr_retr_correct', 0)
                                        for r in gavr.values()]))

    # Error cascade vs base
    base = load(args.base_jsonl) if args.base_jsonl else {}
    if base:
        sids = sorted(set(base) & set(gavr))
        wrong = [s for s in sids if base[s].get('base_correct',
                                                base[s].get('correct', 0)) == 0]
        fixed = [s for s in wrong
                 if gavr[s].get('gavr_correct', gavr[s].get('correct', 0)) == 1]
        out['base_wrong'] = len(wrong)
        out['recovered'] = len(fixed)
        out['recovery_rate'] = round(len(fixed) / max(len(wrong), 1), 4)

    # Retrieval-failure recovery (pointer top-1 wrong but GAVR right)
    retr_fail = [r for r in gavr.values()
                 if r.get('retr_correct') == 0 and 'retr_correct' in r]
    if retr_fail:
        rec = sum(1 for r in retr_fail
                  if r.get('gavr_correct', r.get('correct', 0)) == 1)
        out['ptr_fail'] = len(retr_fail)
        out['ptr_fail_recovered'] = rec
        out['ptr_fail_recovery_rate'] = round(rec / len(retr_fail), 4)

    # Margin calibration
    rows = [r for r in gavr.values() if 'gavr_margin' in r]
    if rows:
        margins = np.array([abs(r['gavr_margin']) for r in rows])
        correct = np.array([r.get('gavr_correct', 0) for r in rows])
        qs = np.quantile(margins, np.linspace(0, 1, args.buckets + 1))
        buckets = []
        for i in range(args.buckets):
            lo, hi = qs[i], qs[i + 1]
            m = (margins >= lo) & (margins <= hi)
            if m.sum() == 0:
                continue
            buckets.append({'margin_range': [round(float(lo), 2),
                                             round(float(hi), 2)],
                            'n': int(m.sum()),
                            'acc': round(float(correct[m].mean()), 4)})
        out['margin_calibration'] = buckets

    # Oracle coverage
    if args.oracle_jsonl:
        oracle = load(args.oracle_jsonl)
        sids = sorted(set(oracle) & set(gavr))
        o = np.mean([oracle[s]['correct'] for s in sids])
        g = np.mean([gavr[s].get('gavr_correct', gavr[s].get('correct', 0))
                     for s in sids])
        out['oracle_acc'] = round(float(o), 4)
        out['oracle_coverage'] = round(float(g / o), 4) if o else None

    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
