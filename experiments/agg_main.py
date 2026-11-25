"""
Aggregate the main VHS results (base / top1 / gavr / oracle) into a table.

Scans outputs/results/vhs_*.jsonl and joins per-sample records by sid, so all
modes must have been run on the same (model, hs, limit). Prints a markdown
table and optionally writes JSON.

    experiments/agg_main.py --models internvl-8b internvl-4b qwen3vl-8b qwen3vl-4b \
        --hs 2 3 5 10 20 50
"""
import argparse
import glob
import json
import os
import re

RESULT_DIR = os.path.join(os.environ.get('ATTN_ROOT', 'outputs'), 'results')


def load_jsonl(path):
    rows = {}
    with open(path) as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            if 'error' in r:
                continue
            rows[r['sid']] = r
    return rows


def result_files(model, hs, mode, k=None):
    """Use the newest run layout, merging its shards in sample order."""
    stem = f'vhs_{model}_{mode}_hs{hs}'
    if k is not None:
        stem += f'_k{k}'
    name_re = re.compile(rf'{re.escape(stem)}(?:_s(\d+))?\.jsonl')
    matches = {}
    for path in glob.glob(os.path.join(RESULT_DIR, f'{stem}*.jsonl')):
        match = name_re.fullmatch(os.path.basename(path))
        if match and os.path.getsize(path) > 0:
            matches[path] = match.group(1)
    if not matches:
        return []
    newest = max(matches, key=os.path.getmtime)
    if matches[newest] is None:
        return [newest]
    shards = [path for path, start in matches.items() if start is not None]
    return sorted(shards, key=lambda path: int(matches[path]))


def rate(rows, key):
    vals = [r[key] for r in rows if key in r]
    return (sum(vals) / len(vals)) if vals else None


def summarize(model, hs, limit=None, k=5):
    out = {'model': model, 'hs': hs}
    ids = None
    for name in ('base', 'top1', 'gavr', 'oracle'):
        paths = result_files(model, hs, name, k if name == 'gavr' else None)
        if not paths:
            out[name] = None
            continue
        rows = {}
        for path in paths:
            rows.update(load_jsonl(path))
        if limit:
            rows = {k: v for k, v in list(rows.items())[:limit]}
        sids = set(rows)
        ids = sids if ids is None else (ids & sids)
        out[name + '_n'] = len(rows)
        out[name + '_file'] = ', '.join(os.path.basename(path) for path in paths)
        out[name + '_rows'] = rows
    if ids is None:
        return None
    n = len(ids)
    out['n'] = n
    if 'base_rows' in out:
        out['base_acc'] = rate([out['base_rows'][i] for i in ids],
                               'base_correct')
    if 'top1_rows' in out:
        rows = [out['top1_rows'][i] for i in ids]
        out['top1_acc'] = rate(rows, 'correct')
        out['ptr_r1'] = rate(rows, 'retr_correct')
    if 'gavr_rows' in out:
        rows = [out['gavr_rows'][i] for i in ids]
        out['gavr_acc'] = rate(rows, 'gavr_correct')
        out['gavr_r1'] = rate(rows, 'gavr_retr_correct')
        if out['gavr_acc'] is not None and out.get('top1_acc') is not None:
            out['delta_vs_top1'] = out['gavr_acc'] - out['top1_acc']
        if out['gavr_acc'] is not None and out.get('base_acc') is not None:
            out['delta_vs_base'] = out['gavr_acc'] - out['base_acc']
    if 'oracle_rows' in out:
        out['oracle_acc'] = rate([out['oracle_rows'][i] for i in ids],
                                 'correct')
    for k in ('base_rows', 'top1_rows', 'gavr_rows', 'oracle_rows'):
        out.pop(k, None)
    return out


def fmt(x):
    return f'{x:.3f}' if isinstance(x, float) else ('—' if x is None else str(x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', nargs='+',
                    default=['internvl-8b', 'internvl-4b', 'qwen3vl-8b',
                             'qwen3vl-4b'])
    ap.add_argument('--hs', nargs='+', type=int, default=[10])
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--k', type=int, default=5)
    ap.add_argument('--json-out', default=None)
    args = ap.parse_args()

    results = []
    print('| model | N | n | Base Joint | Top-1 | GAVR | Δ vs Top-1 | ptr R@1 | GAVR R@1 | Oracle |')
    print('|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
    for model in args.models:
        for hs in args.hs:
            r = summarize(model, hs, args.limit, args.k)
            if r is None:
                print(f'| {model} | {hs} | — | — | — | — | — | — | — | — |')
                continue
            results.append(r)
            print('| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |'.format(
                model, hs, r['n'], fmt(r.get('base_acc')), fmt(r.get('top1_acc')),
                fmt(r.get('gavr_acc')),
                ('{:+.3f}'.format(r['delta_vs_top1'])
                 if r.get('delta_vs_top1') is not None else '—'),
                fmt(r.get('ptr_r1')), fmt(r.get('gavr_r1')),
                fmt(r.get('oracle_acc'))))
    if args.json_out:
        with open(args.json_out, 'w') as f:
            json.dump(results, f, indent=2)
        print(f'\njson -> {args.json_out}')


if __name__ == '__main__':
    main()
