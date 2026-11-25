"""
Synthetic text NIAH head extraction (paper Table 10 / Table 14 analogue).

Each sample is a K-item list of filler sentences, each carrying a unique
password at a random position.  The question asks for the password of item j;
attention from the question's last token to every candidate sentence gives a
text-side "pointer" score, exactly mirroring the image-side pipeline.

Saves the same h5 layout as save_vhs_head_outputs.py so the offline scanners
work unchanged (attn scores are (num_heads, K) instead of (num_heads, n_imgs)).

    data/save_text_niah_heads.py --model internvl-8b --n 500 --k-items 10
"""
import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import h5py
import numpy as np
import torch
from tqdm import tqdm

from utils.model_backend import ModelBackend
from utils.model_utils import ATTN_ROOT, MODEL_CONFIGS

FILLER = [
    "The old library on the hill kept a record of every visitor that summer.",
    "A gentle rain fell over the harbor while the fishermen repaired their nets.",
    "The committee reviewed the budget proposal and suggested minor revisions.",
    "During the festival, lanterns were hung along the narrow cobblestone street.",
    "The engineer measured the bridge once more before signing the final report.",
]


def make_sample(rng, k_items):
    words = "abcdefghijklmnopqrstuvwxyz0123456789"
    sentences, passwords = [], []
    for i in range(k_items):
        pw = ''.join(rng.choice(words) for _ in range(8))
        base = rng.choice(FILLER).split()
        pos = rng.randrange(len(base) + 1)
        base.insert(pos, f"password {pw}")
        sentences.append(' '.join(base))
        passwords.append(pw)
    target = rng.randrange(k_items)
    prompt = ('Below are numbered sentences.\n' +
              '\n'.join(f'Sentence {i}: {s}' for i, s in enumerate(sentences)) +
              f'\n\nQuestion: What is the password in sentence {target}? '
              'Answer with the password only.')
    return prompt, sentences, passwords, target


def find_span(ids_list, sent_ids):
    n = len(sent_ids)
    for i in range(len(ids_list) - n, -1, -1):
        if ids_list[i:i + n] == sent_ids:
            return i, i + n - 1
    return None


@torch.no_grad()
def run(backend, out_root, n=500, k_items=10, seed=0, max_new_tokens=16):
    os.makedirs(out_root, exist_ok=True)
    path = os.path.join(out_root, f'text_niah_k{k_items}.h5')
    done = 0
    if os.path.exists(path):
        with h5py.File(path, 'r') as f:
            done = len(f.keys())
    rng = random.Random(seed)
    f = h5py.File(path, 'a')
    try:
        for i in tqdm(range(done, n), initial=done, total=n):
            prompt, sentences, passwords, target = make_sample(rng, k_items)
            inputs = backend.build_inputs([], prompt)
            ids_list = inputs['ids_list']
            spans = []
            for s in sentences:
                ids = backend.tokenizer(' ' + s, add_special_tokens=False)['input_ids']
                span = find_span(ids_list, ids)
                if span is None:  # tokenizer context shifts; fall back to middle
                    mid = len(ids_list) // 2
                    span = (mid, mid)
                spans.append(span)
            last = len(ids_list) - 1

            qcat, kcat = {}, {}
            hooks = []
            for li in range(backend.num_layers):
                attn = backend.layers[li].self_attn

                def make_q(idx):
                    def fn(_m, _a, output):
                        x = output[0] if isinstance(output, tuple) else output
                        x = x.detach().float()
                        if x.dim() == 3 and x.shape[1] > 1:
                            x = x.reshape(1, x.shape[1], backend.num_heads,
                                          backend.head_dim).permute(0, 2, 1, 3)
                            if last < x.shape[2]:
                                qcat[idx] = x[0, :, last, :]
                    return fn

                def make_k(idx):
                    def fn(_m, _a, output):
                        x = output[0] if isinstance(output, tuple) else output
                        x = x.detach().float()
                        if x.dim() == 3 and x.shape[1] > 1:
                            x = x.reshape(1, x.shape[1], backend.num_kv_heads,
                                          backend.head_dim).permute(0, 2, 1, 3)
                            kcat[idx] = x[0]
                    return fn

                hooks.append(attn.q_proj.register_forward_hook(make_q(li)))
                hooks.append(attn.k_proj.register_forward_hook(make_k(li)))

            try:
                gen = backend.generate(inputs, max_new_tokens=max_new_tokens)
            finally:
                for h in hooks:
                    h.remove()

            scores = {}
            for li in range(backend.num_layers):
                if li not in qcat or li not in kcat:
                    continue
                q = qcat[li].cpu()
                k = kcat[li].cpu().repeat_interleave(backend.kv_groups, dim=0)
                w = torch.softmax(
                    torch.einsum('hd,hsd->hs', q, k) * backend.scale, dim=-1)
                s = np.zeros((backend.num_heads, k_items), dtype=np.float32)
                for j, (a, b) in enumerate(spans):
                    s[:, j] = w[:, a:b + 1].sum(dim=1).numpy()
                scores[li] = s

            gid = f'sample_{i}'
            if gid in f:
                del f[gid]
            grp = f.create_group(gid)
            grp.attrs['pos_idx'] = target
            grp.attrs['passwords'] = json.dumps(passwords)
            grp.attrs['target_password'] = passwords[target]
            grp.attrs['response'] = gen['response']
            grp.attrs['correct'] = int(passwords[target].lower()
                                       in gen['response'].lower())
            ag = grp.create_group('cattok_attn_scores')
            for li, s in scores.items():
                ag.create_dataset(f'layer_{li}', data=s,
                                  compression='gzip', compression_opts=4)
            f.flush()
    finally:
        f.close()
    print(f'done -> {path}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, choices=list(MODEL_CONFIGS.keys()))
    ap.add_argument('--n', type=int, default=500)
    ap.add_argument('--k-items', type=int, default=10)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--outputs-dir', default=None)
    args = ap.parse_args()
    out_root = args.outputs_dir or os.path.join(
        ATTN_ROOT, MODEL_CONFIGS[args.model]['model_dir'], 'text_niah')
    backend = ModelBackend(args.model)
    run(backend, out_root, args.n, args.k_items, args.seed)


if __name__ == '__main__':
    main()
