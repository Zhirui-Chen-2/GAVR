"""
GAVR core: Grounded Anchor-based Visual Retrieval.

Pipeline:

    1. Pointer retrieval   -- the pointer-head attention over the N candidate
                              images (cached in the VHS h5 by
                              data/save_vhs_head_outputs.py) gives per-image
                              scores; take top-k candidates.
    2. Anchor verification -- parse the anchor phrase from the VHS question
                              ("... in the image with a <anchor> ..."), ask the
                              model "Is there a <anchor> in this image?" for
                              every candidate and use P(yes) from the first
                              decode step to re-rank.
    3. Single-image answer -- answer the original yes/no question using only
                              the verified image (focused re-read).
    4. Optional fusion     -- keep the Base-Joint answer when its logit margin
                              exceeds k x the single-image margin.

Everything here is family-agnostic: models are accessed through
utils.model_backend.ModelBackend so InternVL / Qwen3-VL / LLaVA-OV all share
one code path.
"""
import json
import os
import random
import re

import h5py
import numpy as np
import torch
from PIL import Image

from utils.model_utils import COCO_ROOT, DATA_ROOT, ATTN_ROOT, MODEL_CONFIGS

# ── Pointer heads ─────────────────────────────────────────────────────────────
# 4B/8B InternVL + Qwen3-VL: L21H10 (paper Table 1 reproduced on both families).
# InternVL-14B: L24H0 is the reproduced MCQ pointer head.  Its VHS attention
# peak sits deeper (L26H36) -- pass --pointer-layer/--pointer-head to override.
POINTER_HEADS = {
    'internvl-8b':  (21, 10),
    'internvl-4b':  (21, 10),
    'qwen3vl-8b':   (21, 10),
    'qwen3vl-4b':   (21, 10),
    'internvl-14b': (24, 0),
}

VHS_DATA_ROOT = os.path.join(DATA_ROOT, 'single_needle', 'VHs_large')

# "Is there a butterfly in the image with a ladybug?" -> anchor = "ladybug"
_ANCHOR_RE = re.compile(
    r'image with (?:a |an )?(.+?)(?=\s*(?:[,.?;]|$| and | while | but | that )|$)',
    re.IGNORECASE)
_NEEDLE_RE = re.compile(r'is there (?:a |an )?(.+?) in the image',
                        re.IGNORECASE)

VERIFY_PROMPT = 'Is there {phrase} in this image? Answer yes or no.'
ANSWER_HINT = 'Answer yes or no.'


# ── Question parsing ──────────────────────────────────────────────────────────

def parse_anchor(question):
    """Return the anchor phrase (the object that marks the target image)."""
    m = _ANCHOR_RE.search(question or '')
    return m.group(1).strip() if m else None


def parse_needle(question):
    """Return the needle phrase ('Is there a <needle> ...')."""
    m = _NEEDLE_RE.search(question or '')
    return m.group(1).strip() if m else None


# ── VHS dataset loading (must stay identical to save_vhs_head_outputs.py) ─────

def uniquify_ids(samples):
    """Multi-needle jsons contain duplicated ids (yes/no variants); append '#k'."""
    seen = {}
    for s in samples:
        k = seen.get(s['id'], 0)
        seen[s['id']] = k + 1
        if k:
            s['id'] = f"{s['id']}#{k}"
    return samples


def build_haystack_map(samples):
    """Sequential seed-42 shuffle over the FULL json order (resume-safe)."""
    rng = random.Random(42)
    result = {}
    for s in samples:
        hs_list = list(s['pos_image']) + list(s['neg_image'])
        rng.shuffle(hs_list)
        result[s['id']] = hs_list
    return result


def load_vhs_samples(hs, data_root=None):
    """Return (samples, haystack_map) for visual_haystack_{hs}.json."""
    data_root = data_root or VHS_DATA_ROOT
    path = os.path.join(data_root, f'visual_haystack_{hs}.json')
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    samples = uniquify_ids(json.load(open(path)))
    return samples, build_haystack_map(samples)


def sample_gt(sample):
    return sample['conversations'][1]['value'].strip().lower()


def pos_indices(sample, haystack):
    return [haystack.index(p) for p in sample['pos_image']]


def image_abspath(rel):
    return rel if os.path.isabs(rel) else os.path.join(COCO_ROOT, rel)


def load_pil(rel):
    return Image.open(image_abspath(rel)).convert('RGB')


# ── Cached pointer attention ──────────────────────────────────────────────────

def h5_path(model_key, hs, attn_root=None):
    root = attn_root or ATTN_ROOT
    md = MODEL_CONFIGS[model_key]['model_dir']
    return os.path.join(root, md, 'vhs', f'vhs_hs{hs}.h5')


def read_attn_scores(path, sid, layer, phase='cattok'):
    """(num_heads, n_images) attention mass per image at the anchor token."""
    with h5py.File(path, 'r') as f:
        return f[f'sample_{sid}'][f'{phase}_attn_scores'][f'layer_{layer}'][:]


def pointer_scores(path, sid, layer, head, phase='cattok'):
    return read_attn_scores(path, sid, layer, phase)[head]


def pointer_topk(scores, k):
    return list(np.argsort(-np.asarray(scores))[:int(k)])


# ── Yes/no scoring through the backend ───────────────────────────────────────

def _yn_logits(backend, first_logits):
    if first_logits is None:
        return None, None
    return (float(first_logits[backend.yes_id]),
            float(first_logits[backend.no_id]))


def _yn_margin(first_logits, backend):
    ly, ln = _yn_logits(backend, first_logits)
    if ly is None:
        return 0.0
    return abs(ly - ln)


@torch.no_grad()
def answer(backend, image_rels, question, max_new_tokens=8):
    """Ask a yes/no question on the given images.

    Returns dict(pred, correct-able margin, response, p_yes).
    pred = parsed yes/no, falling back to the constrained argmax of the
    first-step yes/no logits (the paper's parser).
    """
    from utils.model_utils import parse_yes_no
    pils = [load_pil(r) for r in image_rels] if image_rels else []
    prompt = question if question.rstrip().endswith(('?', '.')) else question + '?'
    inputs = backend.build_inputs(pils, prompt)
    gen = backend.generate(inputs, max_new_tokens=max_new_tokens,
                           capture_first_logits=True)
    response = gen['response']
    pred = parse_yes_no(response)
    ly, ln = _yn_logits(backend, gen['first_logits'])
    if pred is None and ly is not None:
        pred = 'yes' if ly > ln else 'no'
    margin = _yn_margin(gen['first_logits'], backend)
    if ly is None:
        p_yes = None
    else:
        p_yes = float(torch.softmax(torch.tensor([ly, ln]), dim=0)[0])
    return {'pred': pred, 'margin': margin, 'response': response,
            'p_yes': p_yes}


@torch.no_grad()
def p_yes(backend, image_rel, phrase, max_new_tokens=4):
    """P(yes) of 'Is there <phrase> in this image?' for one candidate image."""
    pils = [load_pil(image_rel)]
    inputs = backend.build_inputs(pils, VERIFY_PROMPT.format(phrase=phrase))
    gen = backend.generate(inputs, max_new_tokens=max_new_tokens,
                           capture_first_logits=True)
    ly, ln = _yn_logits(backend, gen['first_logits'])
    if ly is None:
        return 0.0
    return float(torch.softmax(torch.tensor([ly, ln]), dim=0)[0])


def verify_candidates(backend, haystack, cand_idx, anchor):
    """P(yes) for every candidate index, in candidate order."""
    return [p_yes(backend, haystack[i], anchor) for i in cand_idx]


# ── Fusion gate ───────────────────────────────────────────────────────────────

def fuse_answer(base, gavr, k=10.0):
    """Keep the Base-Joint answer only when its margin is k x the GAVR margin.

    This reproduces the 'base if mj > k * ms else GAVR' gate from the logs;
    the offline grid search found k = 10 (i.e. tautologically trust GAVR) to
    be optimal on every model, so the default changes nothing in practice.
    """
    if base['pred'] is None:
        return gavr
    if base['margin'] > k * max(gavr['margin'], 1e-6):
        return base
    return gavr
