"""GPU runtime verification of the RockAndEgg-v3 design claims (2x T4).

Run: .venv/bin/python gpu_verify.py > gpu_verify_results.json
The checkpoint is selected by RAE_BACKBONE (local directory or Hub id).
"""

import copy
import json
import os
import time

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

MODEL = os.environ.get("RAE_BACKBONE", "moeshawky/RockAndEgg-313M-v2")
DEV = "cuda:0"
S = 512
SEED = 20261116
torch.manual_seed(SEED)

results = {
    "model_path": MODEL,
    "devices": torch.cuda.device_count(),
    "gpu_names": [
        torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
    ],
}


def kv_bytes(cache):
    total = 0
    layers = getattr(cache, "layers", None)
    if layers is not None:
        for lay in layers:
            for name in ("keys", "values"):
                t = getattr(lay, name, None)
                if t is not None and torch.is_tensor(t):
                    total += t.numel() * t.element_size()
    else:  # fallback for older layouts
        for name in ("key_cache", "value_cache"):
            for t in getattr(cache, name, None) or []:
                if torch.is_tensor(t):
                    total += t.numel() * t.element_size()
    return total


def load(dtype, attn="sdpa"):
    # transformers >= 4.56: `dtype` replaces deprecated `torch_dtype`.
    return AutoModel.from_pretrained(MODEL, dtype=dtype, attn_implementation=attn).to(
        DEV
    ).eval(), "dtype"


def make_inputs(tok, state_len, options_per_q, q_count, dev):
    state_text = "The customer asked about billing and refunds. " * 40
    state_ids = tok(
        state_text,
        add_special_tokens=False,
        return_tensors="pt",
        truncation=True,
        max_length=state_len,
    )["input_ids"].to(dev)
    if state_ids.shape[1] < state_len:
        pad = torch.full(
            (1, state_len - state_ids.shape[1]), tok.eos_token_id, device=dev
        )
        state_ids = torch.cat([state_ids, pad], dim=1)
    suffix_texts, groups = [], []
    q = 0
    for n in options_per_q:
        groups.append([])
        for j in range(n):
            suffix_texts.append(
                f"choice: Which team should handle this? [SEP] option_{j}: "
                f"billing refunds technical outages account access {q}. [SEP]"
            )
            groups[-1].append(len(suffix_texts) - 1)
        q += 1
    enc = tok(suffix_texts, add_special_tokens=False, padding=True, return_tensors="pt")
    ids, mask = enc["input_ids"].to(dev), enc["attention_mask"].to(dev)
    return state_ids, ids, mask, groups


def gather_last(hidden, mask):
    idx = mask.long().cumsum(-1).argmax(-1)
    return hidden[torch.arange(hidden.shape[0], device=hidden.device), idx]


def build_head(d, dev):
    torch.manual_seed(SEED)
    layer = nn.TransformerEncoderLayer(
        d,
        nhead=16,
        dim_feedforward=2048,
        dropout=0.0,
        activation="gelu",
        batch_first=True,
        norm_first=True,
    )
    return (
        nn.TransformerEncoder(layer, 2).to(dev).eval(),
        nn.Sequential(nn.Linear(d, 256), nn.SiLU(), nn.Linear(256, 1)).to(dev).eval(),
        nn.Embedding(3, d).to(dev).eval(),
    )


def run_case(model, head, state_ids, suf_ids, suf_mask, groups, capture=None):
    mixer, scorer, qemb = head
    with torch.inference_mode():
        so = model(input_ids=state_ids, use_cache=True)
        state_hidden = so.last_hidden_state
        cache = so.past_key_values
        cache_bytes = kv_bytes(cache)
        M = suf_ids.shape[0]
        fork = copy.deepcopy(cache)
        ret = fork.batch_repeat_interleave(M)
        S = state_ids.shape[1]
        ones = torch.ones((M, S), dtype=suf_mask.dtype, device=suf_mask.device)
        full_mask = torch.cat([ones, suf_mask], dim=1)
        so2 = model(
            input_ids=suf_ids,
            attention_mask=full_mask,
            past_key_values=fork,
            use_cache=False,
        )
        if capture is not None:
            capture["position_ids"] = capture.get("position_ids")
        opt = gather_last(so2.last_hidden_state, suf_mask)
        state_vec = torch.nn.functional.normalize(state_hidden[:, -1, :], dim=-1)
        out = []
        for g in groups:
            h = opt[g].unsqueeze(0)
            prior = (
                torch.nn.functional.normalize(h.squeeze(0), dim=-1) @ state_vec.T
            ).squeeze(-1)
            h = h + qemb(torch.tensor([0], device=h.device)).unsqueeze(1)
            h = mixer(h)
            out.append(scorer(h).squeeze(0).squeeze(-1) + prior)
        return {
            "logits": out,
            "cache_bytes": cache_bytes,
            "fork_return": ret,
            "fork_batch": fork.get_seq_length(),
            "state_last_finite": bool(torch.isfinite(state_vec).all()),
        }


def permuted(ids, mask, groups, seed):
    """Permute only the ORDER in which a question's option rows are presented
    to the mixer. Row contents and backbone inputs are unchanged; this is the
    property the 'option-order invariance' claim is about."""
    g = torch.Generator().manual_seed(seed)
    new_groups = []
    for grp in groups:
        p = torch.randperm(len(grp), generator=g).tolist()
        new_groups.append([grp[i] for i in p])
    return ids, mask, new_groups


def maxdiff(a, b):
    return max(float((x - y).abs().max()) for x, y in zip(a, b))


# ---------------------------------------------------------------- main
tok = AutoTokenizer.from_pretrained(MODEL)
d = 1024

for dtype_name, dtype in [("fp32", torch.float32), ("fp16", torch.float16)]:
    model, kwarg = load(dtype)
    head = build_head(d, DEV)
    state_ids, suf_ids, suf_mask, groups = make_inputs(tok, S, [3, 2, 4], 3, DEV)

    capture = {}

    def cap_hook(mod, args):
        if len(args) > 1 and args[1] is not None:
            capture["position_ids"] = args[1].detach().clone()

    hook = model.rotary_emb.register_forward_pre_hook(cap_hook)
    base = run_case(model, head, state_ids, suf_ids, suf_mask, groups, capture)
    hook.remove()

    pid = capture.get("position_ids")
    perm_ids, perm_mask, perm_groups = permuted(suf_ids, suf_mask, groups, SEED + 1)
    perm = run_case(model, head, state_ids, perm_ids, perm_mask, perm_groups)
    # unpermute logits to original option order for comparison
    restored = []
    for orig_g, new_g, lg in zip(groups, perm_groups, perm["logits"]):
        idx_map = {row: i for i, row in enumerate(new_g)}
        restored.append(torch.stack([lg[idx_map[row]] for row in orig_g]))
    # question order permutation (reverse the list of question groups)
    qrev_ids, qrev_mask = suf_ids.clone(), suf_mask.clone()
    rev_groups = []
    for grp in reversed(groups):
        rev_groups.append(grp)
    qrev = run_case(model, head, state_ids, qrev_ids, qrev_mask, rev_groups)
    qrestored = []
    for orig_g, lg in zip(groups, list(reversed(qrev["logits"]))):
        qrestored.append(lg)

    # latency: cold + warm CUDA-event timing of one full case
    for _ in range(2):
        run_case(model, head, state_ids, suf_ids, suf_mask, groups)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(10):
        run_case(model, head, state_ids, suf_ids, suf_mask, groups)
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) * 1000 / 10

    # Direct isolation probes: (a) state prefill must be independent of the
    # suffix batch; (b) changing one question's options must not change others.
    with torch.inference_mode():
        state_a = model(input_ids=state_ids, use_cache=True).last_hidden_state
    alt_ids = suf_ids.clone()
    for row in groups[0]:
        alt_ids[row] = suf_ids[row].flip(-1)  # different option content, same mask
    with torch.inference_mode():
        alt_out = run_case(model, head, state_ids, alt_ids, suf_mask, groups)
        state_b = model(input_ids=state_ids, use_cache=True).last_hidden_state
    state_independence_diff = float((state_a - state_b).abs().max())
    leakage_diff = maxdiff(base["logits"][1:], alt_out["logits"][1:])
    changed_q0_diff = float((base["logits"][0] - alt_out["logits"][0]).abs().max())

    results[dtype_name] = {
        "load_kwarg": kwarg,
        "attn_implementation": model.config._attn_implementation,
        "kv_cache_bytes": base["cache_bytes"],
        "kv_cache_MiB": round(base["cache_bytes"] / 2**20, 3),
        "fork_return": base["fork_return"],
        "fork_seq_length": base["fork_batch"],
        "position_ids_shape": list(pid.shape) if pid is not None else None,
        "position_ids_first_last": [int(pid[0, 0]), int(pid[0, -1])]
        if pid is not None
        else None,
        "position_ids_same_all_rows": bool(pid.shape[0] == 1)
        if pid is not None
        else None,
        "state_last_finite": base["state_last_finite"],
        "option_permutation_max_abs_diff": maxdiff(base["logits"], restored),
        "question_permutation_max_abs_diff": maxdiff(base["logits"], qrestored),
        "state_independence_max_abs_diff": state_independence_diff,
        "cross_question_leakage_max_abs_diff": leakage_diff,
        "changed_question0_did_change_logits": changed_q0_diff,
        "full_case_latency_ms": round(ms, 3),
    }
    del model, head
    torch.cuda.empty_cache()

# flash-attention attempt on T4 (Turing)
try:
    fam, _ = load(torch.float16, attn="flash_attention_2")
    results["flash_attention_2"] = {
        "loaded": True,
        "impl": fam.config._attn_implementation,
    }
    del fam
except Exception as e:
    results["flash_attention_2"] = {
        "loaded": False,
        "error": f"{type(e).__name__}: {str(e)[:600]}",
    }

print(json.dumps(results, indent=2))
