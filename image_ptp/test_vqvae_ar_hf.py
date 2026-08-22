"""Check the wrapped AR teacher against the original `models/ar.py`.

The wrapper splits `MultiheadAttention`'s packed projections apart and rebuilds
the layer loop, so it has to be shown to compute the same thing. Only after that
does a difference in training results mean anything.

Usage:
    PYTHONPATH=src:. python image_ptp/test_vqvae_ar_hf.py --repo ~/ptp-vqvae
"""
import argparse
import sys
from pathlib import Path

import torch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", type=str, default="/home/mengy13/ptp-vqvae")
    p.add_argument("--ckpt", type=str, default="checkpoints/ar_mnist_raster.pt")
    p.add_argument("--batch-size", type=int, default=4)
    return p.parse_args()


def report(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}{'  ' + detail if detail else ''}")
    return ok


def main():
    args = parse_args()
    torch.manual_seed(0)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    sys.path.insert(0, args.repo)
    import os
    os.chdir(args.repo)
    from models.ar import ARTransformer
    from image_ptp.vqvae_ar_hf import build

    model, meta = build(args.ckpt, device=device, dtype=torch.float32)
    reference = ARTransformer(
        num_codes=meta["num_codes"], d_model=meta["d_model"], nhead=meta["nhead"],
        num_layers=meta["num_layers"], dim_feedforward=meta["dim_feedforward"],
        max_seq_len=meta["max_seq_len"],
    ).to(device).eval()
    reference.load_state_dict(meta["ar"])

    b, seq_len = args.batch_size, meta["h"] * meta["w"]
    bos = meta["num_codes"]
    codes = torch.randint(0, bos, (b, seq_len - 1), device=device)
    input_ids = torch.cat([torch.full((b, 1), bos, device=device), codes], dim=1)
    passed = []

    ref_logits = reference(input_ids)
    ours = model(input_ids=input_ids).logits
    delta = (ours - ref_logits).abs().max().item()
    passed.append(report("matches ARTransformer", delta < 1e-3, f"max|d|={delta:.2e}"))

    # The teacher must be deterministic: PTP evaluates it while the module is in
    # train mode, so any surviving dropout would put noise into the bin edges.
    model.train()
    a = model(input_ids=input_ids).logits
    c = model(input_ids=input_ids).logits
    model.eval()
    delta = (a - c).abs().max().item()
    passed.append(report("no dropout in train mode", delta == 0.0, f"max|d|={delta:.2e}"))

    from transformers import DynamicCache
    cache = DynamicCache()
    split = seq_len // 2
    model(input_ids=input_ids[:, :split], past_key_values=cache, use_cache=True)
    stepped = model(input_ids=input_ids[:, split:], past_key_values=cache,
                    use_cache=True).logits
    delta = (stepped - ours[:, split:]).abs().max().item()
    passed.append(report("cached continuation matches", delta < 1e-3, f"max|d|={delta:.2e}"))

    cache = DynamicCache()
    model(input_ids=input_ids[:, :split], past_key_values=cache, use_cache=True)
    via_embeds = model(
        inputs_embeds=model.tok_emb(input_ids[:, split:]),
        position_ids=torch.arange(split, seq_len, device=device),
        past_key_values=cache, use_cache=False).logits
    delta = (via_embeds - ours[:, split:]).abs().max().item()
    passed.append(report("inputs_embeds path matches", delta < 1e-3, f"max|d|={delta:.2e}"))

    from torch.nn.attention.flex_attention import create_block_mask
    block_mask = create_block_mask(lambda b_, h_, q, kv: q >= kv, B=None, H=None,
                                   Q_LEN=seq_len, KV_LEN=seq_len, device=device)
    flex = model(input_ids=input_ids, attention_mask=block_mask).logits
    delta = (flex - ours).abs().max().item()
    passed.append(report("BlockMask reproduces causal", delta < 5e-3, f"max|d|={delta:.2e}"))

    from torch import nn
    names = {n.split(".")[-1] for n, m in model.named_modules() if isinstance(m, nn.Linear)}
    wanted = {"q_proj", "k_proj", "v_proj", "o_proj", "linear1", "linear2"}
    passed.append(report("LoRA targets addressable", wanted <= names,
                         f"missing={sorted(wanted - names)}"))

    print(f"\n{sum(passed)}/{len(passed)} checks passed")
    return 0 if all(passed) else 1


if __name__ == "__main__":
    raise SystemExit(main())
