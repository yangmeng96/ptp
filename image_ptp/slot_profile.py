"""Where inside a block does an O-PTP student stop being right?

`correct` collapses a block into one number, which cannot tell three very
different failures apart:

  uniform      every slot is right about 1/8 of the time -- the student is a
               weak predictor everywhere and the block structure is irrelevant
  cliff        slot 0 is nearly right and later slots collapse -- the student
               cannot condition on tokens it has not seen, which is the
               parallel-prediction problem itself
  contagion    later slots are only wrong *after* an earlier slot was wrong --
               each slot is fine on its own but errors propagate, because slot k
               must infer t_i..t_{k-1} from u rather than read them

The three call for different fixes, so measure them apart. `p(correct | prefix
of the block was correct)` separates cliff from contagion: under contagion the
conditional stays flat while the marginal decays.

Bin width is reported alongside, since a slot whose teacher bin is narrow is
one where u has to be read precisely, and that is where a student that has only
half-learned the inverse CDF should fail first.

Usage:
    PYTHONPATH=src:. python image_ptp/slot_profile.py \
        --ar-ckpt ~/ptp-vqvae/checkpoints/ar_cifar10_raster.pt \
        --data ~/ptp-image-results/cifar_tokens.pt \
        --ckpt ~/ptp-image-exp/cifar-S32/last.ckpt --block-len 8
"""
import argparse
import json
from pathlib import Path

import torch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ar-ckpt", type=str, default=None)
    p.add_argument("--data", type=str, required=True)
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--mode", type=str, default="optp", choices=["optp", "cptp"])
    p.add_argument("--gated", action="store_true")
    p.add_argument("--num-layers", type=int, default=8)
    p.add_argument("--backbone", type=str, default="vqvae_ar",
                   choices=["vqvae_ar", "llamagen"])
    p.add_argument("--llamagen-root", type=str, default="/home/mengy13/LlamaGen")
    p.add_argument("--gpt-model", type=str, default="GPT-B")
    p.add_argument("--gpt-ckpt", type=str, default=None)
    p.add_argument("--adapter-name", type=str, default="linear_interpolation",
                   choices=["linear_interpolation", "binary", "quarter_cos",
                            "sawtooth", "round"],
                   help="must match the checkpoint: each class keys its u "
                        "embedding differently, and a finer variant also "
                        "changes the parameter's shape")
    p.add_argument("--adapter-kwargs", type=str, default=None,
                   help='JSON, e.g. {"num_embeddings": 129}')
    p.add_argument("--block-len", type=int, default=8)
    p.add_argument("--split", type=str, default="val",
                   choices=["val", "all"],
                   help="'all' for a separately prepared test file, whose "
                        "sequences are unseen in their entirety")
    p.add_argument("--val-split", type=int, default=256)
    p.add_argument("--images", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--first-slot-value", type=float, default=0.5)
    p.add_argument("--bos", type=int, default=512)
    p.add_argument("--code-vocab", type=int, default=16384)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


@torch.no_grad()
def profile(model, tokens, first, left, right, block_len, mode, first_slot,
            device, batch_size, shuffle_u, seed):
    """Return per-slot hit counts, plus the same split by block position."""
    torch.manual_seed(seed)
    seq_len = tokens.shape[1]
    starts = list(range(1, seq_len - block_len + 1, block_len))
    hit = torch.zeros(block_len)
    n = torch.zeros(block_len)
    # Conditioned on every earlier slot in the same block being right.
    hit_cond = torch.zeros(block_len)
    n_cond = torch.zeros(block_len)
    width_ok, width_bad = [], []
    # Accuracy by where the block sits in the sequence.
    by_start_hit = torch.zeros(len(starts))
    by_start_n = torch.zeros(len(starts))

    for s in range(0, tokens.shape[0], batch_size):
        t = tokens[s:s + batch_size].to(device)
        l = left[s:s + batch_size].to(device)
        r = right[s:s + batch_size].to(device)
        ids = torch.cat([first[s:s + batch_size].to(device)[:, None], t], 1)
        for bi, start in enumerate(starts):
            span = min(block_len, seq_len - start + 1)
            lo = l[:, start - 1:start - 1 + span]
            hi = r[:, start - 1:start - 1 + span]
            u = lo + (hi - lo) * torch.rand(lo.shape, device=device)
            if shuffle_u:
                u = u[torch.randperm(u.shape[0], device=device)]
            if mode == "cptp":
                fed = torch.empty_like(u)
                fed[:, 0] = first_slot
                fed[:, 1:] = u[:, :-1]
            else:
                fed = u
            _, comp = model(input_ids=ids[:, :start], auxiliaries=fed)
            logits = comp.logits[:, :span].float()
            if mode == "cptp":
                cdf = torch.softmax(logits, -1).cumsum(-1)
                pred = torch.searchsorted(cdf.contiguous(),
                                          u.unsqueeze(-1).contiguous()).squeeze(-1)
                pred = pred.clamp(max=logits.shape[-1] - 1)
            else:
                pred = logits.argmax(-1)

            ok = (pred == ids[:, start:start + span])
            hit[:span] += ok.sum(0).cpu()
            n[:span] += ok.shape[0]
            by_start_hit[bi] += ok.float().mean(1).sum().cpu()
            by_start_n[bi] += ok.shape[0]

            # "every earlier slot right" -- slot 0 is unconditional by definition
            prefix_ok = torch.ones_like(ok[:, :1])
            run = torch.cat([prefix_ok, ok[:, :-1].cumprod(1).bool()], 1)
            hit_cond[:span] += (ok & run).sum(0).cpu()
            n_cond[:span] += run.sum(0).cpu()

            if not shuffle_u:
                w = (hi - lo)
                width_ok.append(w[ok].cpu())
                width_bad.append(w[~ok].cpu())

    out = {"hit": hit, "n": n, "hit_cond": hit_cond, "n_cond": n_cond,
           "by_start": (by_start_hit, by_start_n, starts)}
    if width_ok:
        out["width_ok"] = torch.cat(width_ok)
        out["width_bad"] = torch.cat(width_bad)
    return out



def first_positions(payload, n, args):
    """The token that opens each sequence.

    The VQ-VAE arms prepend one shared BOS. LlamaGen instead carries the class
    label in the extended vocabulary, so the opening token differs per sample and
    a constant would condition every sequence on class 0.
    """
    if args.backbone == "llamagen":
        labels = payload["labels"][:n].long()
        return labels + args.code_vocab
    return torch.full((n,), args.bos, dtype=torch.long)


def main():
    args = parse_args()
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    from ptp.transformer import MixedTransformerModel
    from image_ptp.gated_full import GatedFullTransformerModel

    if args.backbone == "llamagen":
        from image_ptp.llamagen_hf import build as build_lg
        root = Path(args.llamagen_root).expanduser()
        inner, _ = build_lg(root, args.gpt_model,
                            args.gpt_ckpt or str(root / "pretrained_models/c2i_B_256.pt"),
                            dtype=torch.float32, device=device)
    else:
        from image_ptp.vqvae_ar_hf import build_module
        inner = build_module(args.ar_ckpt, device=device, dtype=torch.float32,
                             num_layers=args.num_layers)
    cls = GatedFullTransformerModel if args.gated else MixedTransformerModel
    model = cls(model_id=inner, dtype=torch.float32,
                adapter_name=args.adapter_name,
                adapter_kwargs=(json.loads(args.adapter_kwargs)
                                if args.adapter_kwargs else None),
                attn_implementation="flex_attention").to(device).eval()
    state = torch.load(Path(args.ckpt).expanduser(), map_location="cpu",
                       weights_only=False)["state_dict"]
    missing, unexpected = model.load_state_dict(
        {k[len("model."):]: v for k, v in state.items() if k.startswith("model.")},
        strict=False)
    stale = [k for k in list(missing) + list(unexpected) if "u_embed" in k]
    assert not stale, f"auxiliary embedding did not load: {stale[:3]}"

    payload = torch.load(Path(args.data).expanduser(), map_location="cpu")
    total = payload["tokens"].shape[0]
    split = total if args.split == "all" else min(args.val_split, total // 4)
    hi = min(split, args.images)
    tokens = payload["tokens"][:hi].long()
    left, right = payload["left_bin_edges"][:hi], payload["right_bin_edges"][:hi]
    first = first_positions(payload, hi, args)
    print(f"{hi} held-out sequences of {tokens.shape[1]} tokens, "
          f"block {args.block_len}, mode {args.mode}")

    real = profile(model, tokens, first, left, right, args.block_len,
                   args.mode, args.first_slot_value, device, args.batch_size,
                   False, args.seed)
    shuf = profile(model, tokens, first, left, right, args.block_len,
                   args.mode, args.first_slot_value, device, args.batch_size,
                   True, args.seed)

    print("\nslot  accuracy  shuffled  lift   p(ok | block prefix ok)  bin width")
    for k in range(args.block_len):
        acc = (real["hit"][k] / real["n"][k]).item()
        sh = (shuf["hit"][k] / max(shuf["n"][k], 1)).item()
        cond = (real["hit_cond"][k] / max(real["n_cond"][k], 1)).item()
        print(f" {k:2d}   {acc:7.4f}  {sh:7.4f}  {acc / max(sh, 1e-9):5.2f}x  "
              f"        {cond:7.4f}          -")

    w_ok, w_bad = real.get("width_ok"), real.get("width_bad")
    if w_ok is not None and w_ok.numel() and w_bad.numel():
        print(f"\nteacher bin width where the student is right: "
              f"median {w_ok.median():.4f}  (n={w_ok.numel()})")
        print(f"                              and where wrong: "
              f"median {w_bad.median():.4f}  (n={w_bad.numel()})")

    bh, bn, starts = real["by_start"]
    print("\nblock start position -> accuracy")
    step = max(1, len(starts) // 8)
    for i in range(0, len(starts), step):
        print(f"  t={starts[i]:4d}  {(bh[i] / bn[i]).item():.4f}")


if __name__ == "__main__":
    main()
