"""Score an O-PTP or a C-PTP student under its own decoding rule.

The two differ in how a token comes out of the block, and scoring one with the
other's rule silently measures nothing:

  O-PTP   slot k is given u_k and its output is meant to be one-hot; the token
          is the argmax.
  C-PTP   slot k is given u_{k-1} and its output is meant to be the teacher's
          distribution; the token is Pick(u_k; P_k), computed here rather than
          by the network.

Both are then scored the same way -- how many tokens are right before the first
mistake, which is what a speculative decoder can keep -- and both are compared
against a run with the auxiliaries taken from other sequences, which is the
floor a student that ignores them would reach.

Usage:
    PYTHONPATH=src:. python image_ptp/eval_cptp.py --mode cptp \
        --ckpt ~/ptp-image-exp/mnist-C7/last.ckpt --block-len 7
"""
import argparse
import json
from pathlib import Path

import torch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ar-ckpt", type=str,
                   default="/home/mengy13/ptp-vqvae/checkpoints/ar_mnist_raster.pt")
    p.add_argument("--data", type=str,
                   default="/home/mengy13/ptp-image-results/mnist_tokens.pt")
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--mode", type=str, default="optp", choices=["optp", "cptp"])
    p.add_argument("--gated", action="store_true")
    p.add_argument("--num-layers", type=int, default=8)
    p.add_argument("--block-len", type=int, default=7)
    p.add_argument("--images", type=int, default=1024)
    p.add_argument("--split", type=str, default="val",
                   choices=["val", "train", "all"],
                   help="ImageTokenDataModule holds out the FIRST val_split "
                        "sequences, so reading tokens[:images] mixes the two and "
                        "flatters late checkpoints that have started memorising. "
                        "Use 'all' with a separately prepared test file, where "
                        "every sequence is already unseen")
    p.add_argument("--val-split", type=int, default=256,
                   help="must match the value the run trained with")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--first-slot-value", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default=None)
    return p.parse_args()


def leading_run(correct):
    wrong = (~correct).float()
    return (wrong.cumsum(dim=1) == 0).sum(dim=1)


@torch.no_grad()
def score(model, tokens, left, right, bos, block_len, mode, first_slot,
          device, batch_size, shuffle_u=False, seed=0):
    runs, hits, total = [], 0, 0
    torch.manual_seed(seed)
    seq_len = tokens.shape[1]
    starts = list(range(1, seq_len - block_len + 1, block_len))
    for s in range(0, tokens.shape[0], batch_size):
        t = tokens[s:s + batch_size].to(device)
        l = left[s:s + batch_size].to(device)
        r = right[s:s + batch_size].to(device)
        ids = torch.cat([torch.full((t.shape[0], 1), bos, device=device), t], 1)
        for start in starts:
            span = min(block_len, seq_len - start + 1)
            # Edge j belongs to ids[j+1]; the block predicts ids[start..stop].
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

            truth = ids[:, start:start + span]
            correct = pred == truth
            runs.append(leading_run(correct).cpu())
            hits += int(correct.sum())
            total += correct.numel()
    return torch.cat(runs).float().mean().item(), hits / max(total, 1)


def main():
    args = parse_args()
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    from image_ptp.vqvae_ar_hf import build_module
    from ptp.transformer import MixedTransformerModel
    from image_ptp.gated_full import GatedFullTransformerModel

    inner = build_module(args.ar_ckpt, device=device, dtype=torch.float32,
                         num_layers=args.num_layers)
    cls = GatedFullTransformerModel if args.gated else MixedTransformerModel
    model = cls(model_id=inner, dtype=torch.float32,
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
    # `ImageTokenDataModule.setup` builds val from make(0, split) and train from
    # make(split, len), so the held-out sequences are the first ones in the file.
    split = min(args.val_split, total // 4)
    lo, hi = {"val": (0, split), "train": (split, total),
              "all": (0, total)}[args.split]
    hi = min(hi, lo + args.images)
    tokens = payload["tokens"][lo:hi].long()
    left = payload["left_bin_edges"][lo:hi]
    right = payload["right_bin_edges"][lo:hi]
    bos = 512

    real_run, real_acc = score(model, tokens, left, right, bos, args.block_len,
                               args.mode, args.first_slot_value, device,
                               args.batch_size, shuffle_u=False, seed=args.seed)
    shuf_run, shuf_acc = score(model, tokens, left, right, bos, args.block_len,
                               args.mode, args.first_slot_value, device,
                               args.batch_size, shuffle_u=True, seed=args.seed)
    result = {
        "ckpt": args.ckpt, "mode": args.mode, "gated": args.gated,
        "block_len": args.block_len, "split": args.split, "images": hi - lo,
        "correct": real_run, "correct_shuffled": shuf_run,
        "correct_lift": real_run / max(shuf_run, 1e-9),
        "accuracy": real_acc, "accuracy_shuffled": shuf_acc,
        "accuracy_lift": real_acc / max(shuf_acc, 1e-9),
    }
    print("%-28s %-5s mode=%-5s block=%-2d  correct %.3f (shuf %.3f, lift %.2fx)  "
          "acc %.4f (shuf %.4f, lift %.2fx)" % (
              Path(args.ckpt).parent.name, args.split, args.mode, args.block_len,
              real_run, shuf_run, result["correct_lift"],
              real_acc, shuf_acc, result["accuracy_lift"]))
    if args.out:
        Path(args.out).expanduser().write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
