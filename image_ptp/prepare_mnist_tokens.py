"""Encode MNIST into VQ-VAE code sequences and cache the teacher's bin edges.

The edges are the important part. `ptp.transformer` recovers the teacher by
switching LoRA adapters off, so with no adapters present -- a full finetune --
`ar_forward` returns the weights currently being trained and the run silently
becomes self-distillation against a moving target. Measured on the first
attempt: 47% of bins had drifted more than half their own width away from the
frozen teacher's.

Computing the edges here from a genuinely frozen teacher and passing them
through the batch takes `ptp.lit` down its `bin_edges_left`/`bin_edges_right`
path, which skips `predict_bin_edges` entirely. No repo changes needed, and
full finetune becomes a real distillation again.

Written to match `image_ptp/pregenerate.py`'s output so `ImageTokenDataModule`
serves both without a special case. The BOS token takes the slot the class label
occupies for LlamaGen: with `code_vocab` set to the codebook size and every
label zero, `[label + code_vocab, codes...]` comes out as `[BOS, k_0, ...]`,
which is exactly the layout the AR teacher was trained on.

Unlike the LlamaGen data these are encodings of real images rather than teacher
samples. The `ptp-vqvae` run trained on the same thing, so keeping it that way
preserves the comparison.

Usage:
    python prepare_mnist_tokens.py --repo ~/ptp-vqvae \
        --out ~/ptp-image-results/mnist_tokens.pt
"""
import argparse
import os
import sys
from pathlib import Path

import torch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", type=str, default="/home/mengy13/ptp-vqvae")
    p.add_argument("--teacher-ckpt", type=str, default="checkpoints/ar_mnist_raster.pt")
    p.add_argument("--split", type=str, default="train", choices=["train", "test"])
    p.add_argument("--limit", type=int, default=0, help="0 uses the whole split")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--out", type=str, required=True)
    return p.parse_args()


def main():
    args = parse_args()
    sys.path.insert(0, args.repo)
    os.chdir(args.repo)
    from models.ar import codes_flat_to_seq
    from utils.helper import load_vqvae
    from torchvision import datasets
    sys.path.insert(0, "/home/mengy13/ptp")
    from image_ptp.vqvae_ar_hf import build

    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    meta = torch.load(args.teacher_ckpt, map_location="cpu", weights_only=False)
    perm = meta["perm"].to(device)
    num_codes = meta["num_codes"]

    vqvae, tfm = load_vqvae("mnist", device)
    ds = datasets.MNIST(root="data/mnist", train=args.split == "train",
                        download=False, transform=tfm)
    total = len(ds) if args.limit in (0, None) else min(args.limit, len(ds))
    print(f"encoding {total} {args.split} images, codebook {num_codes}")

    chunks = []
    for start in range(0, total, args.batch_size):
        stop = min(start + args.batch_size, total)
        images = torch.stack([ds[i][0] for i in range(start, stop)]).to(device)
        chunks.append(codes_flat_to_seq(vqvae.encode(images), perm).cpu())
    tokens = torch.cat(chunks)

    # Bin edges from the frozen teacher, laid out the way ptp.lit reads them:
    # entry j belongs to input_ids[j+1], so the array is one shorter than the
    # sequence fed to the model.
    teacher, _ = build(args.teacher_ckpt, device=device, dtype=torch.float32)
    lefts, rights = [], []
    for start in range(0, tokens.shape[0], args.batch_size):
        chunk = tokens[start:start + args.batch_size].long().to(device)
        ids = torch.cat([torch.full((chunk.shape[0], 1), num_codes, device=device), chunk], 1)
        probs = torch.softmax(teacher(input_ids=ids).logits[:, :-1].float(), dim=-1)
        cdf = probs.cumsum(-1)
        target = chunk.unsqueeze(-1)
        right = cdf.gather(2, target).squeeze(2)
        lefts.append((right - probs.gather(2, target).squeeze(2)).cpu())
        rights.append(right.cpu())
    left, right = torch.cat(lefts), torch.cat(rights)
    width = right - left
    print(f"bin width: median={width.median():.4f} mean={width.mean():.4f} "
          f"min={width.min():.2e}")

    # An interval is only meaningful if drawing u inside it and inverting the
    # teacher's CDF gives the token back. Check before anything trains on it.
    probe = slice(0, min(256, tokens.shape[0]))
    chunk = tokens[probe].long().to(device)
    ids = torch.cat([torch.full((chunk.shape[0], 1), num_codes, device=device), chunk], 1)
    cdf = torch.softmax(teacher(input_ids=ids).logits[:, :-1].float(), -1).cumsum(-1)
    u = (left[probe].to(device) + width[probe].to(device)
         * torch.rand(chunk.shape, device=device))
    recovered = torch.searchsorted(cdf.contiguous(), u.unsqueeze(-1).contiguous()).squeeze(-1)
    rate = (recovered == chunk).float().mean().item()
    print(f"oracle inverse-CDF recovers the true token: {rate:.4f}")
    assert rate > 0.99, "bin edges do not describe this teacher"

    payload = {
        "tokens": tokens.to(torch.int32),
        "left_bin_edges": left,
        "right_bin_edges": right,
        # Zero labels, so the data module emits BOS = 0 + code_vocab.
        "labels": torch.zeros(tokens.shape[0], dtype=torch.int32),
        "config": {
            "source": f"mnist-{args.split}", "num_codes": num_codes,
            "seq_len": tokens.shape[1], "num_images": tokens.shape[0],
            "order": meta["order"],
        },
    }
    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out)
    print(f"wrote {out}: {tuple(tokens.shape)} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
