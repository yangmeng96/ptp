"""Encode MNIST into VQ-VAE code sequences, in the format the data module reads.

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

    payload = {
        "tokens": tokens.to(torch.int32),
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
