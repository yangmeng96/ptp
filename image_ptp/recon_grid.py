"""Real images above, their tokeniser reconstructions below.

The numbers say the CIFAR tokeniser destroys three quarters of the class
identity before any generation happens. This is the same claim without a metric
in the way.
"""
import argparse
import importlib
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dataset", default="cifar10")
    ap.add_argument("--patch", default="1,8")
    ap.add_argument("--ch", type=int, default=128)
    ap.add_argument("--cvae", type=int, default=32)
    ap.add_argument("--vocab", type=int, default=512)
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    device = "cuda"

    os.environ.update(CH=str(args.ch), CVAE=str(args.cvae), VOCAB=str(args.vocab),
                      PATCH=args.patch, DATASET=args.dataset)
    for m in [k for k in sys.modules if k.startswith("image_ptp.mnist_var")]:
        del sys.modules[m]
    mv = importlib.import_module("image_ptp.mnist_var")
    mv.ensure_process_group()
    vae = mv.MnistVQVAE().to(device)
    vae.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
    vae.eval()

    x, _ = next(iter(mv.data_loader(False, batch=args.n, workers=0)))
    x = x[:args.n].to(device)
    with torch.no_grad():
        rec = vae(x)[0].clamp(-1, 1)
    from torchvision.utils import save_image
    grid = torch.cat([x, rec]).cpu()
    save_image(grid, args.out, nrow=args.n, normalize=True, value_range=(-1, 1),
               padding=2)
    print(f"wrote {args.out}: top row real, bottom row through the tokeniser")


if __name__ == "__main__":
    main()
