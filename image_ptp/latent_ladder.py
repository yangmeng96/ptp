"""How well does f_hat approximate f, scale by scale, under different ladders?

The encoder and decoder are shared across every ladder -- VAR has one of each,
and the scales are only successive quantisations of the same latent f. So
coarsening cannot damage them, and whatever it costs has to appear as a worse
f_hat. This measures that directly, with no decoder and no FID in the way:

  * relative error   ||f - f_hat|| / ||f||  at the end of each ladder
  * per scale, the norm of the residual going in, and how much of it the
    codebook actually removes

If the codebook can absorb a large residual as readily as a small one, a coarse
ladder should lose little and the argument that it cannot is wrong.
"""
import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, "/home/mengy13/VAR")


@torch.no_grad()
def trace(vae, f, patch_nums):
    """Replay VectorQuantizer2.f_to_idxBl_or_fhat, recording what each scale sees."""
    q = vae.quantize
    B, C, H, W = f.shape
    SN = len(patch_nums)
    f_rest = f.clone()
    f_hat = torch.zeros_like(f_rest)
    rows = []
    for si, pn in enumerate(patch_nums):
        before = f_rest.norm().item()
        z = (F.interpolate(f_rest, size=(pn, pn), mode="area")
             .permute(0, 2, 3, 1).reshape(-1, C)) if si != SN - 1 else \
            f_rest.permute(0, 2, 3, 1).reshape(-1, C)
        d = (z.pow(2).sum(1, keepdim=True) + q.embedding.weight.pow(2).sum(1)
             - 2 * z @ q.embedding.weight.T)
        idx = d.argmin(1)
        h = q.embedding(idx).view(B, pn, pn, C).permute(0, 3, 1, 2)
        if pn != H:
            h = F.interpolate(h, size=(H, W), mode="bicubic")
        h = q.quant_resi[si / (SN - 1)](h)
        f_hat = f_hat + h
        f_rest = f_rest - h
        rows.append((pn, before, f_rest.norm().item(),
                     h.norm().item() / max(before, 1e-9)))
    return f_hat, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vae", default="/home/mengy13/VAR/checkpoints/vae_ch160v4096z32.pth")
    ap.add_argument("--data", default="/extra/ucibdl1/shared/data/imagenet/val")
    ap.add_argument("--images", type=int, default=256)
    ap.add_argument("--ladders", default="1,2,3,4,5,6,8,10,13,16|1,2,4,8,16|1,16")
    args = ap.parse_args()
    device = "cuda"
    torch.set_grad_enabled(False)

    from models.vqvae import VQVAE
    vae = VQVAE(vocab_size=4096, z_channels=32, ch=160, test_mode=True,
                share_quant_resi=4,
                v_patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16)).to(device).eval()
    vae.load_state_dict(torch.load(args.vae, map_location="cpu"), strict=True)

    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader, Subset
    tfm = transforms.Compose([transforms.Resize(292), transforms.CenterCrop(256),
                              transforms.ToTensor(),
                              transforms.Normalize((0.5,) * 3, (0.5,) * 3)])
    ds = datasets.ImageFolder(args.data, transform=tfm)
    g = torch.Generator().manual_seed(0)
    sub = Subset(ds, torch.randperm(len(ds), generator=g)[:args.images].tolist())
    x = torch.cat([b for b, _ in DataLoader(sub, batch_size=32, num_workers=8)])
    f = vae.quant_conv(vae.encoder(x.to(device)))
    print(f"{x.shape[0]} images, latent {tuple(f.shape[1:])}, ||f|| = {f.norm():.1f}\n")

    # A ceiling that owes nothing to the ladder: quantise f once, at full
    # resolution, straight into the codebook.
    for spec in args.ladders.split("|"):
        pn = tuple(int(v) for v in spec.split(","))
        f_hat, rows = trace(vae, f, pn)
        rel = ((f - f_hat).norm() / f.norm()).item()
        print(f"ladder {spec}")
        print(f"  final ||f - f_hat|| / ||f|| = {rel:.4f}")
        print(f"  {'scale':>6} {'residual in':>12} {'residual out':>13} {'absorbed':>9}")
        for p, b, a, frac in rows:
            print(f"  {p:>6} {b:12.1f} {a:13.1f} {1 - a / b:8.1%}")
        print()
    print("LATENT_LADDER_DONE")


if __name__ == "__main__":
    main()
