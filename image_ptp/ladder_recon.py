"""How much reconstruction does coarsening the ladder cost, on a good tokeniser?

Two things have been confounded in the CIFAR VAR work. The tokeniser trained
here is poor -- real images through it and back score FID ~100 against published
tokenisers' ~1-8 -- and the ladder was coarsened from (1,2,4,8) to (1,8), which
is the whole point of the experiment. Both damage reconstruction, and until now
there was no way to say which did how much.

VAR ships a pretrained multi-scale tokeniser, and its residual quantiser takes
the ladder as a call-time argument: `f_to_idxBl_or_fhat(f, v_patch_nums=...)`
only requires the last scale to match the latent resolution. So the same
known-good tokeniser can be run at any ladder, and the coarsening cost isolated
with no training at all.
"""
import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, "/home/mengy13/VAR")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vae", default="/home/mengy13/VAR/checkpoints/vae_ch160v4096z32.pth")
    ap.add_argument("--data", default="/extra/ucibdl1/shared/data/imagenet/val")
    ap.add_argument("--images", type=int, default=2000)
    ap.add_argument("--ladders", default="1,2,3,4,5,6,8,10,13,16|1,2,4,8,16|1,4,16|1,16|16")
    args = ap.parse_args()
    device = "cuda"
    torch.set_grad_enabled(False)

    from models.vqvae import VQVAE
    from image_ptp.fid_features import InceptionFeatures, frechet
    vae = VQVAE(vocab_size=4096, z_channels=32, ch=160, test_mode=True,
                share_quant_resi=4, v_patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
                ).to(device).eval()
    vae.load_state_dict(torch.load(args.vae, map_location="cpu"), strict=True)

    from torchvision import datasets, transforms
    tfm = transforms.Compose([
        transforms.Resize(292), transforms.CenterCrop(256),
        transforms.ToTensor(), transforms.Normalize((0.5,) * 3, (0.5,) * 3)])
    ds = datasets.ImageFolder(args.data, transform=tfm)
    g = torch.Generator().manual_seed(0)
    idx = torch.randperm(len(ds), generator=g)[:args.images].tolist()
    from torch.utils.data import DataLoader, Subset
    loader = DataLoader(Subset(ds, idx), batch_size=16, num_workers=8)

    ext = InceptionFeatures(device=device)

    # 50k images at 256x256 is 39 GB held as float32, so nothing is kept: each
    # pass streams the loader and accumulates features only.
    def sweep(pn=None):
        feats, se, n = [], 0.0, 0
        for x, _ in loader:
            x = x.to(device)
            if pn is None:
                out = x
            else:
                f = vae.quant_conv(vae.encoder(x))
                fhat = vae.quantize.f_to_idxBl_or_fhat(f, to_fhat=True,
                                                       v_patch_nums=pn)[-1]
                out = vae.fhat_to_img(fhat)
                se += float(F.mse_loss(out, x, reduction="sum"))
                n += x.numel()
            feats.append(ext(out).cpu())
        psnr = (10 * torch.log10(torch.tensor(4.0 / (se / n)))).item() if n else float("nan")
        return torch.cat(feats), psnr

    f_real, _ = sweep(None)
    print(f"{f_real.shape[0]} ImageNet val images at 256x256, "
          f"inception {f_real.shape[1]}-d\n")
    print(f"{'ladder':<28} {'levels':>7} {'tokens':>7} {'PSNR dB':>8} {'recon FID':>10}")
    for spec in args.ladders.split("|"):
        pn = tuple(int(v) for v in spec.split(","))
        f_rec, psnr = sweep(pn)
        print(f"{spec:<28} {len(pn):>7} {sum(p * p for p in pn):>7} {psnr:8.2f} "
              f"{frechet(f_rec, f_real):10.3f}", flush=True)
    print("LADDER_RECON_DONE")


if __name__ == "__main__":
    main()
