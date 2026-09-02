"""LlamaGen's tokeniser through the same pipeline as VAR's, for one comparison.

The argument this settles: LlamaGen's tokeniser is effectively a single-level
VAR -- one 16x16 grid, 256 tokens, no ladder at all -- and it is reported at
rFID ~2. If that reproduces here, then level count is not what limits
reconstruction, the geometric decay measured on VAR's truncated ladder is a
property of running that checkpoint off its own schedule, and a coarse ladder
has no tokeniser problem provided the tokeniser is trained for it.

Same images, same InceptionV3, same preprocessing, same sample count as
ladder_recon, so the two numbers are directly comparable.
"""
import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, "/home/mengy13/LlamaGen")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/home/mengy13/LlamaGen/pretrained_models/vq_ds16_c2i.pt")
    ap.add_argument("--data", default="/extra/ucibdl1/shared/data/imagenet/val")
    ap.add_argument("--images", type=int, default=2000)
    args = ap.parse_args()
    device = "cuda"
    torch.set_grad_enabled(False)

    from tokenizer.tokenizer_image.vq_model import VQ_models
    from image_ptp.fid_features import (InceptionFeatures, frechet,
                                    check_sample_count)
    vq = VQ_models["VQ-16"](codebook_size=16384, codebook_embed_dim=8).to(device).eval()
    sd = torch.load(args.ckpt, map_location="cpu")
    vq.load_state_dict(sd["model"] if "model" in sd else sd)

    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader, Subset
    tfm = transforms.Compose([transforms.Resize(292), transforms.CenterCrop(256),
                              transforms.ToTensor(),
                              transforms.Normalize((0.5,) * 3, (0.5,) * 3)])
    ds = datasets.ImageFolder(args.data, transform=tfm)
    g = torch.Generator().manual_seed(0)
    sub = Subset(ds, torch.randperm(len(ds), generator=g)[:args.images].tolist())
    loader = DataLoader(sub, batch_size=16, num_workers=8)

    ext = InceptionFeatures(device=device)
    f_real, f_rec, se, n = [], [], 0.0, 0
    for x, _ in loader:
        x = x.to(device)
        quant, _, _ = vq.encode(x)
        rec = vq.decode(quant).clamp(-1, 1)
        se += float(F.mse_loss(rec, x, reduction="sum"))
        n += x.numel()
        f_real.append(ext(x).cpu())
        f_rec.append(ext(rec).cpu())
    psnr = 10 * torch.log10(torch.tensor(4.0 / (se / n)))
    check_sample_count(len(torch.cat(f_real)), 2048)
    print(f"{n // (3 * 256 * 256)} ImageNet val images at 256x256\n")
    print(f"{'tokeniser':<28} {'levels':>7} {'tokens':>7} {'PSNR dB':>8} {'recon FID':>10}")
    print(f"{'LlamaGen VQ-16':<28} {1:>7} {16 * 16:>7} {float(psnr):8.2f} "
          f"{frechet(torch.cat(f_rec), torch.cat(f_real)):10.3f}")
    print("LLAMAGEN_RECON_DONE")


if __name__ == "__main__":
    main()
