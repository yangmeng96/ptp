"""What would VAR lose by emitting several of its scales in one forward pass?

Coarsening the ladder needs a retrained tokeniser and leans on within-scale
dependence, which is 0.04% and leaves PTP nothing to distil. Merging adjacent
*scales* instead leans on the cross-scale dependence VAR is actually
autoregressive in, keeps the pretrained tokeniser untouched at rFID 0.755, and
needs no teacher training -- var_d16 already gives every position's conditional
distribution under teacher forcing.

Two things decide which scales to merge, and neither needs any training:

  * how much a scale's prediction depends on the ones that would be merged with
    it, as the cross-entropy a frozen prefix costs -- this is what PTP's
    auxiliaries would have to carry
  * what each scale's forward actually costs, since 1x1 through 4x4 is 30 tokens
    against 16x16's 256, and batch-1 latency stops being flat well below that
"""
import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, "/home/mengy13/VAR")


def var_input_frozen(quantize, gt, freeze_from):
    """VAR's teacher-forcing input, but with f_hat frozen from scale freeze_from.

    A verbatim copy of idxBl_to_var_input except that scales at or after
    freeze_from stop accumulating: they all see the f_hat that existed before
    the merged block began, which is exactly what a single forward emitting that
    block would have. freeze_from=SN reproduces the original function.
    """
    next_scales = []
    B, C = gt[0].shape[0], quantize.Cvae
    H = W = quantize.v_patch_nums[-1]
    SN = len(quantize.v_patch_nums)
    f_hat = gt[0].new_zeros(B, C, H, W, dtype=torch.float32)
    frozen = None
    pn_next = quantize.v_patch_nums[0]
    for si in range(SN - 1):
        h = F.interpolate(quantize.embedding(gt[si]).transpose(1, 2)
                          .view(B, C, pn_next, pn_next), size=(H, W), mode="bicubic")
        f_hat = f_hat + quantize.quant_resi[si / (SN - 1)](h)
        if si + 1 == freeze_from:
            frozen = f_hat.clone()
        pn_next = quantize.v_patch_nums[si + 1]
        use = f_hat if (frozen is None or si + 1 < freeze_from) else frozen
        next_scales.append(F.interpolate(use, size=(pn_next, pn_next), mode="area")
                           .view(B, C, -1).transpose(1, 2))
    return torch.cat(next_scales, dim=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vae", default="/home/mengy13/VAR/checkpoints/vae_ch160v4096z32.pth")
    ap.add_argument("--var", default="/home/mengy13/VAR/checkpoints/var_d16.pth")
    ap.add_argument("--data", default="/extra/ucibdl1/shared/data/imagenet/val")
    ap.add_argument("--images", type=int, default=512)
    args = ap.parse_args()
    device = "cuda"
    torch.set_grad_enabled(False)

    from models import build_vae_var
    pn = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
    vae, var = build_vae_var(device=device, patch_nums=pn, num_classes=1000,
                             depth=16, shared_aln=False)
    vae.load_state_dict(torch.load(args.vae, map_location="cpu"), strict=True)
    var.load_state_dict(torch.load(args.var, map_location="cpu"), strict=True)
    vae.eval(); var.eval()
    var.cond_drop_rate = 0.0     # forward drops labels at any time otherwise

    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader, Subset
    tfm = transforms.Compose([transforms.Resize(292), transforms.CenterCrop(256),
                              transforms.ToTensor(),
                              transforms.Normalize((0.5,) * 3, (0.5,) * 3)])
    ds = datasets.ImageFolder(args.data, transform=tfm)
    g = torch.Generator().manual_seed(0)
    sub = Subset(ds, torch.randperm(len(ds), generator=g)[:args.images].tolist())
    loader = DataLoader(sub, batch_size=16, num_workers=8)

    bounds, cur = [], 0
    for p in pn:
        bounds.append((cur, cur + p * p)); cur += p * p
    L = cur

    # ---- 1. what a frozen prefix costs, per scale
    print("Cross-entropy per scale under teacher forcing, and under a prefix")
    print("frozen from scale index F -- the loss PTP's auxiliaries must carry.\n")
    freezes = [len(pn), 1, 2, 3, 4, 5, 6, 7]
    tot = {f: torch.zeros(len(pn)) for f in freezes}
    n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        gt = vae.img_to_idxBl(x)
        truth = torch.cat(gt, 1)
        for f in freezes:
            xin = var_input_frozen(vae.quantize, gt, f)
            logits = var(y, xin).float()
            for si, (a, b) in enumerate(bounds):
                ce = F.cross_entropy(logits[:, a:b].reshape(-1, logits.shape[-1]),
                                     truth[:, a:b].reshape(-1), reduction="mean")
                tot[f][si] += float(ce) * x.shape[0]
        n += x.shape[0]
    hdr = "  ".join(f"{p}x{p}".rjust(6) for p in pn)
    print(f"{'freeze':>8}  {hdr}")
    for f in freezes:
        row = "  ".join(f"{v / n:6.3f}" for v in tot[f])
        tag = "none" if f == len(pn) else f"from {pn[f]}x{pn[f]}"
        print(f"{tag:>8}  {row}")
    print(f"\n({n} images)")

    # ---- 2. what each scale's forward actually costs at batch 1
    print("\nPer-scale forward cost, batch 1, KV cache on:")
    times = []
    orig = var.get_logits
    def timed(*a, **k):
        torch.cuda.synchronize(); times.append(time.perf_counter())
        return orig(*a, **k)
    var.get_logits = timed
    for _ in range(2):                      # warm up, then measure
        times.clear()
        var.autoregressive_infer_cfg(B=1, label_B=torch.tensor([0], device=device),
                                     cfg=1.5, top_k=900, top_p=0.95, g_seed=0)
        torch.cuda.synchronize(); times.append(time.perf_counter())
    var.get_logits = orig
    per = [(times[i + 1] - times[i]) * 1000 for i in range(len(times) - 1)]
    total = sum(per)
    print(f"{'scale':>7} {'tokens':>7} {'ms':>8} {'share':>7}")
    for p, t in zip(pn, per):
        print(f"{f'{p}x{p}':>7} {p * p:>7} {t:8.2f} {t / total:6.1%}")
    print(f"{'total':>7} {sum(q * q for q in pn):>7} {total:8.2f}")
    print("MERGE_SCALES_DONE")


if __name__ == "__main__":
    main()
