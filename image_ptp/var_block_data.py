"""Training data for a student that emits VAR's first K scales in one forward.

For each image: the true tokens of those scales, and the interval the pretrained
VAR assigns to each of them. A u drawn inside its interval determines the token,
so (class, u) -> tokens is a deterministic map and the student is learning that
map rather than averaging over anything.

The intervals come from the *guided* distribution, at the same cfg the sampler
uses. Distilling from the unguided one and then sampling with guidance was worth
a factor of four on the MNIST version of this: the student emits what its
teacher's CDF selects and has no second forward at inference to mix against, so
whatever guidance is worth has to be inside the intervals it trained on.
"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, "/home/mengy13/VAR")


@torch.no_grad()
def guided_logits(var, label_B, x_BLCv, cfg, upto):
    """VAR's logits for the first `upto` positions, guided as the sampler guides.

    VAR's own forward drops labels at cond_drop_rate whatever mode it is in, so
    the null class is supplied explicitly here rather than left to chance.
    """
    B = label_B.shape[0]
    null = torch.full_like(label_B, var.num_classes)
    both = torch.cat([label_B, null])
    x2 = torch.cat([x_BLCv, x_BLCv])
    logits = var(both, x2).float()[:, :upto]
    return logits[:B], logits[B:]


def scale_ramp(patch_nums, cfg, upto, device):
    """VAR ramps guidance linearly across scales; one weight per position."""
    out, n = [], 0
    for si, p in enumerate(patch_nums):
        if n >= upto:
            break
        take = min(p * p, upto - n)
        out.append(torch.full((take,), cfg * si / (len(patch_nums) - 1), device=device))
        n += take
    return torch.cat(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vae", default="/home/mengy13/VAR/checkpoints/vae_ch160v4096z32.pth")
    ap.add_argument("--var", default="/home/mengy13/VAR/checkpoints/var_d16.pth")
    ap.add_argument("--data", default="/extra/ucibdl1/shared/data/imagenet/train")
    ap.add_argument("--images", type=int, default=200000)
    ap.add_argument("--scales", type=int, default=4, help="how many leading scales")
    ap.add_argument("--cfg", type=float, default=1.5)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    device = "cuda"
    torch.set_grad_enabled(False)

    from models import build_vae_var
    pn = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
    K = sum(p * p for p in pn[:args.scales])
    vae, var = build_vae_var(device=device, patch_nums=pn, num_classes=1000,
                             depth=16, shared_aln=False)
    vae.load_state_dict(torch.load(args.vae, map_location="cpu"), strict=True)
    var.load_state_dict(torch.load(args.var, map_location="cpu"), strict=True)
    vae.eval(); var.eval()
    var.cond_drop_rate = 0.0

    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader, Subset
    tfm = transforms.Compose([transforms.Resize(292), transforms.CenterCrop(256),
                              transforms.ToTensor(),
                              transforms.Normalize((0.5,) * 3, (0.5,) * 3)])
    ds = datasets.ImageFolder(args.data, transform=tfm)
    g = torch.Generator().manual_seed(0)
    idx = torch.randperm(len(ds), generator=g)[:args.images].tolist()
    loader = DataLoader(Subset(ds, idx), batch_size=args.batch, num_workers=8)

    ramp = scale_ramp(pn, args.cfg, K, device).view(1, -1, 1)
    toks, lefts, rights, labels = [], [], [], []
    seen = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        gt = vae.img_to_idxBl(x)
        truth = torch.cat(gt, 1)[:, :K]
        xin = vae.quantize.idxBl_to_var_input(gt)
        lc, lu = guided_logits(var, y, xin, args.cfg, K)
        probs = torch.softmax((1 + ramp) * lc - ramp * lu, dim=-1)
        cdf = probs.cumsum(-1)
        right = cdf.gather(2, truth.unsqueeze(-1)).squeeze(-1)
        left = right - probs.gather(2, truth.unsqueeze(-1)).squeeze(-1)
        toks.append(truth.to(torch.int16).cpu())
        lefts.append(left.cpu()); rights.append(right.cpu()); labels.append(y.cpu())
        seen += x.shape[0]
        if seen % (args.batch * 200) == 0:
            print(f"  {seen}/{args.images}", flush=True)
    payload = dict(tokens=torch.cat(toks), left=torch.cat(lefts),
                   right=torch.cat(rights), labels=torch.cat(labels),
                   patch_nums=pn, scales=args.scales, cfg=args.cfg, K=K)

    # A u drawn inside its interval must invert to the token it came from, or
    # the student is being trained against edges that describe nothing.
    t = payload["tokens"][:256].long().to(device)
    l, r = payload["left"][:256].to(device), payload["right"][:256].to(device)
    print(f"\ninterval width: median {float((r - l).median()):.5f}  "
          f"min {float((r - l).min()):.2e}  frac<1e-4 {float(((r - l) < 1e-4).float().mean()):.3f}")
    print(f"intervals inside [0,1]: {bool((l >= -1e-6).all() and (r <= 1 + 1e-6).all())}")
    print(f"left < right everywhere: {bool((r > l).all())}")
    torch.save(payload, args.out)
    print(f"wrote {args.out}: {payload['tokens'].shape[0]} images x {K} positions")
    print("BLOCK_DATA_DONE")


if __name__ == "__main__":
    main()
