"""Where the student's token sits in the teacher's ranking.

Exact agreement is the wrong headline: the teacher gives the real image's own
token a median probability of 0.001, its own favourite only 0.025-0.11, and half
its mass sits in the top ~180 of 4096. So a student that never matches the
teacher's pick can still be choosing sensibly. What matters is how far down the
teacher's ordering its choice falls.

This needs the teacher's full distribution, which is 98 GB stored, so it is
recomputed here over a small held-out set instead.
"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, "/home/mengy13/VAR")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--students", required=True, help="name=ckpt,name=ckpt")
    ap.add_argument("--vae", default="/home/mengy13/VAR/checkpoints/vae_ch160v4096z32.pth")
    ap.add_argument("--var", default="/home/mengy13/VAR/checkpoints/var_d16.pth")
    ap.add_argument("--data", default="/extra/ucibdl1/shared/data/imagenet/val")
    ap.add_argument("--images", type=int, default=512)
    ap.add_argument("--scales", type=int, default=4)
    ap.add_argument("--cfg", type=float, default=1.5)
    ap.add_argument("--out", default="/home/mengy13/ptp-image-results/student_rank.pt")
    args = ap.parse_args()
    device = "cuda"
    torch.set_grad_enabled(False)

    from models import build_vae_var
    from image_ptp.var_block_student import (BlockStudent, scale_causal_mask,
                                             draw_u)
    from image_ptp.var_block_data import scale_ramp
    pn = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
    K = sum(p * p for p in pn[:args.scales])
    vae, var = build_vae_var(device=device, patch_nums=pn, num_classes=1000,
                             depth=16, shared_aln=False)
    vae.load_state_dict(torch.load(args.vae, map_location="cpu"), strict=True)
    var.load_state_dict(torch.load(args.var, map_location="cpu"), strict=True)
    vae.eval(); var.eval(); var.cond_drop_rate = 0.0

    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader, Subset
    tfm = transforms.Compose([transforms.Resize(292), transforms.CenterCrop(256),
                              transforms.ToTensor(),
                              transforms.Normalize((0.5,) * 3, (0.5,) * 3)])
    ds = datasets.ImageFolder(args.data, transform=tfm)
    g = torch.Generator().manual_seed(0)
    sub = Subset(ds, torch.randperm(len(ds), generator=g)[:args.images].tolist())
    ramp = scale_ramp(pn, args.cfg, K, device).view(1, -1, 1)

    students = {}
    for spec in args.students.split(","):
        name, ck = spec.split("=", 1)
        blob = torch.load(ck, map_location="cpu")
        a = blob["args"]
        sd = blob["model"]
        mode = a.get("mode", "optp")
        st = BlockStudent(pn, args.scales, vocab=sd["head_ce.weight"].shape[0],
                          n_class=sd["cls.weight"].shape[0] - 1,
                          dim=a["dim"], depth=a["depth"],
                          u_encoding=a.get("u_encoding", "binary"), mode=mode)
        st.load_state_dict(sd)
        students[name] = (st.to(device).eval(), a, mode,
                          scale_causal_mask(pn, args.scales, device, mode))

    ranks = {n: [] for n in students}
    ranks["teacher's own sample"] = []
    ranks["the real image"] = []
    for x, y in DataLoader(sub, batch_size=8, num_workers=4):
        x, y = x.to(device), y.to(device)
        gt = vae.img_to_idxBl(x)
        truth = torch.cat(gt, 1)[:, :K]
        xin = vae.quantize.idxBl_to_var_input(gt)
        null = torch.full_like(y, var.num_classes)
        lg = var(torch.cat([y, null]), torch.cat([xin, xin])).float()[:, :K]
        B = x.shape[0]
        probs = torch.softmax((1 + ramp) * lg[:B] - ramp * lg[B:], -1)
        order = probs.argsort(dim=-1, descending=True)
        rank_of = torch.empty_like(order)
        rank_of.scatter_(2, order, torch.arange(order.shape[-1], device=device)
                         .expand_as(order))
        left = probs.cumsum(-1).gather(2, truth.unsqueeze(-1)).squeeze(-1) \
            - probs.gather(2, truth.unsqueeze(-1)).squeeze(-1)
        right = left + probs.gather(2, truth.unsqueeze(-1)).squeeze(-1)

        ranks["the real image"].append(
            rank_of.gather(2, truth.unsqueeze(-1)).squeeze(-1).cpu())
        drawn = torch.multinomial(probs.reshape(-1, probs.shape[-1]), 1).view(B, K)
        ranks["teacher's own sample"].append(
            rank_of.gather(2, drawn.unsqueeze(-1)).squeeze(-1).cpu())
        for n, (st, a, mode, mask) in students.items():
            u = draw_u(left, right, a["gaussian"])
            out = st(u, y, mask)[0]
            if mode == "cptp":
                # its head is the teacher's distribution, so the token comes from
                # inverting that CDF at this slot's own u, not from its mode
                cq = out.softmax(-1).cumsum(-1)
                pred = torch.searchsorted(cq.contiguous(),
                                          u.unsqueeze(-1).contiguous()
                                          ).squeeze(-1).clamp(max=out.shape[-1] - 1)
            else:
                pred = out.argmax(-1)
            ranks[n].append(rank_of.gather(2, pred.unsqueeze(-1)).squeeze(-1).cpu())

    ranks = {n: torch.cat(v).float() for n, v in ranks.items()}
    torch.save(ranks, args.out)
    qs = torch.tensor([0.10, 0.25, 0.50, 0.75, 0.90])
    print(f"{args.images} held-out images, {K} positions each, teacher guided "
          f"at cfg {args.cfg}\n")
    print(f"{'whose token':<26} " + " ".join(f"{f'p{int(q*100)}':>7} " for q in qs)
          + f"{'<=10':>7} {'<=100':>7} {'<=900':>7}")
    for n, v in ranks.items():
        f = v.flatten()
        q = torch.quantile(f[torch.randperm(len(f))[:200000]], qs)
        print(f"{n:<26} " + " ".join(f"{int(x):7d} " for x in q)
              + f"{float((f <= 10).float().mean()):7.3f} "
                f"{float((f <= 100).float().mean()):7.3f} "
                f"{float((f <= 900).float().mean()):7.3f}")
    print("\n(rank 0 = teacher's own favourite)")
    print("STUDENT_RANK_DONE")


if __name__ == "__main__":
    main()
