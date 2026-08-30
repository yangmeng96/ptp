"""Distil the within-scale AR teacher back into one parallel call per scale.

The teacher models the joint over a scale and pays 65 sequential forwards for it
on the (1,8) ladder, where the parallel VAR pays 2. O-PTP is meant to buy the
first for the price of the second: every position of a scale is handed its own
u_i, all of them at once, and has to emit the token the teacher's inverse CDF
selects -- which requires inferring, from u alone, what the earlier positions
produced.

    parallel VAR, 2 forwards      2.0618 nats/token   the floor
    PTP student,  2 forwards           ?
    within-scale AR, 65 forwards  1.4576 nats/token   the ceiling
    (1,2,4,8) parallel, 4 forwards 1.4206             what this is competing with

Scored two ways, because they answer different questions. Cross-entropy against
the true tokens says how good a generative model it is. Exact agreement with the
teacher's own pick -- and the same figure with u shuffled between images -- says
whether it is reading u at all, which is the failure mode every LlamaGen run fell
into before there was enough data.
"""
import argparse
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, "/home/mengy13/ptp")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--patch", type=str, default="1,8")
    p.add_argument("--dir", type=str,
                   default="/home/mengy13/ptp-image-results/mnist_var_r8")
    p.add_argument("--adapter", type=str, default="binary",
                   choices=["binary", "linear_interpolation", "quarter_cos"])
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=128)
    # Guidance is worth a factor of four to this teacher and nothing at all to a
    # student distilled from its raw conditional, because the student has no
    # second forward to mix against. Baking a fixed scale into the intervals is
    # the only way it collects any of that.
    p.add_argument("--cfg", type=float, default=0.0)
    p.add_argument("--teacher-tag", type=str, default="var_within_scale")
    p.add_argument("--student-tag", type=str, default="var_ptp_student")
    return p.parse_args()


def main():
    args = parse_args()
    os.environ["PATCH"] = args.patch
    os.environ["OUT"] = args.dir
    os.environ.setdefault("VAR_ROOT", "/home/mengy13/VAR")
    device = "cuda"
    d = Path(args.dir)

    from image_ptp import mnist_var as mv
    from image_ptp.within_scale_var import (build_within_scale_var,
                                            build_ptp_student, var_input,
                                            bin_edges)

    mv.ensure_process_group()
    vae = mv.MnistVQVAE().to(device)
    vae.load_state_dict(torch.load(d / "vqvae.pt", map_location="cpu"))
    vae.eval().requires_grad_(False)

    teacher = build_within_scale_var(vae, mv.PATCH, num_classes=mv.NUM_CLASSES,
                                     device=device)
    teacher.load_state_dict(torch.load(d / f"{args.teacher_tag}.pt", map_location="cpu"))
    teacher.eval().requires_grad_(False)
    teacher.cond_drop_rate = 0.0

    student = build_ptp_student(vae, mv.PATCH, adapter=args.adapter,
                                num_classes=mv.NUM_CLASSES, device=device)
    print(f"teacher {args.teacher_tag}, cfg {args.cfg} -> {args.student_tag}.pt",
          flush=True)
    print(f"student {sum(p.numel() for p in student.parameters())/1e6:.1f}M, "
          f"u encoding {args.adapter}, scales {mv.PATCH}", flush=True)

    train = mv.mnist_loader(True, batch=args.batch_size)
    test = mv.mnist_loader(False, batch=args.batch_size, workers=2)

    # An interval is only meaningful if drawing u inside it and inverting the
    # teacher's CDF returns the token. Check before anything trains on it.
    x, y = next(iter(test))
    x, y = x.to(device), y.to(device)
    gt = vae.img_to_idxBl(x)
    truth = torch.cat(gt, 1)
    left, right, probs = bin_edges(teacher, vae, y, gt, device, cfg=args.cfg)
    u = left + (right - left) * torch.rand(left.shape, device=device)
    rec = torch.searchsorted(probs.cumsum(-1).contiguous(),
                             u.unsqueeze(-1).contiguous()).squeeze(-1)
    rate = float((rec == truth).float().mean())
    print(f"oracle inverse-CDF recovers the true token: {rate:.4f}", flush=True)
    assert rate > 0.99, "bin edges do not describe this teacher"
    print(f"bin width median {float((right-left).median()):.4f}", flush=True)

    opt = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=0.05,
                            betas=(0.9, 0.95))

    def evaluate():
        student.eval()
        student.cond_drop_rate = 0.0
        ce = agree = shuf = n = 0.0
        with torch.no_grad():
            for x, y in test:
                x, y = x.to(device), y.to(device)
                gt = vae.img_to_idxBl(x)
                truth = torch.cat(gt, 1)
                left, right, _ = bin_edges(teacher, vae, y, gt, device, cfg=args.cfg)
                u = left + (right - left) * torch.rand(left.shape, device=device)
                xin = var_input(vae, gt, device)
                lg = student(y, xin, u=u).float()
                ce += float(F.cross_entropy(lg.reshape(-1, mv.VOCAB),
                                            truth.reshape(-1))) * x.shape[0]
                agree += float((lg.argmax(-1) == truth).float().mean()) * x.shape[0]
                perm = torch.randperm(x.shape[0], device=device)
                lg2 = student(y, xin, u=u[perm]).float()
                shuf += float((lg2.argmax(-1) == truth).float().mean()) * x.shape[0]
                n += x.shape[0]
        student.train()
        student.cond_drop_rate = 0.1
        return ce / n, agree / n, shuf / n

    best = float("inf")
    started = time.time()
    for ep in range(args.epochs):
        for x, y in train:
            x, y = x.to(device, non_blocking=True), y.to(device)
            with torch.no_grad():
                gt = vae.img_to_idxBl(x)
                truth = torch.cat(gt, 1)
                left, right, _ = bin_edges(teacher, vae, y, gt, device, cfg=args.cfg)
                u = left + (right - left) * torch.rand(left.shape, device=device)
                xin = var_input(vae, gt, device)
            lg = student(y, xin, u=u)
            loss = F.cross_entropy(lg.reshape(-1, mv.VOCAB), truth.reshape(-1))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            opt.step()
        ce, agree, shuf = evaluate()
        mark = ""
        if ce < best:
            best = ce
            torch.save(student.state_dict(), d / f"{args.student_tag}.pt")
            mark = "  <- kept"
        print(f"epoch {ep+1}/{args.epochs} test CE {ce:.4f}  agrees with teacher "
              f"{agree:.4f}  shuffled u {shuf:.4f}  lift {agree/max(shuf,1e-9):.2f}x "
              f"({time.time()-started:.0f}s){mark}", flush=True)

    print(f"\nbest PTP student {best:.4f} nats/token in 2 forwards")
    print("  floor   parallel VAR, 2 forwards        2.0618")
    print("  ceiling within-scale AR, 65 forwards    1.4576")
    print("  target  (1,2,4,8) parallel, 4 forwards  1.4206")
    print("VAR_PTP_DONE", flush=True)


if __name__ == "__main__":
    main()
