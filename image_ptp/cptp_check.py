"""Does the C-PTP student's output actually become the teacher's distribution?

That is what the construction claims: with its own u withheld, the target token
is distributed as p, and the minimiser of cross-entropy against samples from p
is p itself. Agreement rates cannot check this -- they only ask whether one
sample matched -- but the distributions can be compared directly wherever the
teacher's full conditional fits in memory. MNIST's vocabulary is 512, so it does.

An O-PTP student is run through the same comparison as a control: its optimum is
a one-hot, so it should be far from p by construction, and if it is not then the
two modes are not doing what their masks say.
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
    ap.add_argument("--students", required=True, help="name=ckpt,name=ckpt")
    ap.add_argument("--dir", default="/home/mengy13/ptp-image-results/mnist_var_r2")
    ap.add_argument("--patch", default="1,2,4,8")
    ap.add_argument("--scales", type=int, default=3)
    ap.add_argument("--cfg", type=float, default=1.5)
    ap.add_argument("--images", type=int, default=2000)
    args = ap.parse_args()
    device = "cuda"
    torch.set_grad_enabled(False)

    from image_ptp.mnist_merge import load as load_mnist
    from image_ptp.var_block_student import (BlockStudent, scale_causal_mask,
                                             draw_u)
    from image_ptp.var_block_data import scale_ramp
    mv, vae, var = load_mnist(args.dir, args.patch)
    pn = var.patch_nums
    K = sum(p * p for p in pn[:args.scales])
    ramp = scale_ramp(pn, args.cfg, K, device).view(1, -1, 1)

    students = {}
    for spec in args.students.split(","):
        name, ck = spec.split("=", 1)
        blob = torch.load(ck, map_location="cpu")
        a = blob["args"]
        st = BlockStudent(pn, args.scales, vocab=mv.VOCAB, dim=a["dim"],
                          depth=a["depth"], n_class=mv.NUM_CLASSES,
                          u_encoding=a.get("u_encoding", "binary"),
                          mode=a.get("mode", "optp"))
        st.load_state_dict(blob["model"])
        students[name] = (st.to(device).eval(), a,
                          scale_causal_mask(pn, args.scales, device,
                                            a.get("mode", "optp")))

    acc = {n: dict(kl=0.0, tv=0.0, agree=0.0, ent=0.0) for n in students}
    tot = 0
    for x, y in mv.data_loader(False, batch=250, workers=4):
        if tot >= args.images:
            break
        x, y = x.to(device), y.to(device)
        gt = vae.img_to_idxBl(x)
        truth = torch.cat(gt, 1)[:, :K]
        xin = vae.quantize.idxBl_to_var_input(gt)
        null = torch.full_like(y, var.num_classes)
        lg = var(torch.cat([y, null]), torch.cat([xin, xin])).float()[:, :K]
        B = x.shape[0]
        p = torch.softmax((1 + ramp) * lg[:B] - ramp * lg[B:], -1)
        cdf = p.cumsum(-1)
        right = cdf.gather(2, truth.unsqueeze(-1)).squeeze(-1)
        left = right - p.gather(2, truth.unsqueeze(-1)).squeeze(-1)
        for n, (st, a, mask) in students.items():
            u = draw_u(left, right, a["gaussian"])
            q = st(u, y, mask)[0].softmax(-1)
            acc[n]["kl"] += float((p * ((p + 1e-9).log() - (q + 1e-9).log()))
                                  .sum(-1).mean()) * B
            acc[n]["tv"] += float(0.5 * (p - q).abs().sum(-1).mean()) * B
            acc[n]["agree"] += float((q.argmax(-1) == truth).float().mean()) * B
            acc[n]["ent"] += float(-(q * (q + 1e-9).log()).sum(-1).mean()) * B
        tot += B
    pent = float(-(p * (p + 1e-9).log()).sum(-1).mean())
    print(f"{tot} held-out MNIST images, {K} positions, vocab {mv.VOCAB}, "
          f"cfg {args.cfg}")
    print(f"teacher's own entropy: {pent:.3f} nats  (exp {2.718281828 ** pent:.1f} "
          f"effective choices)\n")
    print(f"{'student':<16} {'mode':>6} {'KL(p||q)':>10} {'total var':>10} "
          f"{'H(q)':>8} {'argmax=truth':>13}")
    for n, v in acc.items():
        mode = students[n][1].get("mode", "optp")
        print(f"{n:<16} {mode:>6} {v['kl']/tot:10.4f} {v['tv']/tot:10.4f} "
              f"{v['ent']/tot:8.3f} {v['agree']/tot:13.4f}")
    print("\nA correct C-PTP student should have KL near zero and H(q) near the "
          "teacher's;\nan O-PTP student should have H(q) near zero and a large KL.")
    print("CPTP_CHECK_DONE")


if __name__ == "__main__":
    main()
