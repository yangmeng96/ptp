"""Is the within-scale AR teacher's sampler faithful, and is the teacher itself sound?

Its samples score worse than the PTP student distilled from it, which cannot be
true of a correct teacher and a correct sampler. Three things could produce it:
the incremental loop could differ from the forward the teacher was trained on;
the teacher could be memorising rather than modelling; or free running could
diverge from teacher forcing far faster than the CE suggests. Each is measured
here separately.
"""
import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="/home/mengy13/ptp-image-results/mnist_var_r8")
    ap.add_argument("--scales", default="1,8")
    args = ap.parse_args()
    device = "cuda"

    import image_ptp.mnist_var as mv
    from image_ptp.within_scale_var import build_within_scale_var, var_input
    mv.PATCH = tuple(int(v) for v in args.scales.split(","))
    mv.OUT = args.out_dir
    mv.ensure_process_group()

    vae = mv.MnistVQVAE().to(device)
    vae.load_state_dict(torch.load(Path(args.out_dir) / "vqvae.pt", map_location="cpu"))
    vae.eval().requires_grad_(False)
    tea = build_within_scale_var(vae, mv.PATCH, num_classes=mv.NUM_CLASSES, device=device)
    tea.load_state_dict(torch.load(Path(args.out_dir) / "var_within_scale.pt",
                                   map_location="cpu"))
    tea.eval().requires_grad_(False)
    tea.cond_drop_rate = 0.0

    bounds, cur = [], 0
    for pn in mv.PATCH:
        bounds.append((cur, cur + pn * pn, pn))
        cur += pn * pn
    L = cur

    x, y = next(iter(mv.mnist_loader(False, batch=128, workers=2)))
    x, y = x.to(device), y.to(device)
    gt = vae.img_to_idxBl(x)
    truth = torch.cat(gt, 1)
    B = truth.shape[0]

    # ---- 1. the sampler, replayed on ground truth, must reproduce training's forward
    ref = tea(y, var_input(vae, gt, device), truth=truth).float()
    tok = torch.zeros_like(truth)
    worst = 0.0
    for si, (a, b, pn) in enumerate(bounds):
        if si == 0:
            x_in = torch.zeros(B, L - tea.first_l, vae.Cvae, device=device)
        else:
            x_in = vae.quantize.idxBl_to_var_input([tok[:, s0:s1] for s0, s1, _ in bounds[:si]])
        for j in range(a, b):
            lg = tea(y, x_in, truth=tok).float()[:, j]
            worst = max(worst, float((lg - ref[:, j]).abs().max()))
            tok[:, j] = truth[:, j]        # force, so the two paths stay aligned
    print(f"[1] sampler vs training forward, forced on truth: max |dlogit| {worst:.2e}")
    print("    (any value above ~1e-3 means the incremental loop is not the "
          "forward the teacher was trained on)", flush=True)

    # ---- 2. memorisation: teacher-forced CE on train vs test
    for name, loader in (("train", mv.mnist_loader(True, batch=256, workers=2)),
                         ("test", mv.mnist_loader(False, batch=256, workers=2))):
        tot = n = 0.0
        with torch.no_grad():
            for i, (xb, yb) in enumerate(loader):
                if i == 20:
                    break
                xb, yb = xb.to(device), yb.to(device)
                g = vae.img_to_idxBl(xb)
                t = torch.cat(g, 1)
                lg = tea(yb, var_input(vae, g, device), truth=t)
                tot += float(F.cross_entropy(lg.reshape(-1, mv.VOCAB), t.reshape(-1))) * xb.shape[0]
                n += xb.shape[0]
        print(f"[2] teacher-forced CE, {name}: {tot/n:.4f} nats/token", flush=True)

    # ---- 3. free running: how fast does the model leave the distribution it was
    #        trained on? Measured as the entropy it predicts under its own prefix
    #        versus under a real one, at the same positions.
    a, b, _ = bounds[-1]
    for label, use_own in (("real prefix", False), ("own samples", True)):
        tok = truth.clone()
        ent = torch.zeros(b - a, device=device)
        x_in = vae.quantize.idxBl_to_var_input([truth[:, s0:s1] for s0, s1, _ in bounds[:-1]])
        for j in range(a, b):
            p = torch.softmax(tea(y, x_in, truth=tok).float()[:, j], -1)
            ent[j - a] = -(p * (p + 1e-12).log()).sum(-1).mean()
            tok[:, j] = torch.multinomial(p, 1).squeeze(1) if use_own else truth[:, j]
        q = [ent[:8].mean(), ent[8:24].mean(), ent[24:48].mean(), ent[48:].mean()]
        print(f"[3] predicted entropy under {label:12s}: "
              + "  ".join(f"{float(v):.3f}" for v in q), flush=True)
    print("AUDIT_DONE", flush=True)


if __name__ == "__main__":
    main()
