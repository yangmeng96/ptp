"""The merged-scale experiment on MNIST, where it should have been done first.

The ImageNet work asks whether VAR's leading scales can be emitted in one
forward and the lost cross-scale conditioning put back with PTP. That is a
different experiment from the one MNIST validated earlier -- that one coarsened
the ladder and distilled a within-scale AR teacher -- so nothing here has been
checked at small scale. It is also a hundred times cheaper here: the teacher and
the tokeniser already exist, and the data prep is minutes rather than hours.

    (1,2,4,8), 85 tokens        merge 2 ->  5 tokens, 3 forwards
                                merge 3 -> 21 tokens, 2 forwards

ImageNet's merge=4 is 30 tokens in one forward, so merge=3 here is the closest
match in shape.
"""
import argparse
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, "/home/mengy13/VAR")


def load(out_dir, patch):
    os.environ["PATCH"] = patch
    os.environ["OUT"] = out_dir
    os.environ.setdefault("DATASET", "mnist")
    for m in [k for k in sys.modules if k.startswith("image_ptp.mnist_var")]:
        del sys.modules[m]
    import importlib
    mv = importlib.reload(importlib.import_module("image_ptp.mnist_var"))
    mv.ensure_process_group()
    vae, var = mv.build_var()
    vae.load_state_dict(torch.load(Path(out_dir) / "vqvae.pt", map_location="cpu"))
    var.load_state_dict(torch.load(Path(out_dir) / "var.pt", map_location="cpu"))
    vae.eval(); var.eval()
    var.cond_drop_rate = 0.0          # forward drops labels in any mode otherwise
    return mv, vae, var


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/home/mengy13/ptp-image-results/mnist_var_r2")
    ap.add_argument("--patch", default="1,2,4,8")
    ap.add_argument("--merge", default="1,2,3")
    ap.add_argument("--images", type=int, default=10000)
    ap.add_argument("--batch", type=int, default=250)
    ap.add_argument("--cfg", type=float, default=1.5)
    args = ap.parse_args()
    device = "cuda"
    torch.set_grad_enabled(False)

    mv, vae, var = load(args.dir, args.patch)
    from image_ptp.merge_sample import sample_merged, expected_prefix
    from image_ptp.fid_features import (InceptionFeatures, frechet,
                                        check_sample_count)
    ext = InceptionFeatures(device=device)

    real, labels = [], []
    for x, y in mv.data_loader(False, batch=256, workers=4):
        real.append(x); labels.append(y)
        if sum(t.shape[0] for t in real) >= args.images:
            break
    real = torch.cat(real)[:args.images]
    f_real = torch.cat([ext(real[i:i + 256].to(device)).cpu()
                        for i in range(0, real.shape[0], 256)])
    check_sample_count(f_real.shape[0], f_real.shape[1])

    # The tokeniser's own reconstruction, so the arms are read against what the
    # codebook allows rather than against the real images alone.
    rec = torch.cat([vae(real[i:i + 256].to(device))[0].clamp(-1, 1).cpu()
                     for i in range(0, real.shape[0], 256)])
    f_rec = torch.cat([ext(rec[i:i + 256].to(device)).cpu()
                       for i in range(0, rec.shape[0], 256)])
    print(f"{real.shape[0]} MNIST test images, patch {var.patch_nums}")
    print(f"{'arm':<28} {'forwards':>9} {'FID':>9}")
    print(f"{'tokeniser reconstruction':<28} {0:>9} {frechet(f_rec, f_real):9.3f}",
          flush=True)

    for prefix_mode in ("zero", "expected"):
        for k in (int(v) for v in args.merge.split(",")):
            if k == 1 and prefix_mode == "expected":
                continue                       # nothing to prefix
            pre = ([t.to(device) for t in expected_prefix(var, k, cfg=args.cfg)]
                   if (prefix_mode == "expected" and k > 1) else None)
            feats = []
            for i in range(0, args.images, args.batch):
                b = min(args.batch, args.images - i)
                lab = torch.randint(0, mv.NUM_CLASSES, (b,), device=device,
                                    generator=torch.Generator(device=device).manual_seed(i))
                img = sample_merged(var, b, lab, k, cfg=args.cfg, top_k=100,
                                    top_p=0.95, g_seed=i, prefix=pre)
                feats.append(ext(img * 2 - 1).cpu())       # [0,1] -> [-1,1]
            tok = sum(p * p for p in var.patch_nums[:k])
            name = (f"merge {k} ({tok} tok), {prefix_mode}" if k > 1
                    else "official sampler")
            print(f"{name:<28} {len(var.patch_nums) - k + 1:>9} "
                  f"{frechet(torch.cat(feats), f_real):9.3f}", flush=True)
    print("MNIST_MERGE_DONE")


if __name__ == "__main__":
    main()
