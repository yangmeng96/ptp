"""Reconstruction quality per tokeniser, in the metric the samplers are judged in.

PSNR is what the VQVAE training loop reports and it is not the quantity that
decides anything here: a plain MSE autoencoder can hold a respectable PSNR while
blurring away the high frequencies, and blur is what Frechet distance punishes
hardest. Published tokenisers that reach low reconstruction FID -- VQGAN,
LlamaGen's, VAR's -- all add a perceptual loss and a discriminator on top of MSE,
which is the part this project's tokeniser does not have.

So the capacity sweep has to be read in FID, not PSNR. If widening the network
moves PSNR and leaves reconstruction FID where it was, capacity was not the
binding constraint and the loss is.

    python -m image_ptp.recon_fid name=/path/vqvae.pt:ch:cvae:vocab:patch ...
"""
import argparse
import importlib
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def load_vqvae(ckpt, ch, cvae, vocab, patch, dataset, device):
    """Each swept tokeniser has its own width, so the module has to be rebuilt:
    mnist_var reads CH/CVAE/VOCAB/PATCH from the environment at import time."""
    os.environ.update(CH=str(ch), CVAE=str(cvae), VOCAB=str(vocab),
                      PATCH=patch, DATASET=dataset)
    for m in [k for k in sys.modules if k.startswith("image_ptp.mnist_var")]:
        del sys.modules[m]
    mv = importlib.reload(importlib.import_module("image_ptp.mnist_var"))
    mv.ensure_process_group()
    vae = mv.MnistVQVAE().to(device)
    vae.load_state_dict(torch.load(ckpt, map_location="cpu"))
    return mv, vae.eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("specs", nargs="+")
    ap.add_argument("--dataset", default="cifar10")
    ap.add_argument("--images", type=int, default=2000)
    args = ap.parse_args()
    device = "cuda"
    os.environ["DATASET"] = args.dataset

    # The project's own small classifier defines a space nothing published can be
    # read against; InceptionV3's cached weights give the standard one. Both are
    # available so the switch can be checked rather than assumed.
    net = os.environ.get("FID_NET", "inception")
    if net == "inception":
        from image_ptp.fid_features import InceptionFeatures, frechet
        ext = InceptionFeatures(device=device)
        feat = ext
        head = None
    else:
        from image_ptp.score_var_samples import Classifier, frechet
        ch_in = 3 if args.dataset == "cifar10" else 1
        clf_path = Path(
            f"/home/mengy13/ptp-image-results/var_score_clf_{args.dataset}_bn.pt")
        assert clf_path.exists(), f"{clf_path} missing; run the scorer once first"
        clf = Classifier(in_ch=ch_in).to(device)
        clf.load_state_dict(torch.load(clf_path, map_location="cpu"))
        clf.eval()
        feat, head = clf.features, clf.head

    # Real features once, from the same loader every arm will use.
    os.environ.update(PATCH="1,8", DATASET=args.dataset)
    for m in [k for k in sys.modules if k.startswith("image_ptp.mnist_var")]:
        del sys.modules[m]
    boot = importlib.import_module("image_ptp.mnist_var")
    real, labels = [], []
    for x, y in boot.data_loader(False, batch=256, workers=2):
        real.append(x)
        labels.append(y)
        if sum(t.shape[0] for t in real) >= args.images:
            break
    real = torch.cat(real)[:args.images]
    labels = torch.cat(labels)[:args.images]
    with torch.no_grad():
        f_real = torch.cat([feat(real[i:i + 256].to(device)).cpu()
                            for i in range(0, real.shape[0], 256)])
    print(f"{args.dataset}: {real.shape[0]} held-out images, "
          f"{net} features {f_real.shape[1]}-d\n")
    print(f"{'tokeniser':<18} {'ch':>4} {'Cvae':>5} {'vocab':>6} "
          f"{'PSNR dB':>8} {'recon FID':>10} {'label acc':>10}")

    for spec in args.specs:
        name, rest = spec.split("=", 1)
        ckpt, ch, cvae, vocab, patch = rest.rsplit(":", 4)
        if not Path(ckpt).exists():
            print(f"{name:<18} <missing {ckpt}>")
            continue
        _, vae = load_vqvae(ckpt, int(ch), int(cvae), int(vocab), patch,
                            args.dataset, device)
        recs, se, n = [], 0.0, 0
        with torch.no_grad():
            for i in range(0, real.shape[0], 256):
                x = real[i:i + 256].to(device)
                r = vae(x)[0].clamp(-1, 1)
                se += float(F.mse_loss(r, x, reduction="sum"))
                n += x.numel()
                recs.append(r.cpu())
        recs = torch.cat(recs)
        psnr = 10 * torch.log10(torch.tensor(4.0 / (se / n)))   # range is [-1,1]
        with torch.no_grad():
            f = torch.cat([feat(recs[i:i + 256].to(device)).cpu()
                           for i in range(0, recs.shape[0], 256)])
            pred = (torch.cat([head(f[i:i + 256].to(device)).argmax(-1).cpu()
                               for i in range(0, f.shape[0], 256)])
                    if head is not None else torch.full_like(labels, -1))
        print(f"{name:<18} {ch:>4} {cvae:>5} {vocab:>6} {float(psnr):8.2f} "
              f"{frechet(f, f_real):10.3f} "
              + (f"{float((pred == labels).float().mean()):10.4f}"
                 if head is not None else f"{'n/a':>10}"))
        del vae
        torch.cuda.empty_cache()
    print("RECON_FID_DONE")


if __name__ == "__main__":
    main()
