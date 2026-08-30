"""Put numbers on the three MNIST ladders' samples.

The grids said (1,8)+PTP looks like (1,2,4,8) and better than (1,8) alone, which
is the claim the reconstruction-term CE could not support. Eyeballing a hundred
digits is not a measurement, so score them:

  label accuracy   does the sample show the digit it was asked for, according to
                   a classifier trained on real MNIST
  FID              Frechet distance in that classifier's penultimate features
                   against the real test set -- fidelity and diversity together
  spread           mean pairwise feature distance inside one requested class,
                   which separates "clean but collapsed" from "clean and varied"

Real test images are scored the same way as a floor: their FID against a held-out
half of themselves is what a perfect sampler would reach here, not zero.

The three samplers do not agree on output range -- VAR's own sampler returns
[0,1] and fhat_to_img returns [-1,1] -- so everything is converted explicitly
before it reaches the classifier.
"""
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, "/home/mengy13/ptp")
os.environ.setdefault("VAR_ROOT", "/home/mengy13/VAR")
device = "cuda"
N_SAMPLES = int(os.environ.get("N_SAMPLES", 2000))
SEED = 0


class Classifier(nn.Module):
    def __init__(self, feat=128):
        super().__init__()
        # Global average pooling threw away the spatial layout, which is most of
        # what tells a 6 from a 9; that classifier stalled at 0.944 and the
        # assertion caught it.
        self.body = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),   # 32 -> 16
            nn.Conv2d(32, 64, 3, 1, 1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),   # 16 -> 8
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),  # 8 -> 4
            nn.Flatten(), nn.Linear(128 * 4 * 4, feat), nn.ReLU())
        self.head = nn.Linear(feat, 10)

    def features(self, x):
        return self.body(x)

    def forward(self, x):
        return self.head(self.body(x))


def frechet(a, b):
    """Frechet distance between two Gaussians fitted to feature sets."""
    mu_a, mu_b = a.mean(0), b.mean(0)
    ca = torch.cov(a.T.double())
    cb = torch.cov(b.T.double())
    diff = (mu_a - mu_b).double()
    # sqrt(ca @ cb) via the eigendecomposition of a symmetric similarity
    ea, va = torch.linalg.eigh(ca)
    ea = ea.clamp_min(0)
    half = va @ torch.diag(ea.sqrt()) @ va.T
    m = half @ cb @ half
    em = torch.linalg.eigvalsh(m).clamp_min(0)
    return float(diff.dot(diff) + ca.trace() + cb.trace() - 2 * em.sqrt().sum())



def draw(logits, top_k=0, top_p=0.0):
    """One token per row, through VAR's own truncation function.

    VAR truncates its sampling at top-k 100 / top-p 0.95; the within-scale
    teacher was drawing from the untruncated 512-way tail, 64 times per image,
    so it paid for that tail 64 times over. Calling VAR's helper rather than
    reimplementing it is what makes the arms identical rather than merely
    similar -- it mutates its input, hence the clone.
    """
    from models.helpers import sample_with_top_k_top_p_
    return sample_with_top_k_top_p_(
        logits.float().clone().unsqueeze(1), top_k=top_k, top_p=top_p
    ).view(-1)


@torch.no_grad()
def sample_within_scale_ar(mv, vae, teacher, labels, top_k=0, top_p=0.0):
    """Sample the within-scale AR teacher: one forward per token inside a scale.

    This is the ceiling a PTP student distilled from it could reach, measured in
    the same units as the students rather than in per-token cross-entropy, which
    is not comparable across ladders of different length: (1,8) costs 94.7 nats
    an image over 65 tokens where (1,2,4,8) costs 120.8 over 85.

    Slot i reads the embedding of the token at i-1 of its own scale, so the
    sequence is filled left to right and the whole forward is repeated; 65 passes
    for this ladder. Slow by construction, and only ever run offline.
    """
    B = labels.shape[0]
    L = teacher.L
    bounds, cur = [], 0
    for pn in mv.PATCH:
        bounds.append((cur, cur + pn * pn, pn))
        cur += pn * pn
    tokens = torch.zeros(B, L, dtype=torch.long, device=device)
    for si, (a, b, pn) in enumerate(bounds):
        if si == 0:
            x_in = torch.zeros(B, L - teacher.first_l, vae.Cvae, device=device)
        else:
            gt = [tokens[:, s0:s1] for s0, s1, _ in bounds[:si]]
            x_in = vae.quantize.idxBl_to_var_input(gt)
            if x_in is None:
                x_in = torch.zeros(B, 0, vae.Cvae, device=device)
        for j in range(a, b):
            logits = teacher(labels, x_in, truth=tokens).float()
            tokens[:, j] = draw(logits[:, j], top_k, top_p)
    return [tokens[:, a:b] for a, b, _ in bounds]



@torch.no_grad()
def sample_raster_ar(clf_unused=None, n=2000, batch=250, top_k=0, top_p=0.0):
    """The ptp-vqvae MNIST AR teacher: 49 tokens, one forward each, no ladder.

    An absolute reference for what a fully autoregressive model reaches on this
    metric, sampled the same way as everything else -- no guidance, no
    truncation. It is unconditional, so it gets no label accuracy.
    """
    import os as _os
    from image_ptp.vqvae_ar_hf import build
    ck = "/home/mengy13/ptp-vqvae/checkpoints/ar_mnist_raster.pt"
    teacher, meta = build(ck, device=device, dtype=torch.float32)
    teacher.eval()
    h, w, bos = meta["h"], meta["w"], meta["num_codes"]
    seq_len = h * w
    cwd = _os.getcwd()
    sys.path.insert(0, "/home/mengy13/ptp-vqvae")
    _os.chdir("/home/mengy13/ptp-vqvae")
    from models.ar import seq_to_codes_grid
    from utils.helper import load_vqvae
    vq, _ = load_vqvae("mnist", device)
    _os.chdir(cwd)
    inv = torch.argsort(meta["perm"]).to(device)
    out = []
    for _ in range(0, n, batch):
        seq = torch.full((batch, 1), bos, dtype=torch.long, device=device)
        for _ in range(seq_len):
            lg = teacher(input_ids=seq).logits[:, -1].float()
            lg[:, bos] = -torch.inf
            seq = torch.cat([seq, draw(lg, top_k, top_p).unsqueeze(1)], 1)
        img = vq.decode(seq_to_codes_grid(seq[:, 1:], inv, h, w)).float()
        # 28x28 in [-1,1]; the scoring classifier is trained on 32x32
        out.append(F.pad(img, (2, 2, 2, 2), value=-1.0).cpu())
    del teacher, vq
    torch.cuda.empty_cache()
    return torch.cat(out)[:n]


TOP_K, TOP_P = int(os.environ.get('TOP_K', 0)), float(os.environ.get('TOP_P', 0.0))


def main():
    torch.manual_seed(SEED)
    from image_ptp import mnist_var as mv_boot     # for mnist_loader only
    os.environ["PATCH"] = "1,8"
    os.environ["OUT"] = "/home/mengy13/ptp-image-results/mnist_var_r8"

    # ---- classifier on real MNIST ----
    clf = Classifier().to(device)
    opt = torch.optim.AdamW(clf.parameters(), lr=1e-3, weight_decay=1e-4)
    train = mv_boot.mnist_loader(True, batch=256)
    test = mv_boot.mnist_loader(False, batch=512, workers=2)
    for ep in range(8):
        clf.train()
        for x, y in train:
            loss = F.cross_entropy(clf(x.to(device)), y.to(device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    clf.eval()
    with torch.no_grad():
        hit = n = 0
        feats_real, labels_real = [], []
        for x, y in test:
            x = x.to(device)
            f = clf.features(x)
            hit += int((clf.head(f).argmax(-1) == y.to(device)).sum())
            n += x.shape[0]
            feats_real.append(f.cpu())
            labels_real.append(y)
    acc = hit / n
    print(f"classifier test accuracy {acc:.4f}", flush=True)
    assert acc > 0.98, "the classifier is too weak for its verdicts to mean anything"
    feats_real = torch.cat(feats_real)
    labels_real = torch.cat(labels_real)

    half = feats_real.shape[0] // 2
    floor = frechet(feats_real[:half], feats_real[half:])
    print(f"floor: real against real, split in half -> FID {floor:.3f}", flush=True)

    # ---- the three samplers ----
    from image_ptp.sample_var_ptp import load_ladder, sample_ptp, decode
    from image_ptp.within_scale_var import build_ptp_student

    def score(name, images, want):
        with torch.no_grad():
            f = torch.cat([clf.features(images[i:i + 256].to(device)).cpu()
                           for i in range(0, images.shape[0], 256)])
            pred = torch.cat([clf.head(f[i:i + 256].to(device)).argmax(-1).cpu()
                              for i in range(0, f.shape[0], 256)])
        lab_acc = float((pred == want).float().mean())
        fid = frechet(f, feats_real)
        spreads = []
        for c in range(10):
            sel = f[want == c]
            if sel.shape[0] > 2:
                spreads.append(torch.cdist(sel, sel).sum() /
                               (sel.shape[0] ** 2 - sel.shape[0]))
        spread = float(torch.stack(spreads).mean())
        print(f"{name:<26} label acc {lab_acc:.4f}   FID {fid:8.3f}   "
              f"spread {spread:.3f}", flush=True)

    how = "untruncated" if not (TOP_K or TOP_P) else "VAR's own setting"
    print(f"\n===== sampling: top_k={TOP_K} top_p={TOP_P} cfg=0 ({how}) =====\n",
          flush=True)
    want = torch.arange(10).repeat_interleave(N_SAMPLES // 10)
    lab = want.to(device)

    img = sample_raster_ar(n=N_SAMPLES, top_k=TOP_K, top_p=TOP_P)
    with torch.no_grad():
        f = torch.cat([clf.features(img[i:i + 256].to(device)).cpu()
                       for i in range(0, img.shape[0], 256)])
    print(f"{'raster AR, 49 fwd (uncond)':<30} label acc    n/a   "
          f"FID {frechet(f, feats_real):8.3f}", flush=True)

    mv8, vae8, var8 = load_ladder("1,8", "/home/mengy13/ptp-image-results/mnist_var_r8")
    torch.manual_seed(SEED)
    with torch.no_grad():
        img = torch.cat([var8.autoregressive_infer_cfg(
            B=lab[i:i + 256].shape[0], label_B=lab[i:i + 256], cfg=0.0,
            top_k=TOP_K, top_p=TOP_P).cpu()
            for i in range(0, lab.shape[0], 256)])
    score("(1,8) parallel, 2 fwd", img * 2 - 1, want)     # [0,1] -> [-1,1]
    del var8
    torch.cuda.empty_cache()

    student = build_ptp_student(vae8, mv8.PATCH, adapter="binary",
                                num_classes=mv8.NUM_CLASSES, device=device)
    student.load_state_dict(torch.load(Path(mv8.OUT) / "var_ptp_student.pt",
                                       map_location="cpu"))
    student.eval()
    student.cond_drop_rate = 0.0
    with torch.no_grad():
        img = torch.cat([decode(mv8, vae8, sample_ptp(
            mv8, vae8, student, lab[i:i + 256])).cpu()
            for i in range(0, lab.shape[0], 256)])
    score("(1,8) + PTP, 2 fwd", img, want)                # already [-1,1]
    del student, vae8
    torch.cuda.empty_cache()

    # The teacher the PTP student was distilled from: its own samples are the
    # ceiling, in the same metric.
    from image_ptp.within_scale_var import build_within_scale_var
    mv8b, vae8b, _ = load_ladder("1,8", "/home/mengy13/ptp-image-results/mnist_var_r8")
    tea = build_within_scale_var(vae8b, mv8b.PATCH, num_classes=mv8b.NUM_CLASSES,
                                 device=device)
    tea.load_state_dict(torch.load(Path(mv8b.OUT) / "var_within_scale.pt",
                                   map_location="cpu"))
    tea.eval()
    tea.cond_drop_rate = 0.0
    with torch.no_grad():
        img = torch.cat([decode(mv8b, vae8b, sample_within_scale_ar(
            mv8b, vae8b, tea, lab[i:i + 256], TOP_K, TOP_P)).cpu()
            for i in range(0, lab.shape[0], 256)])
    score("(1,8) within-scale AR, 65 fwd", img, want)
    del tea, vae8b
    torch.cuda.empty_cache()

    mv2, vae2, var2 = load_ladder("1,2,4,8",
                                  "/home/mengy13/ptp-image-results/mnist_var_r2")
    torch.manual_seed(SEED)
    with torch.no_grad():
        img = torch.cat([var2.autoregressive_infer_cfg(
            B=lab[i:i + 256].shape[0], label_B=lab[i:i + 256], cfg=0.0,
            top_k=TOP_K, top_p=TOP_P).cpu()
            for i in range(0, lab.shape[0], 256)])
    score("(1,2,4,8) parallel, 4 fwd", img * 2 - 1, want)
    print("SCORE_DONE", flush=True)


if __name__ == "__main__":
    main()
