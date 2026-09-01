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
    def __init__(self, feat=128, in_ch=1):
        super().__init__()
        # Global average pooling threw away the spatial layout, which is most of
        # what tells a 6 from a 9; that classifier stalled at 0.944 and the
        # assertion caught it.
        # Batch norm, because without it this reached 0.8428 on CIFAR against a
        # 0.85 bar -- and a classifier that weak cannot define a feature space
        # worth measuring Frechet distances in. MNIST cleared 0.99 either way.
        def block(i, o, pool=False):
            layers = [nn.Conv2d(i, o, 3, 1, 1), nn.BatchNorm2d(o), nn.ReLU()]
            return layers + ([nn.MaxPool2d(2)] if pool else [])
        self.body = nn.Sequential(
            *block(in_ch, 32), *block(32, 32, pool=True),     # 32 -> 16
            *block(32, 64), *block(64, 64, pool=True),        # 16 -> 8
            *block(64, 128), *block(128, 128, pool=True),     # 8 -> 4
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
def sample_within_scale_ar(mv, vae, teacher, labels, top_k=0, top_p=0.0,
                           cfg=0.0):
    """Sample the within-scale AR teacher: one forward per token inside a scale.

    This is the ceiling a PTP student distilled from it could reach, measured in
    the same units as the students rather than in per-token cross-entropy, which
    is not comparable across ladders of different length: (1,8) costs 94.7 nats
    an image over 65 tokens where (1,2,4,8) costs 120.8 over 85.

    Slot i reads the embedding of the token at i-1 of its own scale, so the
    sequence is filled left to right and the whole forward is repeated; 65 passes
    for this ladder. Slow by construction, and only ever run offline.

    It is guided the way VAR guides itself -- the same null class it was trained
    to drop to, the same ramp across scales. Without that this arm was the only
    unguided one in the table, and guidance is worth a factor of three here.
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
        # VAR ramps guidance linearly across scales; match it exactly.
        t = cfg * si / max(len(bounds) - 1, 1)
        null = torch.full_like(labels, teacher.num_classes)
        for j in range(a, b):
            logits = teacher(labels, x_in, truth=tokens).float()[:, j]
            if t > 0:
                un = teacher(null, x_in, truth=tokens).float()[:, j]
                logits = (1 + t) * logits - t * un
            tokens[:, j] = draw(logits, top_k, top_p)
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
    name = "cifar10" if DATASET == "cifar10" else "mnist"
    ck = f"/home/mengy13/ptp-vqvae/checkpoints/ar_{name}_raster.pt"
    teacher, meta = build(ck, device=device, dtype=torch.float32)
    teacher.eval()
    h, w, bos = meta["h"], meta["w"], meta["num_codes"]
    seq_len = h * w
    # ptp-vqvae has its own top-level `models` package, which shadows VAR's for
    # every import after this one. Put the path back and drop what it loaded.
    cwd, root = _os.getcwd(), "/home/mengy13/ptp-vqvae"
    saved = list(sys.path)
    before = set(sys.modules)
    sys.path.insert(0, root)
    _os.chdir(root)
    try:
        from models.ar import seq_to_codes_grid
        from utils.helper import load_vqvae
        vq, _ = load_vqvae(name, device)
    finally:
        _os.chdir(cwd)
        sys.path[:] = saved
        for m in set(sys.modules) - before:
            f = getattr(sys.modules[m], "__file__", None) or ""
            if f.startswith(root):
                del sys.modules[m]
    inv = torch.argsort(meta["perm"]).to(device)
    out = []
    for _ in range(0, n, batch):
        seq = torch.full((batch, 1), bos, dtype=torch.long, device=device)
        for _ in range(seq_len):
            lg = teacher(input_ids=seq).logits[:, -1].float()
            lg[:, bos] = -torch.inf
            seq = torch.cat([seq, draw(lg, top_k, top_p).unsqueeze(1)], 1)
        img = vq.decode(seq_to_codes_grid(seq[:, 1:], inv, h, w)).float()
        if img.shape[-1] != 32:      # MNIST decodes to 28; CIFAR is already 32
            img = F.pad(img, (2, 2, 2, 2), value=-1.0)
        out.append(img.cpu())
    del teacher, vq
    torch.cuda.empty_cache()
    return torch.cat(out)[:n]


TOP_K, TOP_P = int(os.environ.get('TOP_K', 0)), float(os.environ.get('TOP_P', 0.0))
CFG = float(os.environ.get('CFG', 0.0))
STUDENT_TAGS = [t for t in os.environ.get(
    'STUDENT_TAGS', 'var_ptp_student').split(',') if t]
DATASET = os.environ.get('DATASET', 'mnist')
CHANNELS = 3 if DATASET == 'cifar10' else 1
RES = "/home/mengy13/ptp-image-results"
DIR_R8 = os.environ.get('DIR_R8', f"{RES}/mnist_var_r8")
DIR_R2 = os.environ.get('DIR_R2', f"{RES}/mnist_var_r2")
# This small CNN reaches 0.99 on MNIST and nothing like it on CIFAR; the point of
# the assertion is to catch a broken classifier, so the bar has to follow the
# dataset rather than stay at a number only one of them can clear.
MIN_ACC = float(os.environ.get('MIN_ACC', 0.85 if DATASET == 'cifar10' else 0.98))
WITH_RASTER = os.environ.get('WITH_RASTER', '1') == '1'
TEACHER_TAG = os.environ.get('TEACHER_TAG', 'var_within_scale')
# Comparing several students means re-running the arms they share; the
# teacher costs 130 forwards an image at cfg > 0, so skipping it matters.
ARMS = set(os.environ.get('ARMS', 'raster,par8,ptp,ar8,par2468').split(','))


def main():
    torch.manual_seed(SEED)
    from image_ptp import mnist_var as mv_boot     # for mnist_loader only
    os.environ["PATCH"] = "1,8"
    os.environ["OUT"] = DIR_R8

    # ---- classifier on real MNIST ----
    ckpt = Path(os.environ.get("CLF_CKPT", f"{RES}/var_score_clf_{DATASET}_bn.pt"))
    clf = Classifier(in_ch=CHANNELS).to(device)
    test = mv_boot.mnist_loader(False, batch=512, workers=2)
    if ckpt.exists():
        clf.load_state_dict(torch.load(ckpt, map_location="cpu"))
        print(f"scoring classifier loaded from {ckpt}", flush=True)
    else:
        # CIFAR needs a real recipe here: random crop as well as flip, many more
        # epochs, and a schedule. Eight epochs of the MNIST recipe reached 0.758,
        # and the assertion below rightly refused it.
        from torchvision import datasets, transforms
        from torch.utils.data import DataLoader
        epochs = 60 if DATASET == "cifar10" else 8
        if DATASET == "cifar10":
            tfm = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5,) * 3, (0.5,) * 3)])
            ds = datasets.CIFAR10(root="/home/mengy13/ptp-vqvae/data/cifar",
                                  train=True, download=False, transform=tfm)
            train = DataLoader(ds, batch_size=256, shuffle=True, num_workers=4,
                               drop_last=True, persistent_workers=True)
        else:
            train = mv_boot.mnist_loader(True, batch=256)
        opt = torch.optim.AdamW(clf.parameters(), lr=1e-3, weight_decay=5e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs * len(train))
        for ep in range(epochs):
            clf.train()
            for x, y in train:
                loss = F.cross_entropy(clf(x.to(device)), y.to(device))
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                sched.step()

        torch.save(clf.state_dict(), ckpt)
        print(f"scoring classifier trained and cached to {ckpt}", flush=True)
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
    assert acc > MIN_ACC, "the classifier is too weak for its verdicts to mean anything"
    feats_real = torch.cat(feats_real)
    labels_real = torch.cat(labels_real)

    # MNIST's test set is stored as two differently-sourced blocks, so splitting
    # it at the midpoint measures that shift rather than sampling noise -- it put
    # the floor at 89 while a generated arm scored 48. Shuffle first, and draw the
    # same count the arms use so the small-sample bias is the same one.
    perm = torch.randperm(feats_real.shape[0], generator=torch.Generator().manual_seed(0))
    print(f"floor: {N_SAMPLES} real against all real -> "
          f"FID {frechet(feats_real[perm[:N_SAMPLES]], feats_real):.3f}"
          f"   (midpoint split, unshuffled: "
          f"{frechet(feats_real[:feats_real.shape[0]//2], feats_real[feats_real.shape[0]//2:]):.3f}"
          f" -- that is the block shift, not a floor)", flush=True)

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
    print(f"\n===== sampling: top_k={TOP_K} top_p={TOP_P} cfg={CFG} ({how}) ====="
          "\n      the PTP student takes no cfg: its output is meant to be the\n"
          "      one-hot the teacher CDF selects, and there is no second pass\n"
          "      to mix against without changing what it was distilled to be.\n",
          flush=True)
    want = torch.arange(10).repeat_interleave(N_SAMPLES // 10)
    lab = want.to(device)

    if WITH_RASTER:
        img = sample_raster_ar(n=N_SAMPLES, top_k=TOP_K, top_p=TOP_P)
        with torch.no_grad():
            f = torch.cat([clf.features(img[i:i + 256].to(device)).cpu()
                           for i in range(0, img.shape[0], 256)])
        print(f"{'raster AR (uncond)':<30} label acc    n/a   "
              f"FID {frechet(f, feats_real):8.3f}", flush=True)

    # The tokeniser's own ceiling: real held-out images encoded and decoded, no
    # generation at all. Every arm decodes through this, so nothing can score
    # better than it, and on CIFAR the VQVAE reaches only ~20 dB where MNIST
    # reaches ~24. Without this row there is no way to tell a bad sampler from a
    # bad codebook.
    def recon_ceiling(mv, vae, name):
        """Real held-out images through the tokeniser and back, no generation.

        Every arm decodes through this quantiser, so nothing can score better
        than it. On CIFAR the VQVAE reaches about 20 dB against MNIST's 24, and
        without this row a lossy codebook and a bad sampler look identical.
        """
        imgs, labs, n = [], [], 0
        for x, y in mv.data_loader(False, batch=256, workers=2):
            if n >= N_SAMPLES:
                break
            with torch.no_grad():
                rec = vae(x.to(device))[0].clamp(-1, 1)   # forward = encode,
            imgs.append(rec.cpu())                        # quantise, decode
            labs.append(y)
            n += x.shape[0]
        score(name, torch.cat(imgs)[:N_SAMPLES], torch.cat(labs)[:N_SAMPLES])

    mv8, vae8, var8 = load_ladder("1,8", DIR_R8)
    torch.manual_seed(SEED)
    with torch.no_grad():
        img = torch.cat([var8.autoregressive_infer_cfg(
            B=lab[i:i + 256].shape[0], label_B=lab[i:i + 256], cfg=CFG,
            top_k=TOP_K, top_p=TOP_P).cpu()
            for i in range(0, lab.shape[0], 256)])
    recon_ceiling(mv8, vae8, "(1,8) VQVAE recon, 0 fwd")
    score("(1,8) parallel, 2 fwd", img * 2 - 1, want)     # [0,1] -> [-1,1]
    del var8
    torch.cuda.empty_cache()

    # Every student is scored in the same run, against the same cached
    # classifier and the same arms, so the numbers are one table and not several.
    for tag in STUDENT_TAGS:
        student = build_ptp_student(vae8, mv8.PATCH, adapter="binary",
                                    num_classes=mv8.NUM_CLASSES, device=device)
        student.load_state_dict(torch.load(Path(mv8.OUT) / f"{tag}.pt",
                                           map_location="cpu"))
        student.eval()
        student.cond_drop_rate = 0.0
        torch.manual_seed(SEED)
        with torch.no_grad():
            img = torch.cat([decode(mv8, vae8, sample_ptp(
                mv8, vae8, student, lab[i:i + 256])).cpu()
                for i in range(0, lab.shape[0], 256)])
        score(f"(1,8) + PTP [{tag}], 2 fwd", img, want)      # already [-1,1]
        del student
        torch.cuda.empty_cache()
    del vae8
    torch.cuda.empty_cache()

    # The teacher the PTP student was distilled from: its own samples are the
    # ceiling, in the same metric.
    from image_ptp.within_scale_var import build_within_scale_var
    mv8b, vae8b, _ = load_ladder("1,8", DIR_R8)
    tea = build_within_scale_var(vae8b, mv8b.PATCH, num_classes=mv8b.NUM_CLASSES,
                                 device=device)
    tea.load_state_dict(torch.load(Path(mv8b.OUT) / f"{TEACHER_TAG}.pt",
                                   map_location="cpu"))
    tea.eval()
    tea.cond_drop_rate = 0.0
    with torch.no_grad():
        img = torch.cat([decode(mv8b, vae8b, sample_within_scale_ar(
            mv8b, vae8b, tea, lab[i:i + 256], TOP_K, TOP_P, CFG)).cpu()
            for i in range(0, lab.shape[0], 256)])
    score(f"(1,8) AR [{TEACHER_TAG}], 65 fwd", img, want)
    del tea, vae8b
    torch.cuda.empty_cache()

    mv2, vae2, var2 = load_ladder("1,2,4,8", DIR_R2)
    torch.manual_seed(SEED)
    with torch.no_grad():
        img = torch.cat([var2.autoregressive_infer_cfg(
            B=lab[i:i + 256].shape[0], label_B=lab[i:i + 256], cfg=CFG,
            top_k=TOP_K, top_p=TOP_P).cpu()
            for i in range(0, lab.shape[0], 256)])
    recon_ceiling(mv2, vae2, "(1,2,4,8) VQVAE recon, 0 fwd")
    score("(1,2,4,8) parallel, 4 fwd", img * 2 - 1, want)
    print("SCORE_DONE", flush=True)


if __name__ == "__main__":
    main()
