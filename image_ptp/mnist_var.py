"""A small VAR on MNIST, to check whether within-scale independence travels.

The measurement on the pretrained ImageNet VAR said a scale carries essentially
no information about itself: a head shown every other token of the scale scored
0.04% better than one shown none, with 86.7M trainable parameters per arm and
the model's top blocks unfrozen. The explanation offered for that -- a residual
quantiser removes, at each scale, whatever the previous scales could predict, so
what is left is close to spatially white -- is a claim about residual multi-scale
quantisation in general, not about that one checkpoint. This trains an
independent one on different data to see whether the null repeats.

`VectorQuantizer2` and `VAR` are reused unchanged, since they are the part the
claim is about. Only the autoencoder around them is new: VAR's own is wired for
3-channel 256px input with 16x downsampling, which leaves a 2x2 latent on MNIST.

Stages are separate so each can be checkpointed and rerun:
    STAGE=vqvae   train the tokeniser
    STAGE=var     train the transformer on its tokens
    STAGE=sample  decode a grid, to confirm the thing generates digits at all
    STAGE=loo     the leave-one-out measurement

`VectorQuantizer2.forward` calls `torch.distributed.get_world_size()` outside any
initialisation guard, so a single-process group has to exist before training.
"""
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

VAR_ROOT = Path(os.environ.get("VAR_ROOT", "/home/mengy13/VAR"))
sys.path.insert(0, str(VAR_ROOT))

OUT = Path(os.environ.get("OUT", "/home/mengy13/ptp-image-results/mnist_var"))
PATCH = tuple(int(x) for x in os.environ.get("PATCH", "1,2,3,4,6,8").split(","))
VOCAB = int(os.environ.get("VOCAB", 512))
CVAE = int(os.environ.get("CVAE", 32))
IMG = 32
NUM_CLASSES = 10
device = "cuda"


def ensure_process_group():
    import torch.distributed as tdist
    if not tdist.is_initialized():
        tdist.init_process_group(backend="gloo", init_method="tcp://127.0.0.1:29517",
                                 rank=0, world_size=1)


class ResBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.net = nn.Sequential(
            nn.GroupNorm(8, c), nn.SiLU(), nn.Conv2d(c, c, 3, 1, 1),
            nn.GroupNorm(8, c), nn.SiLU(), nn.Conv2d(c, c, 3, 1, 1))

    def forward(self, x):
        return x + self.net(x)


class MnistVQVAE(nn.Module):
    """The attributes VAR reads off its tokeniser: Cvae, vocab_size, quantize."""

    def __init__(self, ch=128):
        super().__init__()
        from models.quant import VectorQuantizer2
        self.Cvae, self.vocab_size = CVAE, VOCAB
        self.encoder = nn.Sequential(
            nn.Conv2d(1, ch // 2, 4, 2, 1), nn.SiLU(),       # 32 -> 16
            nn.Conv2d(ch // 2, ch, 4, 2, 1),                 # 16 -> 8
            ResBlock(ch), ResBlock(ch),
            nn.GroupNorm(8, ch), nn.SiLU(), nn.Conv2d(ch, CVAE, 1))
        self.decoder = nn.Sequential(
            nn.Conv2d(CVAE, ch, 3, 1, 1), ResBlock(ch), ResBlock(ch),
            nn.GroupNorm(8, ch), nn.SiLU(),
            nn.ConvTranspose2d(ch, ch // 2, 4, 2, 1), nn.SiLU(),
            nn.ConvTranspose2d(ch // 2, 1, 4, 2, 1))
        self.quant_conv = nn.Conv2d(CVAE, CVAE, 3, 1, 1)
        self.post_quant_conv = nn.Conv2d(CVAE, CVAE, 3, 1, 1)
        self.quantize = VectorQuantizer2(
            vocab_size=VOCAB, Cvae=CVAE, using_znorm=False, beta=0.25,
            default_qresi_counts=0, v_patch_nums=PATCH,
            quant_resi=0.5, share_quant_resi=4)

    def forward(self, x):
        f_hat, usages, vq_loss = self.quantize(self.quant_conv(self.encoder(x)))
        return self.decoder(self.post_quant_conv(f_hat)), usages, vq_loss

    @torch.no_grad()
    def img_to_idxBl(self, x, v_patch_nums=None):
        f = self.quant_conv(self.encoder(x))
        return self.quantize.f_to_idxBl_or_fhat(f, to_fhat=False,
                                                v_patch_nums=v_patch_nums or PATCH)

    @torch.no_grad()
    def fhat_to_img(self, f_hat):
        return self.decoder(self.post_quant_conv(f_hat)).clamp_(-1, 1)


def mnist_loader(train=True, batch=256, workers=4):
    from torchvision import datasets, transforms
    tfm = transforms.Compose([
        transforms.Pad(2),                       # 28 -> 32
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))])
    ds = datasets.MNIST(root="/home/mengy13/ptp-vqvae/data/mnist", train=train,
                        download=False, transform=tfm)
    from torch.utils.data import DataLoader
    return DataLoader(ds, batch_size=batch, shuffle=train, num_workers=workers,
                      drop_last=train, persistent_workers=workers > 0)


def build_var(depth=8, embed_dim=512, heads=8):
    from models.var import VAR
    vae = MnistVQVAE().to(device)
    var = VAR(vae_local=vae, num_classes=NUM_CLASSES, depth=depth,
              embed_dim=embed_dim, num_heads=heads, drop_rate=0.0,
              attn_drop_rate=0.0, drop_path_rate=0.0, norm_eps=1e-6,
              shared_aln=False, cond_drop_rate=0.1, attn_l2_norm=True,
              patch_nums=PATCH, flash_if_available=True,
              fused_if_available=True).to(device)
    var.init_weights(init_adaln=0.5, init_adaln_gamma=1e-5, init_head=0.02,
                     init_std=-1)
    return vae, var


def stage_vqvae(epochs=12):
    ensure_process_group()
    OUT.mkdir(parents=True, exist_ok=True)
    vae = MnistVQVAE().to(device)
    opt = torch.optim.AdamW(vae.parameters(), lr=3e-4, weight_decay=0.01)
    ld = mnist_loader(True)
    print(f"vqvae params {sum(p.numel() for p in vae.parameters())/1e6:.1f}M, "
          f"scales {PATCH}, {sum(p*p for p in PATCH)} tokens", flush=True)
    for ep in range(epochs):
        vae.train()
        tot = n = 0.0
        for x, _ in ld:
            x = x.to(device, non_blocking=True)
            rec, _, vq = vae(x)
            loss = F.mse_loss(rec, x) + vq
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(vae.parameters(), 1.0)
            opt.step()
            tot += float(loss) * x.shape[0]
            n += x.shape[0]
        vae.eval()
        with torch.no_grad():
            xs, _ = next(iter(mnist_loader(False, 256, 0)))
            rec, _, _ = vae(xs.to(device))
            psnr = 10 * torch.log10(4.0 / F.mse_loss(rec, xs.to(device)))
        print(f"epoch {ep+1}/{epochs} loss {tot/n:.4f} test PSNR {float(psnr):.2f} dB",
              flush=True)
    torch.save(vae.state_dict(), OUT / "vqvae.pt")
    print(f"wrote {OUT/'vqvae.pt'}", flush=True)


def stage_var(epochs=30, depth=8, embed_dim=512):
    ensure_process_group()
    vae, var = build_var(depth=depth, embed_dim=embed_dim)
    vae.load_state_dict(torch.load(OUT / "vqvae.pt", map_location="cpu"))
    vae.eval().requires_grad_(False)
    opt = torch.optim.AdamW(var.parameters(), lr=1e-4, weight_decay=0.05,
                            betas=(0.9, 0.95))
    ld = mnist_loader(True, batch=128)
    test = mnist_loader(False, batch=128, workers=2)
    print(f"var params {sum(p.numel() for p in var.parameters())/1e6:.1f}M, "
          f"L={var.L}", flush=True)
    started = time.time()
    best = float("inf")
    for ep in range(epochs):
        var.train()
        tot = n = 0.0
        for x, y in ld:
            x, y = x.to(device, non_blocking=True), y.to(device)
            with torch.no_grad():
                gt = vae.img_to_idxBl(x)
                xin = vae.quantize.idxBl_to_var_input(gt)
            logits = var(y, xin)
            truth = torch.cat(gt, dim=1)
            loss = F.cross_entropy(logits.reshape(-1, VOCAB), truth.reshape(-1))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(var.parameters(), 1.0)
            opt.step()
            tot += float(loss) * x.shape[0]
            n += x.shape[0]
        var.eval()
        var.cond_drop_rate = 0.0        # no self.training guard inside VAR.forward
        with torch.no_grad():
            v = m = 0.0
            for x, y in test:
                x, y = x.to(device), y.to(device)
                gt = vae.img_to_idxBl(x)
                logits = var(y, vae.quantize.idxBl_to_var_input(gt))
                truth = torch.cat(gt, dim=1)
                v += float(F.cross_entropy(logits.reshape(-1, VOCAB),
                                           truth.reshape(-1))) * x.shape[0]
                m += x.shape[0]
        var.cond_drop_rate = 0.1
        mark = ""
        if v / m < best:
            best = v / m
            torch.save(var.state_dict(), OUT / "var.pt")
            mark = "  <- kept"
        print(f"epoch {ep+1}/{epochs} train {tot/n:.4f} test {v/m:.4f} "
              f"({time.time()-started:.0f}s){mark}", flush=True)
    print(f"wrote {OUT/'var.pt'} at test {best:.4f}", flush=True)


def stage_sample(per_class=10, top_k=100, seed=0):
    """One row per digit, at several guidance scales.

    Guidance is the knob that trades fidelity for variety here as everywhere:
    low cfg gives messier digits with more handwriting variation, high cfg gives
    cleaner ones that look more alike.
    """
    ensure_process_group()
    vae, var = build_var()
    vae.load_state_dict(torch.load(OUT / "vqvae.pt", map_location="cpu"))
    var.load_state_dict(torch.load(OUT / "var.pt", map_location="cpu"))
    vae.eval()
    var.eval()
    var.cond_drop_rate = 0.0
    from torchvision.utils import save_image
    labels = torch.arange(NUM_CLASSES, device=device).repeat_interleave(per_class)
    for cfg in (float(x) for x in os.environ.get("CFGS", "1.0,1.5,2.5,4.0").split(",")):
        torch.manual_seed(seed)
        with torch.no_grad():
            img = var.autoregressive_infer_cfg(
                B=labels.shape[0], label_B=labels, cfg=cfg, top_k=top_k,
                top_p=0.95, more_smooth=False)
        path = OUT / f"samples_cfg{cfg:g}.png"
        save_image(img, path, nrow=per_class)
        print(f"cfg {cfg:g} -> {path}", flush=True)
    # A reconstruction strip, to separate what the tokeniser costs from what the
    # transformer costs: anything the autoencoder cannot represent is a ceiling
    # the sampler could never beat.
    xs, ys = next(iter(mnist_loader(False, per_class * 2, 0)))
    with torch.no_grad():
        rec, _, _ = vae(xs.to(device))
    save_image(torch.cat([xs.to(device), rec]).clamp(-1, 1),
               OUT / "reconstruction.png", nrow=per_class * 2, normalize=True,
               value_range=(-1, 1))
    print(f"wrote {OUT/'reconstruction.png'} (top row real, bottom reconstructed)",
          flush=True)


def loo_mask(n, device):
    """(2n, 2n) bool, True = forbidden. Queries 0..n-1, token keys n..2n-1.

    Queries may not read each other: after one layer a query has already taken in
    every token but its own, so query i reading query j would recover t_i.
    """
    m = torch.ones(2 * n, 2 * n, dtype=torch.bool, device=device)
    idx = torch.arange(n, device=device)
    m[idx, idx] = False
    m[idx.unsqueeze(1), (idx + n).unsqueeze(0)] = False
    m[idx, idx + n] = True
    m[idx + n, idx + n] = False
    return m


class LooHead(nn.Module):
    """mode 'none' sees no tokens of the scale; 'loo' sees every one but its own."""

    def __init__(self, mode, c_var, c=384, layers=2, heads=6, max_len=64):
        super().__init__()
        self.mode = mode
        self.proj = nn.Linear(c_var, c)
        self.tok = nn.Embedding(VOCAB + 1, c)
        self.pos = nn.Parameter(torch.zeros(1, max_len, c))
        self.kind = nn.Parameter(torch.zeros(2, c))
        layer = nn.TransformerEncoderLayer(
            d_model=c, nhead=heads, dim_feedforward=4 * c, dropout=0.0,
            batch_first=True, norm_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(layer, num_layers=layers)
        self.norm = nn.LayerNorm(c)
        self.out = nn.Linear(c, VOCAB)

    def forward(self, h_seg, tok_seg):
        b, n, _ = h_seg.shape
        q = self.proj(h_seg) + self.pos[:, :n] + self.kind[0]
        k = self.tok(tok_seg if self.mode == "loo"
                     else torch.full_like(tok_seg, VOCAB))
        k = k + self.pos[:, :n] + self.kind[1]
        x = torch.cat([q, k], dim=1)
        x = self.enc(x, mask=loo_mask(n, x.device))
        return self.out(self.norm(x[:, :n]))


def stage_loo(steps=4000, eval_every=1000, unfreeze=2, c_var=512):
    """Does a scale of this VAR hold information about itself?

    Same design as the ImageNet run: two heads of identical size and shape, each
    with its own trainable copy of the top blocks so their gradients never mix in
    the layers under test. One is shown every other token of the scale, the other
    a constant. The gap between them is the within-scale information.
    """
    import copy
    ensure_process_group()
    vae, var = build_var()
    vae.load_state_dict(torch.load(OUT / "vqvae.pt", map_location="cpu"))
    var.load_state_dict(torch.load(OUT / "var.pt", map_location="cpu"))
    vae.eval().requires_grad_(False)
    var.eval().requires_grad_(False)
    var.cond_drop_rate = 0.0

    cut = len(var.blocks) - unfreeze
    cap = {}

    def pre_hook(mod, args, kwargs):
        cap["x"], cap["cond_BD"], cap["attn_bias"] = (
            kwargs["x"], kwargs["cond_BD"], kwargs["attn_bias"])
    var.blocks[cut].register_forward_pre_hook(pre_hook, with_kwargs=True)

    bounds, cur = [], 0
    for pn in PATCH:
        bounds.append((cur, cur + pn * pn, pn))
        cur += pn * pn

    class Arm(nn.Module):
        def __init__(self, mode):
            super().__init__()
            self.blocks = copy.deepcopy(var.blocks[cut:]).requires_grad_(True)
            self.head = LooHead(mode, c_var)

        def forward(self, truth):
            x = cap["x"]
            for b in self.blocks:
                x = b(x=x, cond_BD=cap["cond_BD"], attn_bias=cap["attn_bias"])
            h = x.float()
            return [self.head(h[:, a:b], truth[:, a:b]) for a, b, _ in bounds]

    arms = {"A_no_context": Arm("none").to(device),
            "C_leave_one_out": Arm("loo").to(device)}
    for name, m in arms.items():
        bad = [k for k, q in m.named_parameters() if not torch.isfinite(q).all()]
        assert not bad, f"{name} non-finite: {bad[:3]}"
        print(f"{name}: {sum(p.numel() for p in m.parameters())/1e6:.1f}M", flush=True)

    with torch.no_grad():
        probe = arms["C_leave_one_out"].head
        hh = torch.randn(2, 8, c_var, device=device)
        t1 = torch.randint(0, VOCAB, (2, 8), device=device)
        t2 = t1.clone()
        t2[:, 3] = (t2[:, 3] + 1) % VOCAB
        o1, o2 = probe(hh, t1), probe(hh, t2)
        own = float((o1[:, 3] - o2[:, 3]).abs().max())
        other = float((o1[:, 5] - o2[:, 5]).abs().max())
    print(f"leakage check: token 3 moves slot 3 by {own:.2e}, slot 5 by {other:.2e}",
          flush=True)
    assert own < 1e-5 and other > 1e-4, "leave-one-out mask is wrong"

    opt = torch.optim.AdamW([
        {"params": [p for m in arms.values() for p in m.blocks.parameters()],
         "lr": 2e-5},
        {"params": [p for m in arms.values() for p in m.head.parameters()],
         "lr": 3e-4}], weight_decay=0.01)

    train_ld = mnist_loader(True, batch=64)
    test_ld = mnist_loader(False, batch=64, workers=2)

    def var_pass(x, y):
        with torch.no_grad():
            gt = vae.img_to_idxBl(x)
            var(y, vae.quantize.idxBl_to_var_input(gt))
        return torch.cat(gt, dim=1)

    def arm_losses(arm, truth, reduction="mean"):
        return [F.cross_entropy(o.reshape(-1, VOCAB),
                                truth[:, a:b].reshape(-1), reduction=reduction)
                for o, (a, b, _) in zip(arm(truth), bounds)]

    @torch.no_grad()
    def evaluate():
        acc = {k: torch.zeros(len(bounds)) for k in list(arms) + ["VAR"]}
        n = 0
        for x, y in test_ld:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                gt = vae.img_to_idxBl(x)
                logits = var(y, vae.quantize.idxBl_to_var_input(gt)).float()
            truth = torch.cat(gt, dim=1)
            for s, (a, b, _) in enumerate(bounds):
                acc["VAR"][s] += float(F.cross_entropy(
                    logits[:, a:b].reshape(-1, VOCAB), truth[:, a:b].reshape(-1),
                    reduction="sum"))
            for k, m in arms.items():
                for s, l in enumerate(arm_losses(m, truth, reduction="sum")):
                    acc[k][s] += float(l)
            n += x.shape[0]
        return {k: v / n for k, v in acc.items()}

    def report(r, tag):
        A, C, V = r["A_no_context"], r["C_leave_one_out"], r["VAR"]
        print(f"\n--- {tag} ---")
        print("scale  tokens   VAR    head A   head C   A-C (nats/tok)  gain/image")
        for s, (a, b, pn) in enumerate(bounds):
            n = b - a
            print(f"{pn:>3}x{pn:<3} {n:>5}  {V[s]/n:6.3f}  {A[s]/n:7.3f}  "
                  f"{C[s]/n:7.3f}  {(A[s]-C[s])/n:13.4f}  {A[s]-C[s]:10.2f}")
        ta, tc = float(A.sum()), float(C.sum())
        print(f"per image: VAR {float(V.sum()):.2f}  A {ta:.2f}  C {tc:.2f}  "
              f"within-scale information {ta-tc:.2f} nats "
              f"({100*(ta-tc)/ta:.2f}% of A)", flush=True)

    report(evaluate(), "before training")
    step, started = 0, time.time()
    while step < steps:
        for x, y in train_ld:
            if step >= steps:
                break
            x, y = x.to(device, non_blocking=True), y.to(device)
            truth = var_pass(x, y)
            loss = sum(sum(arm_losses(m, truth)) for m in arms.values())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for m in arms.values() for p in m.parameters()], 1.0)
            opt.step()
            step += 1
            if step % 200 == 0:
                print(f"step {step}/{steps} loss {float(loss):.3f} "
                      f"({step/(time.time()-started):.2f} it/s)", flush=True)
            if step % eval_every == 0 or step == steps:
                report(evaluate(), f"step {step}")
    print("MNIST_LOO_DONE", flush=True)


if __name__ == "__main__":
    stage = os.environ.get("STAGE", "vqvae")
    if stage == "vqvae":
        stage_vqvae(epochs=int(os.environ.get("EPOCHS", 12)))
    elif stage == "var":
        stage_var(epochs=int(os.environ.get("EPOCHS", 30)))
    elif stage == "sample":
        stage_sample()
    elif stage == "loo":
        stage_loo(steps=int(os.environ.get("STEPS", 4000)))
    else:
        raise SystemExit(f"unknown STAGE {stage}")
