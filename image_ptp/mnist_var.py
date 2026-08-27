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
        print(f"epoch {ep+1}/{epochs} train {tot/n:.4f} test {v/m:.4f} "
              f"({time.time()-started:.0f}s)", flush=True)
        torch.save(var.state_dict(), OUT / "var.pt")
    print(f"wrote {OUT/'var.pt'}", flush=True)


def stage_sample(n=16, cfg=2.0, top_k=100):
    ensure_process_group()
    vae, var = build_var()
    vae.load_state_dict(torch.load(OUT / "vqvae.pt", map_location="cpu"))
    var.load_state_dict(torch.load(OUT / "var.pt", map_location="cpu"))
    vae.eval(); var.eval()
    var.cond_drop_rate = 0.0
    labels = torch.arange(n, device=device) % NUM_CLASSES
    with torch.no_grad():
        img = var.autoregressive_infer_cfg(B=n, label_B=labels, cfg=cfg,
                                           top_k=top_k, top_p=0.95,
                                           more_smooth=False)
    from torchvision.utils import save_image
    path = OUT / "samples.png"
    save_image(img, path, nrow=8)
    print(f"labels {labels.tolist()}")
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    stage = os.environ.get("STAGE", "vqvae")
    if stage == "vqvae":
        stage_vqvae(epochs=int(os.environ.get("EPOCHS", 12)))
    elif stage == "var":
        stage_var(epochs=int(os.environ.get("EPOCHS", 30)))
    elif stage == "sample":
        stage_sample()
    else:
        raise SystemExit(f"unknown STAGE {stage}")
