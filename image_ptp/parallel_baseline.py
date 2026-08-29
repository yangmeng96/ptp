"""A generator with no autoregression at all: label in, every token out at once.

The measurements on MNIST keep coming back near zero, and one reading is that
MNIST is simply easy enough that the structure being measured does not matter --
that a plain network emitting all the tokens in parallel, conditioned on nothing
but the class, would produce digits just as good. That is worth settling with a
picture rather than an argument.

This is the extreme case of the ladder: no prefix scale, no raster prefix, no u.
Every token is drawn independently from p(t_i | class). It shares the tokeniser
with the whole MNIST PTP line, so its samples are directly comparable to the AR
teacher's.
"""
import argparse
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", type=str, default="/home/mengy13/ptp-vqvae")
    p.add_argument("--ar-ckpt", type=str,
                   default="/home/mengy13/ptp-vqvae/checkpoints/ar_mnist_raster.pt")
    p.add_argument("--data", type=str,
                   default="/home/mengy13/ptp-image-results/mnist_tokens.pt")
    p.add_argument("--depth", type=int, default=8)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--steps", type=int, default=8000)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--out-dir", type=str,
                   default="/home/mengy13/ptp-image-results/parallel_mnist")
    return p.parse_args()


class ParallelGenerator(nn.Module):
    """p(t_i | class) for every position at once, with full bidirectional
    attention between positions -- they may look at each other's *inputs*, which
    carry no token information, exactly as VAR's positions do within a scale."""

    def __init__(self, vocab, seq_len, num_classes=10, width=512, depth=8, heads=8):
        super().__init__()
        self.cls = nn.Embedding(num_classes, width)
        self.pos = nn.Parameter(torch.zeros(1, seq_len, width))
        nn.init.trunc_normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=width, nhead=heads, dim_feedforward=4 * width, dropout=0.0,
            batch_first=True, norm_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, vocab)

    def forward(self, labels):
        x = self.cls(labels)[:, None, :] + self.pos
        return self.head(self.norm(self.enc(x)))


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    payload = torch.load(Path(args.data).expanduser(), map_location="cpu")
    tokens = payload["tokens"].long()
    labels = payload["labels"].long()
    # prepare_mnist_tokens.py zeroes the labels, so recover them from the dataset.
    if int(labels.abs().sum()) == 0:
        from torchvision import datasets
        ds = datasets.MNIST(root=f"{args.repo}/data/mnist", train=True,
                            download=False)
        labels = ds.targets[:tokens.shape[0]].long()
    seq_len = tokens.shape[1]
    val = 256
    print(f"{tokens.shape[0]} sequences of {seq_len} tokens, "
          f"{len(labels.unique())} classes", flush=True)

    sys.path.insert(0, "/home/mengy13/ptp")
    from image_ptp.vqvae_ar_hf import build
    teacher, meta = build(args.ar_ckpt, device=device, dtype=torch.float32)
    teacher.eval().requires_grad_(False)
    V, bos = meta["num_codes"], meta["num_codes"]

    model = ParallelGenerator(V, seq_len, width=args.width, depth=args.depth,
                              heads=args.heads).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    print(f"parallel generator {sum(p.numel() for p in model.parameters())/1e6:.1f}M",
          flush=True)

    tr_t, tr_y = tokens[val:].to(device), labels[val:].to(device)
    va_t, va_y = tokens[:val].to(device), labels[:val].to(device)

    def val_loss():
        model.eval()
        with torch.no_grad():
            lg = model(va_y)
            l = F.cross_entropy(lg.reshape(-1, V), va_t.reshape(-1))
        model.train()
        return float(l)

    # What the same tokens cost an autoregressive model: the floor this cannot reach.
    with torch.no_grad():
        ids = torch.cat([torch.full((val, 1), bos, device=device), va_t], 1)
        # The AR model's vocabulary carries BOS on top of the codes, so its
        # logits are one wider than the generator's.
        ar_logits = teacher(input_ids=ids).logits[:, :-1]
        ar = float(F.cross_entropy(
            ar_logits.reshape(-1, ar_logits.shape[-1]), va_t.reshape(-1)))
    print(f"autoregressive teacher on the same held-out tokens: {ar:.4f} nats/token",
          flush=True)

    best = float("inf")
    started = time.time()
    for step in range(1, args.steps + 1):
        i = torch.randint(0, tr_t.shape[0], (args.batch_size,), device=device)
        lg = model(tr_y[i])
        loss = F.cross_entropy(lg.reshape(-1, V), tr_t[i].reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 500 == 0 or step == args.steps:
            v = val_loss()
            mark = ""
            if v < best:
                best, mark = v, "  <- kept"
                torch.save(model.state_dict(), out / "parallel.pt")
            print(f"step {step}/{args.steps} train {float(loss):.4f} val {v:.4f} "
                  f"({step/(time.time()-started):.1f} it/s){mark}", flush=True)

    print(f"\nbest parallel val {best:.4f} vs autoregressive {ar:.4f} nats/token")
    print(f"the gap, {best - ar:.4f} nats/token, is what independence costs here",
          flush=True)

    model.load_state_dict(torch.load(out / "parallel.pt", map_location="cpu"))
    model.eval()
    os.chdir(args.repo)
    sys.path.insert(0, args.repo)
    from models.ar import seq_to_codes_grid
    from utils.helper import load_vqvae
    from torchvision.utils import save_image
    vqvae, _ = load_vqvae("mnist", device)
    inv = torch.argsort(meta["perm"]).to(device)
    h, w = meta["h"], meta["w"]

    torch.manual_seed(0)
    lab = torch.arange(10, device=device).repeat_interleave(10)
    with torch.no_grad():
        probs = torch.softmax(model(lab).float(), -1)
        seq = torch.multinomial(probs.reshape(-1, V), 1).view(lab.shape[0], seq_len)
        img = vqvae.decode(seq_to_codes_grid(seq, inv, h, w)).float()
    save_image(img, out / "samples_parallel.png", nrow=10, normalize=True,
               value_range=(-1, 1))

    with torch.no_grad():
        s = torch.full((lab.shape[0], 1), bos, dtype=torch.long, device=device)
        for _ in range(seq_len):
            p = torch.softmax(teacher(input_ids=s).logits[:, -1].float(), -1)
            p[:, bos] = 0
            s = torch.cat([s, torch.multinomial(p, 1)], 1)
        img = vqvae.decode(seq_to_codes_grid(s[:, 1:], inv, h, w)).float()
    save_image(img, out / "samples_ar.png", nrow=10, normalize=True,
               value_range=(-1, 1))
    print(f"wrote {out}/samples_parallel.png and samples_ar.png", flush=True)


if __name__ == "__main__":
    main()
