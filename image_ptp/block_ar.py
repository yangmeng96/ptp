"""A full autoregressive model over VAR's first K scales, as the PTP ceiling.

The PTP student emits those 30 tokens in one forward from u, and reaches FID
10.71 against the official 7.77. Two things cap it: the single forward, in which
a position learns of earlier scales only through u, and u's finite precision. An
autoregressive model over the same 30 tokens -- 30 steps, each seeing the tokens
actually sampled before it -- has neither limit and models the block's full
joint exactly. It trains on the identical data the student does, so the gap from
the student up to this AR is the price of the compression, and the gap from this
AR down to the official number is everything else.
"""
import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, "/home/mengy13/VAR")

from image_ptp.var_block_student import Block


class BlockAR(nn.Module):
    """Raster AR over the first K scales: position i sees tokens 0..i-1."""

    def __init__(self, patch_nums, scales, vocab=4096, n_class=1000,
                 dim=512, depth=8, heads=8, drop=0.1):
        super().__init__()
        L = sum(p * p for p in patch_nums[:scales])
        self.L, self.vocab = L, vocab
        self.tok = nn.Embedding(vocab + 1, dim)          # +1 = block BOS
        self.lvl = nn.Embedding(scales, dim)
        self.pos = nn.Embedding(L, dim)
        self.cls = nn.Embedding(n_class + 1, dim)
        lvl_id = torch.cat([torch.full((p * p,), si)
                            for si, p in enumerate(patch_nums[:scales])])
        self.register_buffer("lvl_id", lvl_id)
        self.register_buffer("mask", torch.tril(torch.ones(L, L, dtype=torch.bool)))
        self.blocks = nn.ModuleList(Block(dim, heads, drop) for _ in range(depth))
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab)

    def forward(self, fed, label):
        # fed[:, i] is the token that precedes position i (BOS at 0)
        site = (self.lvl(self.lvl_id)[None]
                + self.pos(torch.arange(self.L, device=fed.device))[None]
                + self.cls(label)[:, None])
        x = self.tok(fed) + site
        for b in self.blocks:
            x = b(x, self.mask)
        return self.head(self.norm(x))


def evaluate(model, data, device):
    model.eval()
    tok = data["tokens"][:4096].long().to(device)
    lab = data["labels"][:4096].to(device)
    bos = model.vocab
    fed = torch.cat([torch.full((tok.shape[0], 1), bos, device=device),
                     tok[:, :-1]], 1)
    with torch.no_grad():
        logits = model(fed, lab)
        ce = F.cross_entropy(logits.reshape(-1, model.vocab), tok.reshape(-1))
        acc = (logits.argmax(-1) == tok).float().mean()
    model.train()
    return float(ce), float(acc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--vae", default="/home/mengy13/VAR/checkpoints/vae_ch160v4096z32.pth")
    ap.add_argument("--scales", type=int, default=4)
    ap.add_argument("--dim", type=int, default=512)
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    device = args.device

    tr = torch.load(args.train, map_location="cpu")
    va = torch.load(args.val, map_location="cpu")
    pn, scales = tr["patch_nums"], tr["scales"]
    vocab = int(tr["tokens"].max()) + 1
    model = BlockAR(pn, scales, vocab=vocab, n_class=int(tr["labels"].max()) + 1,
                    dim=args.dim, depth=args.depth).to(device)
    print(f"block AR {sum(p.numel() for p in model.parameters())/1e6:.1f}M, "
          f"L={model.L}, vocab={vocab}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05,
                            betas=(0.9, 0.95))
    N = tr["tokens"].shape[0]
    steps = args.epochs * (N // args.batch_size)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=steps,
                                                pct_start=0.03)
    bos = vocab
    best = -1.0
    for ep in range(args.epochs):
        perm = torch.randperm(N)
        for i in range(0, N - args.batch_size + 1, args.batch_size):
            j = perm[i:i + args.batch_size]
            tok = tr["tokens"][j].long().to(device)
            lab = tr["labels"][j].to(device)
            fed = torch.cat([torch.full((tok.shape[0], 1), bos, device=device),
                             tok[:, :-1]], 1)
            loss = F.cross_entropy(model(fed, lab).reshape(-1, vocab),
                                   tok.reshape(-1))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
        ce, acc = evaluate(model, va, device)
        mark = ""
        if acc > best:
            best = acc
            torch.save(dict(model=model.state_dict(), args=vars(args),
                            vocab=vocab, scales=scales), args.out)
            mark = "  <- kept"
        print(f"epoch {ep+1}/{args.epochs}  loss {float(loss):.4f}  "
              f"val CE {ce:.4f}  teacher-forced acc {acc:.4f}{mark}", flush=True)
    print(f"\nbest teacher-forced acc {best:.4f}, written to {args.out}")
    print("BLOCK_AR_DONE")


if __name__ == "__main__":
    main()
