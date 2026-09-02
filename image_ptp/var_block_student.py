"""A student that emits VAR's first K scales in one forward, driven by u.

Input at every position is the same: an encoding of that position's own u, plus
its level and position embeddings, plus the class. The mask is scale-causal --
position (j, i) attends to every position of scales before j and to itself, but
not to its own scale, because VAR's scales are conditionally independent inside
themselves and reading a neighbour's u there buys nothing. That makes the
inference four steps deep rather than thirty.

Two heads, sharing everything up to the last layer:

  ce   4096-way logits, cross-entropy against the true token -- PTP's own
       contract, and it demands an exact code match
  mse  32-dim regression onto the token's codebook embedding -- the downstream
       scales only ever see f_hat, so a near-miss code costs almost nothing
       there, and this head is scored on that rather than on identity

Both are deterministic given u, and u carries the randomness in both, so neither
collapses to a mean: the interval construction makes (class, u) determine the
tokens outright.
"""
import argparse
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def scale_causal_mask(patch_nums, scales, device):
    """(L, L) bool: position may attend to strictly earlier scales, plus itself."""
    lvl = torch.cat([torch.full((p * p,), si, device=device)
                     for si, p in enumerate(patch_nums[:scales])])
    L = lvl.shape[0]
    allow = lvl[None, :] < lvl[:, None]
    allow |= torch.eye(L, dtype=torch.bool, device=device)
    return allow


class FourierU(nn.Module):
    """Encode a scalar u with a spread of frequencies, then project.

    Interval widths in this project span four orders of magnitude, so a single
    linear layer on the raw scalar cannot resolve the narrow ones; a frequency
    ladder gives the network a basis fine enough to separate them.
    """

    def __init__(self, dim, n_freq=64, max_freq=512.0):
        super().__init__()
        freqs = torch.exp(torch.linspace(0, math.log(max_freq), n_freq))
        self.register_buffer("freqs", freqs)
        self.proj = nn.Linear(2 * n_freq + 1, dim)

    def forward(self, u):                       # (B, L) -> (B, L, dim)
        a = u.unsqueeze(-1) * self.freqs
        return self.proj(torch.cat([u.unsqueeze(-1), a.sin(), a.cos()], -1))


class Block(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(),
                                 nn.Linear(4 * dim, dim))

    def forward(self, x, mask):
        h = self.n1(x)
        x = x + self.attn(h, h, h, attn_mask=~mask, need_weights=False)[0]
        return x + self.mlp(self.n2(x))


class BlockStudent(nn.Module):
    def __init__(self, patch_nums, scales, vocab=4096, cvae=32, n_class=1000,
                 dim=512, depth=8, heads=8):
        super().__init__()
        self.patch_nums, self.scales = patch_nums, scales
        L = sum(p * p for p in patch_nums[:scales])
        self.L = L
        self.u_embed = FourierU(dim)
        self.lvl = nn.Embedding(scales, dim)
        self.pos = nn.Embedding(L, dim)
        self.cls = nn.Embedding(n_class + 1, dim)
        lvl_id = torch.cat([torch.full((p * p,), si)
                            for si, p in enumerate(patch_nums[:scales])])
        self.register_buffer("lvl_id", lvl_id)
        self.blocks = nn.ModuleList(Block(dim, heads) for _ in range(depth))
        self.norm = nn.LayerNorm(dim)
        self.head_ce = nn.Linear(dim, vocab)
        self.head_mse = nn.Linear(dim, cvae)

    def forward(self, u, label, mask):
        B = u.shape[0]
        x = (self.u_embed(u) + self.lvl(self.lvl_id)[None]
             + self.pos(torch.arange(self.L, device=u.device))[None]
             + self.cls(label)[:, None])
        for b in self.blocks:
            x = b(x, mask)
        x = self.norm(x)
        return self.head_ce(x), self.head_mse(x)


def draw_u(left, right, gaussian):
    """Uniform inside the token's interval; the marginal is then U(0,1), which is
    what inference draws from. Gaussian coding is Phi^-1 of that -- a monotone
    reparameterisation, so which interval a draw falls in is unchanged."""
    u = left + (right - left) * torch.rand_like(left)
    if gaussian:
        u = torch.erfinv(2 * u.clamp(1e-6, 1 - 1e-6) - 1) * math.sqrt(2)
    return u


def evaluate(model, data, mask, device, gaussian, n=4096):
    """Exact-code agreement, and how far the emitted embedding lands.

    `agree` is PTP's own criterion and demands the identical code. `embed err`
    is the relative distance between the codebook vector the student implies and
    the true one, which is what the six frozen scales downstream actually
    consume -- a near-miss code costs them almost nothing.

    Both are reported against a shuffled u, because a student that ignores its
    auxiliaries entirely still scores well on a peaked marginal, and only the
    ratio separates reading u from predicting the mode.
    """
    model.eval()
    tok = data["tokens"][:n].long().to(device)
    left, right = data["left"][:n].to(device), data["right"][:n].to(device)
    lab = data["labels"][:n].to(device)
    emb = data["codebook"].to(device)
    out = {}
    with torch.no_grad():
        for name, shuffle in (("real", False), ("shuf", True)):
            u = draw_u(left, right, gaussian)
            if shuffle:
                u = u[torch.randperm(u.shape[0], device=device)]
            ce, ms = model(u, lab, mask)
            pred = ce.argmax(-1)
            out[f"agree_{name}"] = float((pred == tok).float().mean())
            true_emb = emb[tok]
            out[f"emberr_{name}"] = float(
                (ms - true_emb).norm(dim=-1).mean() / true_emb.norm(dim=-1).mean())
            # the CE head's own choice, measured the same way
            out[f"cehead_emberr_{name}"] = float(
                (emb[pred] - true_emb).norm(dim=-1).mean() / true_emb.norm(dim=-1).mean())
    model.train()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--vae", default="/home/mengy13/VAR/checkpoints/vae_ch160v4096z32.pth")
    ap.add_argument("--gaussian", action="store_true")
    ap.add_argument("--loss", default="both", choices=["ce", "mse", "both"])
    ap.add_argument("--mse-weight", type=float, default=1.0)
    ap.add_argument("--dim", type=int, default=768)
    ap.add_argument("--depth", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    # so the loop can be smoke-tested off the cluster
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    device = args.device

    tr = torch.load(args.train, map_location="cpu")
    va = torch.load(args.val, map_location="cpu")
    pn, scales = tr["patch_nums"], tr["scales"]
    # The MSE head regresses onto codebook vectors, so it needs the codebook.
    sd = torch.load(args.vae, map_location="cpu")
    codebook = sd["quantize.embedding.weight"].float()
    tr["codebook"] = va["codebook"] = codebook
    mask = scale_causal_mask(pn, scales, device)

    model = BlockStudent(pn, scales, dim=args.dim, depth=args.depth).to(device)
    print(f"student {sum(p.numel() for p in model.parameters())/1e6:.1f}M, "
          f"L={model.L}, u={'gaussian' if args.gaussian else 'uniform'}, "
          f"loss={args.loss}", flush=True)
    w = (tr["right"] - tr["left"])
    print(f"train intervals: median {float(w.median()):.5f}  "
          f"mean {float(w.mean()):.5f}  frac<1e-3 {float((w < 1e-3).float().mean()):.3f}",
          flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05,
                            betas=(0.9, 0.95))
    N = tr["tokens"].shape[0]
    steps = args.epochs * (N // args.batch_size)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=steps,
                                                pct_start=0.03)
    emb_all = codebook.to(device)
    best, step = float("inf"), 0
    for ep in range(args.epochs):
        perm = torch.randperm(N)
        for i in range(0, N - args.batch_size + 1, args.batch_size):
            j = perm[i:i + args.batch_size]
            tok = tr["tokens"][j].long().to(device, non_blocking=True)
            u = draw_u(tr["left"][j].to(device), tr["right"][j].to(device),
                       args.gaussian)
            lab = tr["labels"][j].to(device)
            ce, ms = model(u, lab, mask)
            loss = 0.0
            if args.loss in ("ce", "both"):
                loss = loss + F.cross_entropy(ce.reshape(-1, ce.shape[-1]),
                                              tok.reshape(-1))
            if args.loss in ("mse", "both"):
                loss = loss + args.mse_weight * F.mse_loss(ms, emb_all[tok])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); step += 1
        m = evaluate(model, va, mask, device, args.gaussian)
        mark = ""
        if m["emberr_real"] < best:
            best = m["emberr_real"]
            torch.save(dict(model=model.state_dict(), args=vars(args)), args.out)
            mark = "  <- kept"
        print(f"epoch {ep+1}/{args.epochs}  loss {float(loss):.4f}  "
              f"agree {m['agree_real']:.4f} (shuf {m['agree_shuf']:.4f}, "
              f"lift {m['agree_real']/max(m['agree_shuf'],1e-9):.2f}x)  "
              f"embErr {m['emberr_real']:.4f} (shuf {m['emberr_shuf']:.4f})"
              f"{mark}", flush=True)
    print(f"\nbest relative embedding error {best:.4f}, written to {args.out}")
    print("BLOCK_STUDENT_DONE")


if __name__ == "__main__":
    main()
