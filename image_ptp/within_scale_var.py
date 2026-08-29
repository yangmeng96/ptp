"""A VAR whose scales are autoregressive inside themselves.

VAR draws every token of a scale independently from p(t | previous scales). On
the standard ladder that costs almost nothing -- the residual quantiser has
already removed what the previous scales could predict, and a head shown every
other token of a scale gains 0.04% on ImageNet, 0.31% on MNIST. But the moment
the ladder is coarsened the assumption starts to bite: (1,8) on MNIST measures
3.42% and its samples are visibly rougher than (1,2,4,8)'s.

That is the opening. A coarser ladder is faster -- at batch 1 every scale costs
one forward regardless of its size, so halving the scales halves the latency --
and what it gives up is exactly the dependence PTP's auxiliaries are built to
carry. To distil that back, a teacher that models it has to exist first, and
VAR's does not: its inputs at scale k are the previous scales interpolated up,
so no position can see what its neighbours in the same scale produced. Making
the mask causal changes nothing, because there is nothing there to reveal.

So the teacher needs a second input stream. Position i of scale k additionally
receives the embedding of t_{k,i-1}, the raster predecessor inside its own scale,
with a learned marker at each scale's first position. The mask becomes causal
within a scale and stays full across scales. Everything else -- the quantiser,
the blocks, the conditioning -- is VAR's.

Sampling from this is slow by construction: a scale of n tokens costs n forwards
instead of one. That is fine, and it is the point. It only ever runs offline, to
supply the distribution a PTP student is trained to invert in a single call.
"""
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

VAR_ROOT = Path(os.environ.get("VAR_ROOT", "/home/mengy13/VAR"))
sys.path.insert(0, str(VAR_ROOT))


def build_within_scale_var(vae, patch_nums, num_classes=10, depth=8,
                           embed_dim=512, heads=8, device="cuda"):
    from models.var import VAR

    class WithinScaleVAR(VAR):
        def __init__(self, **kw):
            super().__init__(**kw)
            V = self.V
            # One extra id marks "first position of this scale", which has no
            # predecessor inside it.
            self.scale_bos = V
            self.wtok = nn.Embedding(V + 1, self.C)
            nn.init.trunc_normal_(self.wtok.weight.data, mean=0, std=0.02)

            # Causal within a scale, unrestricted across scales.
            d = torch.cat([torch.full((pn * pn,), i)
                           for i, pn in enumerate(self.patch_nums)])
            idx = torch.arange(self.L)
            same = d[:, None] == d[None, :]
            allowed = (d[:, None] > d[None, :]) | (same & (idx[None, :] <= idx[:, None]))
            bias = torch.where(allowed, 0.0, -torch.inf).reshape(1, 1, self.L, self.L)
            self.register_buffer("within_scale_bias", bias.contiguous())

            starts = []
            cur = 0
            for pn in self.patch_nums:
                starts.append(cur)
                cur += pn * pn
            self.register_buffer("scale_start",
                                 torch.tensor(starts, dtype=torch.long),
                                 persistent=False)

        def predecessor_ids(self, truth):
            """(B, L) -> the id each position should receive as its within-scale
            predecessor: the token before it in raster order, or scale_bos at the
            first position of each scale."""
            prev = torch.full_like(truth, self.scale_bos)
            prev[:, 1:] = truth[:, :-1]
            prev[:, self.scale_start] = self.scale_bos
            return prev

        def forward(self, label_B, x_BLCv_wo_first_l, truth=None):
            assert truth is not None, "the within-scale stream needs the tokens"
            B = truth.shape[0]
            with torch.cuda.amp.autocast(enabled=False):
                label_B = torch.where(
                    torch.rand(B, device=label_B.device) < self.cond_drop_rate,
                    self.num_classes, label_B)
                sos = cond_BD = self.class_emb(label_B)
                sos = (sos.unsqueeze(1).expand(B, self.first_l, -1)
                       + self.pos_start.expand(B, self.first_l, -1))
                if x_BLCv_wo_first_l is None or x_BLCv_wo_first_l.shape[1] == 0:
                    x_BLC = sos
                else:
                    x_BLC = torch.cat(
                        (sos, self.word_embed(x_BLCv_wo_first_l.float())), dim=1)
                x_BLC = x_BLC + (self.lvl_embed(self.lvl_1L.expand(B, -1))
                                 + self.pos_1LC)
                x_BLC = x_BLC + self.wtok(self.predecessor_ids(truth))

            attn_bias = self.within_scale_bias
            cond_BD_or_gss = self.shared_ada_lin(cond_BD)
            temp = x_BLC.new_ones(8, 8)
            main_type = torch.matmul(temp, temp).dtype
            x_BLC = x_BLC.to(dtype=main_type)
            cond_BD_or_gss = cond_BD_or_gss.to(dtype=main_type)
            attn_bias = attn_bias.to(dtype=main_type)
            for b in self.blocks:
                x_BLC = b(x=x_BLC, cond_BD=cond_BD_or_gss, attn_bias=attn_bias)
            return self.get_logits(x_BLC.float(), cond_BD)

    var = WithinScaleVAR(
        vae_local=vae, num_classes=num_classes, depth=depth, embed_dim=embed_dim,
        num_heads=heads, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0,
        norm_eps=1e-6, shared_aln=False, cond_drop_rate=0.1, attn_l2_norm=True,
        patch_nums=patch_nums, flash_if_available=True, fused_if_available=True,
    ).to(device)
    var.init_weights(init_adaln=0.5, init_adaln_gamma=1e-5, init_head=0.02,
                     init_std=-1)
    return var


def var_input(vae, gt, device):
    """idxBl_to_var_input returns None for a single-scale ladder; the forward
    wants an empty tensor there rather than a special case."""
    x = vae.quantize.idxBl_to_var_input(gt)
    if x is None:
        return torch.zeros(gt[0].shape[0], 0, vae.Cvae, device=device)
    return x


def main():
    import argparse
    import time
    p = argparse.ArgumentParser()
    p.add_argument("--patch", type=str, default="1,8")
    p.add_argument("--vqvae", type=str,
                   default="/home/mengy13/ptp-image-results/mnist_var_r8/vqvae.pt")
    p.add_argument("--out", type=str,
                   default="/home/mengy13/ptp-image-results/mnist_var_r8")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-4)
    args = p.parse_args()

    os.environ["PATCH"] = args.patch
    os.environ["OUT"] = args.out
    sys.path.insert(0, "/home/mengy13/ptp")
    from image_ptp import mnist_var as mv

    device = "cuda"
    mv.ensure_process_group()
    vae = mv.MnistVQVAE().to(device)
    vae.load_state_dict(torch.load(args.vqvae, map_location="cpu"))
    vae.eval().requires_grad_(False)
    var = build_within_scale_var(vae, mv.PATCH, num_classes=mv.NUM_CLASSES,
                                 device=device)
    print(f"within-scale AR VAR: {sum(p.numel() for p in var.parameters())/1e6:.1f}M, "
          f"scales {mv.PATCH}, L={var.L}", flush=True)

    opt = torch.optim.AdamW(var.parameters(), lr=args.lr, weight_decay=0.05,
                            betas=(0.9, 0.95))
    train = mv.mnist_loader(True, batch=128)
    test = mv.mnist_loader(False, batch=128, workers=2)
    out = Path(args.out)
    best = float("inf")
    started = time.time()
    for ep in range(args.epochs):
        var.train()
        tot = n = 0.0
        for x, y in train:
            x, y = x.to(device, non_blocking=True), y.to(device)
            with torch.no_grad():
                gt = vae.img_to_idxBl(x)
                xin = var_input(vae, gt, device)
            truth = torch.cat(gt, dim=1)
            logits = var(y, xin, truth=truth)
            loss = F.cross_entropy(logits.reshape(-1, mv.VOCAB), truth.reshape(-1))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(var.parameters(), 1.0)
            opt.step()
            tot += float(loss) * x.shape[0]
            n += x.shape[0]
        var.eval()
        var.cond_drop_rate = 0.0
        with torch.no_grad():
            v = m = 0.0
            for x, y in test:
                x, y = x.to(device), y.to(device)
                gt = vae.img_to_idxBl(x)
                truth = torch.cat(gt, dim=1)
                logits = var(y, var_input(vae, gt, device), truth=truth)
                v += float(F.cross_entropy(logits.reshape(-1, mv.VOCAB),
                                           truth.reshape(-1))) * x.shape[0]
                m += x.shape[0]
        var.cond_drop_rate = 0.1
        mark = ""
        if v / m < best:
            best = v / m
            torch.save(var.state_dict(), out / "var_within_scale.pt")
            mark = "  <- kept"
        print(f"epoch {ep+1}/{args.epochs} train {tot/n:.4f} test {v/m:.4f} "
              f"({time.time()-started:.0f}s){mark}", flush=True)
    print(f"\nbest {best:.4f} nats/token, written to {out}/var_within_scale.pt")
    print("the parallel VAR on the same tokeniser reaches 2.0618; the gap is what "
          "modelling the within-scale joint recovers, and the ceiling for a PTP "
          "student distilled from this teacher")


if __name__ == "__main__":
    main()
