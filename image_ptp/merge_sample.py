"""Sample from VAR with its first K scales emitted in a single forward pass.

This is the arm PTP would have to beat, and it needs no training: the first K
scales are drawn from the same model with no information about each other, which
is what one parallel pass gives you. Merging the first four costs 1.5% of the
image's description length under teacher forcing and saves 18% of the latency;
whether that 1.5% matters to the samples is what this measures.

Everything else -- tokeniser, transformer, guidance, truncation -- is VAR's own
and untouched, so the only difference from the official sampler is where the
scale boundaries are.
"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, "/home/mengy13/VAR")


@torch.no_grad()
def expected_prefix(var, merge_k, cfg=1.5, chunk=250):
    """Per class, the f_hat a merged block's scales would see if they knew only
    the class -- not what the earlier scales in their own block sampled.

    Feeding word_embed(0) instead is not a neutral input: it is that layer's
    bias, a vector the model never saw at those positions, so a merged arm built
    on it confounds "lost the in-block dependence" with "given something out of
    distribution". The expectation of the embedding under the model's own
    distribution is in distribution by construction -- it is a convex
    combination of codebook vectors -- and carries only what the class already
    implies. VAR's own more_smooth path does the same averaging.

    Computed once for every class and cached, so inference costs no extra pass.
    """
    pn = var.patch_nums
    SN = len(pn)
    dev = var.lvl_1L.device
    lvl_pos = var.lvl_embed(var.lvl_1L) + var.pos_1LC
    out = [[] for _ in range(merge_k - 1)]
    for lo in range(0, var.num_classes, chunk):
        lab = torch.arange(lo, min(lo + chunk, var.num_classes), device=dev)
        B = lab.shape[0]
        sos = cond_BD = var.class_emb(torch.cat(
            (lab, torch.full_like(lab, var.num_classes)), dim=0))
        f_hat = sos.new_zeros(B, var.Cvae, pn[-1], pn[-1])
        ntm = (sos.unsqueeze(1).expand(2 * B, var.first_l, -1)
               + var.pos_start.expand(2 * B, var.first_l, -1)
               + lvl_pos[:, :var.first_l])
        for b in var.blocks:
            b.attn.kv_caching(True)
        cur_L = 0
        for si in range(merge_k - 1):
            cur_L += pn[si] ** 2
            x = ntm
            cond_gss = var.shared_ada_lin(cond_BD)
            for b in var.blocks:
                x = b(x=x, cond_BD=cond_gss, attn_bias=None)
            logits = var.get_logits(x, cond_BD)
            t = cfg * si / var.num_stages_minus_1
            logits = (1 + t) * logits[:B] - t * logits[B:]
            # expectation over the codebook instead of a draw from it
            h = logits.softmax(-1) @ var.vae_quant_proxy[0].embedding.weight
            h = h.transpose_(1, 2).reshape(B, var.Cvae, pn[si], pn[si])
            f_hat, ntm = var.vae_quant_proxy[0].get_next_autoregressive_input(
                si, SN, f_hat, h)
            out[si].append(ntm.clone().cpu())
            nxt = pn[si + 1] ** 2
            ntm = ntm.view(B, var.Cvae, -1).transpose(1, 2)
            ntm = (var.word_embed(ntm) + lvl_pos[:, cur_L:cur_L + nxt]).repeat(2, 1, 1)
        for b in var.blocks:
            b.attn.kv_caching(False)
    return [torch.cat(v) for v in out]      # per scale: (num_classes, C, pn, pn)


@torch.no_grad()
def sample_merged(var, B, label_B, merge_k, cfg=1.5, top_k=900, top_p=0.96,
                  g_seed=None, prefix=None):
    """VAR's own loop, with scales [0, merge_k) emitted in one pass.

    merge_k=1 is the unmodified sampler. The merged positions are fed the input
    that exists before the block -- f_hat is still zero there -- so none of them
    carries information about what the others produced.
    """
    from models.helpers import sample_with_top_k_top_p_
    rng = None
    if g_seed is not None:
        var.rng.manual_seed(g_seed)
        rng = var.rng
    pn = var.patch_nums
    SN = len(pn)
    sos = cond_BD = var.class_emb(torch.cat(
        (label_B, torch.full_like(label_B, var.num_classes)), dim=0))
    lvl_pos = var.lvl_embed(var.lvl_1L) + var.pos_1LC
    f_hat = sos.new_zeros(B, var.Cvae, pn[-1], pn[-1])

    merged_L = sum(p * p for p in pn[:merge_k])
    # Positions past the first scale would normally carry word_embed of the
    # accumulated f_hat; inside the merged block that f_hat does not exist yet,
    # so they get what a zero f_hat gives, plus their own level and position.
    first = (sos.unsqueeze(1).expand(2 * B, var.first_l, -1)
             + var.pos_start.expand(2 * B, var.first_l, -1))
    if prefix is None:
        rest = var.word_embed(sos.new_zeros(2 * B, merged_L - var.first_l, var.Cvae))
    else:
        # Each merged scale gets the f_hat its class implies, marginal over what
        # the earlier scales of its own block would have sampled.
        rest = torch.cat([
            var.word_embed(prefix[j][label_B]
                           .view(B, var.Cvae, -1).transpose(1, 2)).repeat(2, 1, 1)
            for j in range(merge_k - 1)], dim=1)
    next_token_map = torch.cat([first, rest], dim=1) + lvl_pos[:, :merged_L]

    for b in var.blocks:
        b.attn.kv_caching(True)
    cur_L, si = 0, 0
    while si < SN:
        n_scales = merge_k if si == 0 else 1
        span = sum(p * p for p in pn[si:si + n_scales])
        cur_L += span
        x = next_token_map
        cond_gss = var.shared_ada_lin(cond_BD)
        for b in var.blocks:
            x = b(x=x, cond_BD=cond_gss, attn_bias=None)
        logits = var.get_logits(x, cond_BD)
        # Guidance ramps per scale, so a merged pass needs it per position.
        t = torch.cat([torch.full((p * p,), cfg * (si + j) / var.num_stages_minus_1,
                                  device=logits.device)
                       for j, p in enumerate(pn[si:si + n_scales])])
        t = t.view(1, -1, 1)
        logits = (1 + t) * logits[:B] - t * logits[B:]
        idx = sample_with_top_k_top_p_(logits, rng=rng, top_k=top_k,
                                       top_p=top_p, num_samples=1)[:, :, 0]
        off = 0
        for j, p in enumerate(pn[si:si + n_scales]):
            h = var.vae_quant_proxy[0].embedding(idx[:, off:off + p * p])
            h = h.transpose_(1, 2).reshape(B, var.Cvae, p, p)
            f_hat, next_token_map = var.vae_quant_proxy[0] \
                .get_next_autoregressive_input(si + j, SN, f_hat, h)
            off += p * p
        si += n_scales
        if si != SN:
            nxt = pn[si] ** 2
            next_token_map = next_token_map.view(B, var.Cvae, -1).transpose(1, 2)
            next_token_map = (var.word_embed(next_token_map)
                              + lvl_pos[:, cur_L:cur_L + nxt]).repeat(2, 1, 1)
    for b in var.blocks:
        b.attn.kv_caching(False)
    return var.vae_proxy[0].fhat_to_img(f_hat).add_(1).mul_(0.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vae", default="/home/mengy13/VAR/checkpoints/vae_ch160v4096z32.pth")
    ap.add_argument("--var", default="/home/mengy13/VAR/checkpoints/var_d16.pth")
    ap.add_argument("--data", default="/extra/ucibdl1/shared/data/imagenet/val")
    ap.add_argument("--images", type=int, default=20000)
    ap.add_argument("--merge", default="1,2,3,4,5,7")
    ap.add_argument("--batch", type=int, default=25)
    ap.add_argument("--prefix", default="expected",
                    choices=["expected", "zero"])
    args = ap.parse_args()
    device = "cuda"
    torch.set_grad_enabled(False)

    from models import build_vae_var
    from image_ptp.fid_features import (InceptionFeatures, frechet,
                                        check_sample_count)
    pn = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
    vae, var = build_vae_var(device=device, patch_nums=pn, num_classes=1000,
                             depth=16, shared_aln=False)
    vae.load_state_dict(torch.load(args.vae, map_location="cpu"), strict=True)
    var.load_state_dict(torch.load(args.var, map_location="cpu"), strict=True)
    vae.eval(); var.eval()

    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader, Subset
    tfm = transforms.Compose([transforms.Resize(292), transforms.CenterCrop(256),
                              transforms.ToTensor(),
                              transforms.Normalize((0.5,) * 3, (0.5,) * 3)])
    ds = datasets.ImageFolder(args.data, transform=tfm)
    g = torch.Generator().manual_seed(0)
    sub = Subset(ds, torch.randperm(len(ds), generator=g)[:args.images].tolist())
    ext = InceptionFeatures(device=device)
    f_real = torch.cat([ext(x.to(device)).cpu()
                        for x, _ in DataLoader(sub, batch_size=16, num_workers=8)])
    check_sample_count(f_real.shape[0], f_real.shape[1])
    print(f"{f_real.shape[0]} real ImageNet val images\n")
    print(f"{'merged':>8} {'forwards':>9} {'FID':>9}")

    ms = [11.68, 11.17, 11.19, 11.30, 11.39, 11.53, 16.43, 24.28, 34.84, 42.08]
    use_exp = args.prefix == "expected"
    for k in (int(v) for v in args.merge.split(",")):
        prefix = ([t.to(device) for t in expected_prefix(var, k)]
                  if (use_exp and k > 1) else None)
        feats = []
        for i in range(0, args.images, args.batch):
            b = min(args.batch, args.images - i)
            lab = torch.randint(0, 1000, (b,), device=device,
                                generator=torch.Generator(device=device).manual_seed(i))
            img = sample_merged(var, b, lab, k, g_seed=i, prefix=prefix)
            feats.append(ext(img * 2 - 1).cpu())          # [0,1] -> [-1,1]
        lat = ms[0] + sum(ms[k:])
        print(f"{f'first {k}':>8} {len(pn) - k + 1:>9} "
              f"{frechet(torch.cat(feats), f_real):9.3f}   "
              f"{lat:.0f} ms, {sum(ms) / lat:.2f}x", flush=True)
    print("MERGE_SAMPLE_DONE")


if __name__ == "__main__":
    main()
