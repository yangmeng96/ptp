"""Sample with the block student in place of VAR's first K scales, and score it.

Exact agreement with the teacher's choice is sufficient for a correct student
and far from necessary. The teacher itself gives the token a real image actually
has a median probability of 0.001 and still generates at FID 7.8: it is drawing
from a distribution, not reproducing that image. What a student has to preserve
is the measure -- for each token, the share of u that selects it -- and where in
[0,1] those u land does not matter to the marginal at all. So agreement is the
wrong headline and this is the right one.

The KV cache is the part that needs care. VAR's later scales attend back to the
first K, and those cache entries were computed from word_embed(f_hat) inputs. A
student driven by u would leave the wrong keys there, so once its tokens are
decided one VAR forward re-runs those positions with the inputs the frozen
scales expect. That forward costs about 11 ms -- the flat part of the latency
curve -- so K scales still collapse to two passes rather than one.
"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, "/home/mengy13/VAR")


@torch.no_grad()
def sample_with_student(var, student, mask, B, label_B, merge_k, cells=0,
                        gaussian=False, head="ce", cfg=1.5, top_k=900,
                        top_p=0.96, g_seed=None, mode="optp", perm=None):
    from models.helpers import sample_with_top_k_top_p_
    from image_ptp.var_block_student import draw_u
    rng = None
    if g_seed is not None:
        var.rng.manual_seed(g_seed)
        rng = var.rng
    pn, SN = var.patch_nums, len(var.patch_nums)
    dev = var.lvl_1L.device
    merged_L = sum(p * p for p in pn[:merge_k])
    emb = var.vae_quant_proxy[0].embedding

    # ---- 1. the student, one forward, from u alone
    if cells:
        u = torch.randint(0, cells, (B, merged_L), device=dev)
    else:
        u = torch.rand(B, merged_L, device=dev)
        u = draw_u(u, u, gaussian)
    ce, ms = student(u, label_B, mask)
    if mode == "cptp":
        # the head is the teacher's distribution; the token comes from inverting
        # its CDF at this slot's own u, and under an embedding ordering the
        # index the inversion returns is a rank in that order, not a code id
        cq = ce.softmax(-1).cumsum(-1)
        idx = torch.searchsorted(cq.contiguous(), u.unsqueeze(-1).contiguous()
                                 ).squeeze(-1).clamp(max=ce.shape[-1] - 1)
        if perm is not None:
            idx = perm.to(idx.device)[idx]
    elif head == "ce":
        idx = ce.argmax(-1)
    else:
        # nearest codebook entry to the regressed vector: the frozen scales only
        # ever see f_hat, so the code identity is a means, not the target
        idx = torch.cdist(ms.reshape(-1, ms.shape[-1]),
                          emb.weight).argmin(-1).view(B, merged_L)

    return continue_from_block(var, idx, label_B, merge_k, cfg, top_k, top_p, rng)


@torch.no_grad()
def continue_from_block(var, idx, label_B, merge_k, cfg=1.5, top_k=900,
                        top_p=0.96, rng=None):
    """Given the first merge_k scales' tokens, however they were chosen, run the
    frozen scales after them. Fills VAR's own KV cache for those positions from
    word_embed(f_hat) under VAR's own mask -- verified against teacher forcing to
    2.9e-5 -- so the later scales read exactly what they were trained on."""
    from models.helpers import sample_with_top_k_top_p_
    pn, SN = var.patch_nums, len(var.patch_nums)
    dev = var.lvl_1L.device
    B = idx.shape[0]
    merged_L = sum(p * p for p in pn[:merge_k])
    emb = var.vae_quant_proxy[0].embedding

    # ---- 2. f_hat from those tokens, and the inputs the frozen scales expect
    f_hat = torch.zeros(B, var.Cvae, pn[-1], pn[-1], device=dev)
    maps, off = [], 0
    for si in range(merge_k):
        p = pn[si]
        h = emb(idx[:, off:off + p * p]).transpose_(1, 2).reshape(B, var.Cvae, p, p)
        f_hat, ntm = var.vae_quant_proxy[0].get_next_autoregressive_input(
            si, SN, f_hat, h)
        maps.append(ntm)
        off += p * p

    # ---- 3. one VAR forward over the block, to leave the cache the later
    #         scales were trained to read, under VAR's own mask
    sos = cond_BD = var.class_emb(torch.cat(
        (label_B, torch.full_like(label_B, var.num_classes)), dim=0))
    lvl_pos = var.lvl_embed(var.lvl_1L) + var.pos_1LC
    parts = [sos.unsqueeze(1).expand(2 * B, var.first_l, -1)
             + var.pos_start.expand(2 * B, var.first_l, -1)]
    for si in range(merge_k - 1):
        m = maps[si].view(B, var.Cvae, -1).transpose(1, 2)
        parts.append(var.word_embed(m).repeat(2, 1, 1))
    x = torch.cat(parts, dim=1) + lvl_pos[:, :merged_L]
    bias = var.attn_bias_for_masking[:, :, :merged_L, :merged_L]
    for b in var.blocks:
        b.attn.kv_caching(True)
    cond_gss = var.shared_ada_lin(cond_BD)
    for b in var.blocks:
        x = b(x=x, cond_BD=cond_gss, attn_bias=bias.to(x.dtype))

    # ---- 4. the remaining scales, unmodified
    nxt = pn[merge_k] ** 2
    ntm = maps[-1].view(B, var.Cvae, -1).transpose(1, 2)
    ntm = (var.word_embed(ntm) + lvl_pos[:, merged_L:merged_L + nxt]).repeat(2, 1, 1)
    cur_L = merged_L
    for si in range(merge_k, SN):
        cur_L += pn[si] ** 2
        x = ntm
        for b in var.blocks:
            x = b(x=x, cond_BD=cond_gss, attn_bias=None)
        logits = var.get_logits(x, cond_BD)
        t = cfg * si / var.num_stages_minus_1
        logits = (1 + t) * logits[:B] - t * logits[B:]
        ix = sample_with_top_k_top_p_(logits, rng=rng, top_k=top_k, top_p=top_p,
                                      num_samples=1)[:, :, 0]
        h = emb(ix).transpose_(1, 2).reshape(B, var.Cvae, pn[si], pn[si])
        f_hat, ntm = var.vae_quant_proxy[0].get_next_autoregressive_input(
            si, SN, f_hat, h)
        if si != SN - 1:
            n2 = pn[si + 1] ** 2
            ntm = ntm.view(B, var.Cvae, -1).transpose(1, 2)
            ntm = (var.word_embed(ntm)
                   + lvl_pos[:, cur_L:cur_L + n2]).repeat(2, 1, 1)
    for b in var.blocks:
        b.attn.kv_caching(False)
    return var.vae_proxy[0].fhat_to_img(f_hat).add_(1).mul_(0.5)


@torch.no_grad()
def sample_with_ar(var, ar, B, label_B, merge_k, cfg=1.5, top_k=900,
                   top_p=0.96, g_seed=None):
    """30 sequential steps over the block, each seeing the tokens sampled before
    it, then the same continuation the student uses. This is the PTP ceiling:
    the same block, the same data, no single-forward constraint and no u."""
    rng = None
    if g_seed is not None:
        var.rng.manual_seed(g_seed)
        rng = var.rng
    dev = var.lvl_1L.device
    L, bos = ar.L, ar.vocab
    fed = torch.full((B, L), bos, dtype=torch.long, device=dev)
    idx = torch.zeros(B, L, dtype=torch.long, device=dev)
    g = torch.Generator(device=dev).manual_seed(0 if g_seed is None else g_seed)
    for i in range(L):
        logits = ar(fed, label_B)[:, i].float()
        p = torch.softmax(logits, -1)
        idx[:, i] = torch.multinomial(p, 1, generator=g).squeeze(1)
        if i + 1 < L:
            fed[:, i + 1] = idx[:, i]
    return continue_from_block(var, idx, label_B, merge_k, cfg, top_k, top_p, rng)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--students", required=True,
                    help="comma-separated name=checkpoint pairs")
    ap.add_argument("--vae", default="/home/mengy13/VAR/checkpoints/vae_ch160v4096z32.pth")
    ap.add_argument("--var", default="/home/mengy13/VAR/checkpoints/var_d16.pth")
    ap.add_argument("--data", default="/extra/ucibdl1/shared/data/imagenet/val")
    ap.add_argument("--images", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=25)
    ap.add_argument("--heads", default="ce,mse")
    ap.add_argument("--grid", type=int, default=8,
                    help="images per arm saved for visual comparison")
    ap.add_argument("--grid-out",
                    default="/home/mengy13/ptp-image-results/block_samples.png")
    args = ap.parse_args()
    device = "cuda"
    torch.set_grad_enabled(False)

    from models import build_vae_var
    from image_ptp.fid_features import (InceptionFeatures, frechet,
                                        check_sample_count)
    from image_ptp.var_block_student import BlockStudent, scale_causal_mask
    from image_ptp.merge_sample import sample_merged, expected_prefix
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

    grids = {}

    def score(name, sampler):
        feats = []
        for i in range(0, args.images, args.batch):
            b = min(args.batch, args.images - i)
            lab = torch.randint(0, 1000, (b,), device=device,
                                generator=torch.Generator(device=device).manual_seed(i))
            img = sampler(b, lab, i)
            # every arm draws the same labels for batch 0, so the grids line up
            if i == 0:
                grids[name] = img[:args.grid].cpu()
            feats.append(ext(img * 2 - 1).cpu())
        print(f"{name:<34} {frechet(torch.cat(feats), f_real):9.3f}", flush=True)

    print(f"{'arm':<34} {'FID':>9}")
    score("official, 10 forwards",
          lambda b, lab, i: sample_merged(var, b, lab, 1, g_seed=i))

    for spec in args.students.split(","):
        name, ck = spec.split("=", 1)
        blob = torch.load(ck, map_location="cpu")
        a = blob["args"]
        sd = blob["model"]
        scales, cells = 4, 0
        if "vocab" in blob and "head.weight" in sd:          # a BlockAR
            from image_ptp.block_ar import BlockAR
            ar = BlockAR(pn, blob["scales"], vocab=blob["vocab"],
                         n_class=sd["cls.weight"].shape[0] - 1,
                         dim=a["dim"], depth=a["depth"])
            ar.load_state_dict(sd)
            ar = ar.to(device).eval()
            score(f"merge {scales} + {name} [AR, {ar.L} steps]",
                  lambda b, lab, i: sample_with_ar(var, ar, b, lab, scales,
                                                   g_seed=i))
            continue
        mode = a.get("mode", "optp")
        st = BlockStudent(pn, scales, vocab=sd["head_ce.weight"].shape[0],
                          n_class=sd["cls.weight"].shape[0] - 1,
                          dim=a["dim"], depth=a["depth"], n_cells=cells,
                          u_encoding=a.get("u_encoding", "binary"), mode=mode)
        st.load_state_dict(sd)
        st = st.to(device).eval()
        mask = scale_causal_mask(pn, scales, device, mode)
        # merge without the student, on the same code path, so the student's
        # contribution is the only difference between these two rows
        pre = [t.to(device) for t in expected_prefix(var, scales)]
        score(f"merge {scales}, no student",
              lambda b, lab, i: sample_merged(var, b, lab, scales, g_seed=i,
                                              prefix=pre))
        perm = None
        if a.get("train", "").find("embed") >= 0:
            perm = torch.load(a["train"], map_location="cpu").get("perm")
        heads = ["ce"] if mode == "cptp" else args.heads.split(",")
        for head in heads:
            score(f"merge {scales} + {name} [{mode}/{head}]",
                  lambda b, lab, i, h=head: sample_with_student(
                      var, st, mask, b, lab, scales, gaussian=a["gaussian"],
                      head=h, g_seed=i, mode=mode, perm=perm))
    if grids:
        from torchvision.utils import save_image
        names = list(grids)
        rows = torch.cat([grids[n] for n in names])
        save_image(rows, args.grid_out, nrow=args.grid, padding=2)
        print(f"\nwrote {args.grid_out}: one row per arm, same labels across rows")
        for k, n in enumerate(names):
            print(f"  row {k + 1}: {n}")
    print("STUDENT_SAMPLE_DONE")


if __name__ == "__main__":
    main()
