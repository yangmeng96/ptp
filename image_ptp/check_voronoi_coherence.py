"""Does a cell assignment actually put each cell near the token that owns it?

Quotas and geometry are separable, and a solver can satisfy the first while
destroying the second. That is not hypothetical: the upstream Triton auction
failed to load its compiled launcher on one node, kept running, and produced an
assignment whose quotas matched the reference to seven figures while agreeing
with it on 0.14% of positions -- and the student trained on it collapsed from
correct 1.447 to 0.109.

Coherence here is the mean cosine between a cell's sphere vector and the output
embedding of the token that owns the cell, against the mean over a shuffled
ownership. A geometric assignment scores well above its shuffle; a quota-only
one scores at it.
"""
import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def coherence(owner, S):
    """owner: (P, K) token id per cell, -1 for unassigned. S: (V, K) similarity.

    Returns (assigned, shuffled): the mean similarity between a cell and the
    token that owns it, and the same after permuting the cells among themselves.
    A geometric assignment scores well above its shuffle; a quota-only one does
    not, because it has satisfied the counts without using the geometry.
    """
    P, K = owner.shape
    cells = torch.arange(K, device=owner.device)[None, :].expand(P, K)
    ok = owner >= 0
    idx = owner.clamp_min(0)
    real = S[idx, cells][ok].mean()
    perm = torch.randperm(K, device=owner.device)
    shuffled = S[idx[:, perm], cells][ok].mean()
    return float(real), float(shuffled)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher-ckpt",
                    default="/home/mengy13/ptp-vqvae/checkpoints/ar_cifar10_raster.pt")
    ap.add_argument("--tokens", default="/home/mengy13/ptp-image-results/cifar_tokens.pt")
    ap.add_argument("--K", type=int, default=4096)
    ap.add_argument("--top-m", type=int, default=256)
    ap.add_argument("--upstream", default="/home/mengy13/extract-auxiliaries")
    ap.add_argument("--images", type=int, default=4)
    args = ap.parse_args()

    import image_ptp.prepare_voronoi_tokens as pv
    from image_ptp.vqvae_ar_hf import build

    device = "cuda"
    torch.set_grad_enabled(False)
    teacher, meta = build(args.teacher_ckpt, device=device, dtype=torch.float32)
    teacher.eval()
    V, BOS, K = meta["num_codes"], meta["num_codes"], args.K
    W = None
    for n, p in teacher.named_parameters():
        if p.dim() == 2 and p.shape[0] == V + 1:
            W = p.data
    positions = F.normalize(W[:V].float(), dim=-1)
    g = torch.Generator(device=device).manual_seed(0)
    sphere = F.normalize(torch.randn(K, W.shape[1], generator=g, device=device), dim=-1)
    S = (positions @ sphere.T).contiguous()

    d = torch.load(args.tokens, map_location="cpu")
    t = d["tokens"][:args.images].long().to(device)
    ids = torch.cat([torch.full((t.shape[0], 1), BOS, device=device), t], 1)
    Q = torch.softmax(teacher(input_ids=ids).logits[:, :-1, :V].float(), -1).reshape(-1, V)

    print(f"{args.images} images -> {Q.shape[0]} positions, K={K}\n")
    print(f"{'solver':<24} {'coherence':>10} {'shuffled':>10} {'ratio':>7}")

    o = pv.assign_cells(Q, S, K, max_active=args.top_m)
    r, s = coherence(o, S)
    print(f"{'greedy (pure torch)':<24} {r:10.4f} {s:10.4f} {r/abs(s) if s else 0:7.2f}")

    try:
        sys.path.insert(0, args.upstream)
        from proportional_assignment import assign as ref_assign
        o2 = pv.assign_cells_referee(Q, S, K, args.top_m, ref_assign)
        r2, s2 = coherence(o2, S)
        print(f"{'referee (triton auction)':<24} {r2:10.4f} {s2:10.4f} "
              f"{r2/abs(s2) if s2 else 0:7.2f}")
        print(f"\ngreedy vs referee agreement: "
              f"{float((o == o2).float().mean()):.4f}")
    except Exception as e:
        print(f"referee unavailable: {type(e).__name__}: {e}")
    print("COHERENCE_DONE")


if __name__ == "__main__":
    main()
