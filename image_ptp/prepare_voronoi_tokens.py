"""CIFAR training data for the quota-assigned Voronoi auxiliary scheme.

The last untested ingredient of the Voronoi scheme. The scalar-u ladder is
finished: reorderings of the CDF all lost to vocabulary order (per-position
likelihood -46%, geometric -13%, global -5.5%, against a noise floor of 0.2%),
and a discrete embedding table over the same scalar u was neutral. What no
experiment has touched is replacing the one-dimensional interval itself with
cells over a fixed set of unit vectors.

Geometry alone does not give the cells the right mass -- measured here at
V=K=512, the best any pure additive rule manages is a total variation of 0.33
against the teacher, with no sweet spot -- so the masses are pinned by quota,
as the upstream `proportional` backend does: token v receives
largest-remainder-round(Q_v * K) auxiliaries, and a greedy pass hands each
token, in descending quota order, the highest-scoring auxiliaries still free.
The rule is deterministic given Q, so the same cells can be rebuilt anywhere.

The auxiliary drawn for the true token is stored as u = (k + 0.5) / (K + 1),
which RoundingEmbedding with K+1 bins decodes back to exactly k: the id rides
through the repo's scalar-u machinery untouched, and the embedding table is the
auxiliary vocabulary. A true token whose quota rounds to zero gets the mask id
K -- the position trains toward its target with an auxiliary that says only
"not one of mine", the scheme's mask_auxiliary.
"""
import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, "/home/mengy13/ptp")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--teacher-ckpt", type=str,
                   default="/home/mengy13/ptp-vqvae/checkpoints/ar_cifar10_raster.pt")
    p.add_argument("--tokens", type=str, required=True,
                   help="existing payload; its tokens and labels are reused")
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--K", type=int, default=512)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--solver", type=str, default="greedy",
                   choices=["greedy", "referee"],
                   help="'greedy' is the local single-pass assignment, fine at "
                        "K=512. 'referee' calls ptp-on-alps's Triton auction, "
                        "which is what makes larger K tractable -- the greedy "
                        "loops once per target with a nonzero quota, and that "
                        "count grows with K. Both take the same quotas, which "
                        "were checked elementwise against the referee's own")
    p.add_argument("--top-m", type=int, default=256,
                   help="referee only: candidate targets per position. A true "
                        "token outside this list is masked, so it trades the "
                        "cost table against the mask rate")
    p.add_argument("--upstream", type=str, default="/home/mengy13/upstream_ptp")
    p.add_argument("--max-active", type=int, default=128,
                   help="tokens with nonzero quota per position are capped here; "
                        "the effective candidate count is ~32, so 128 is slack")
    return p.parse_args()


def quotas_from_probs(Q, K):
    """Largest-remainder rounding of Q * K to integers summing exactly to K."""
    raw = Q * K
    base = raw.floor()
    short = (K - base.sum(dim=-1)).long()                       # (P,)
    frac = raw - base
    order = frac.argsort(dim=-1, descending=True)
    bump = torch.arange(Q.shape[-1], device=Q.device)[None, :] < short[:, None]
    add = torch.zeros_like(base)
    add.scatter_(1, order, bump.to(base.dtype))
    return (base + add).long()                                   # (P, V)


def assign_cells(Q, S, K, max_active):
    """Greedy quota assignment; returns owner (P, K), -1 where unassigned.

    Tokens are processed in descending quota (ties by id), each taking its
    quota-many best-scoring auxiliaries still free. Deterministic in Q and S.
    """
    P, V = Q.shape
    device = Q.device
    quota = quotas_from_probs(Q, K)                              # (P, V)
    order = (quota * V - torch.arange(V, device=device)[None, :]).argsort(
        dim=-1, descending=True)                                 # quota desc, id asc
    owner = torch.full((P, K), -1, dtype=torch.long, device=device)
    taken = torch.zeros(P, K, dtype=torch.bool, device=device)
    arangeK = torch.arange(K, device=device)[None, :]
    for r in range(min(max_active, V)):
        tok = order[:, r]                                        # (P,)
        q_r = quota.gather(1, tok[:, None]).squeeze(1)           # (P,)
        if int(q_r.max()) == 0:
            break
        score = S[tok]                                           # (P, K)
        score = score.masked_fill(taken, float("-inf"))
        # Only the q_r best matter, and q_r is small once the largest few
        # targets are placed; a full argsort of K is what made this unusable at
        # K = 16384, where sorting 16384 entries per position per round dominated
        # everything else.
        top = int(q_r.max())
        srt = score.topk(top, dim=-1).indices                    # (P, top)
        pick = torch.arange(top, device=device)[None, :] < q_r[:, None]
        chosen = torch.zeros_like(taken)
        chosen.scatter_(1, srt, pick)
        owner = torch.where(chosen, tok[:, None], owner)
        taken |= chosen
    return owner



def assign_cells_referee(Q, S, K, top_m, ref_assign):
    """Quota assignment via the upstream Triton auction; same quotas as greedy.

    The cost table is (positions, top_m, K), which is why top_m has to be
    bounded. A target outside the shortlist cannot receive auxiliaries, so its
    positions come back masked.
    """
    P, V = Q.shape
    device = Q.device
    quota = quotas_from_probs(Q, K)                              # (P, V)
    tok = quota.argsort(dim=1, descending=True)[:, :top_m]       # (P, top_m)
    counts = quota.gather(1, tok)
    # Whatever the shortlist drops has to go somewhere or the quotas will not
    # sum to K; hand it to the position's largest target, which is where the
    # rounding noise is least visible.
    short = K - counts.sum(dim=1)
    counts[:, 0] += short
    cost = (-S[tok]).to(torch.bfloat16).contiguous()             # (P, top_m, K)
    sigma = ref_assign(cost, counts=counts.to(torch.int32), k=8, rounds=3)
    owner = tok.gather(1, sigma.clamp(0, top_m - 1))
    return torch.where(sigma < top_m, owner, torch.full_like(owner, -1))


def main():
    args = parse_args()
    device = "cuda"
    torch.set_grad_enabled(False)
    from image_ptp.vqvae_ar_hf import build

    teacher, meta = build(args.teacher_ckpt, device=device, dtype=torch.float32)
    teacher.eval()
    V, BOS, K = meta["num_codes"], meta["num_codes"], args.K

    W = None
    for n, p in teacher.named_parameters():
        if p.dim() == 2 and p.shape[0] == V + 1:
            W, head = p.data, n
    positions = F.normalize(W[:V].float(), dim=-1)
    g = torch.Generator(device=device).manual_seed(args.seed)
    sphere = F.normalize(torch.randn(K, W.shape[1], generator=g, device=device),
                         dim=-1)
    S = (positions @ sphere.T).contiguous()                      # (V, K)
    print(f"head {head}, sphere {K} x {W.shape[1]}, seed {args.seed}")

    ref_assign = None
    if args.solver == "referee":
        sys.path.insert(0, args.upstream)
        from proportional_assignment import assign as ref_assign
        print(f"solver: upstream auction, top_m {args.top_m}")

    payload = torch.load(Path(args.tokens).expanduser(), map_location="cpu")
    tokens = payload["tokens"].long()
    n_seq, seq_len = tokens.shape
    aux_ids = torch.zeros(n_seq, seq_len, dtype=torch.int16)
    masked_total = 0
    started = time.time()
    gen = torch.Generator(device=device).manual_seed(args.seed + 1)
    for s in range(0, n_seq, args.batch_size):
        t = tokens[s:s + args.batch_size].to(device)
        ids = torch.cat([torch.full((t.shape[0], 1), BOS, device=device), t], 1)
        Q = torch.softmax(teacher(input_ids=ids).logits[:, :-1, :V].float(),
                          dim=-1).reshape(-1, V)
        truth = t.reshape(-1)
        if ref_assign is not None:
            owner = assign_cells_referee(Q, S, K, args.top_m, ref_assign)
        else:
            owner = assign_cells(Q, S, K, args.max_active)        # (P, K)
        mine = owner == truth[:, None]                            # (P, K)
        count = mine.sum(dim=1)
        # uniform draw within the cell; mask id K where the cell is empty
        r = torch.rand(truth.shape[0], generator=gen, device=device)
        pick = (r * count.clamp_min(1)).long()
        csum = mine.long().cumsum(dim=1)
        chosen = ((csum == (pick + 1)[:, None]) & mine).float().argmax(dim=1)
        chosen = torch.where(count > 0, chosen, torch.full_like(chosen, K))
        masked_total += int((count == 0).sum())
        aux_ids[s:s + args.batch_size] = chosen.reshape(t.shape).to(torch.int16).cpu()
        if (s // args.batch_size) % 50 == 0:
            done = min(s + args.batch_size, n_seq)
            rate = done / max(time.time() - started, 1e-9)
            print(f"  {done}/{n_seq}  {rate:.1f} seq/s", flush=True)

    P_total = n_seq * seq_len
    print(f"mask rate: {masked_total / P_total:.4f} "
          f"({masked_total} of {P_total} positions)")

    # The id rides in u; RoundingEmbedding with K+1 bins recovers it exactly.
    u = (aux_ids.float() + 0.5) / (K + 1)
    payload["left_bin_edges"] = u
    payload["right_bin_edges"] = u.clone()
    payload["config"] = dict(payload.get("config", {}))
    payload["config"].update({"auxiliary": "voronoi-proportional", "K": K,
                              "sphere_seed": args.seed, "head": head,
                              "mask_rate": masked_total / P_total})
    payload["aux_ids"] = aux_ids
    # The trainer must embed the *same* vectors the assignment used. Rebuilding
    # them from a seed is not enough: a CUDA generator and a CPU generator with
    # one seed give different draws, so the two ends silently disagreed.
    payload["sphere"] = sphere.cpu()
    out = Path(args.out).expanduser()
    torch.save(payload, out)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
