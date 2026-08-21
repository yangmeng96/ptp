"""Does reordering the codebook turn relaxed acceptance into one contiguous target?

Bins are laid out in codebook index order, so the j codes a relaxed test would
accept occupy j scattered slivers of [0, 1] rather than one interval. The
student has to learn u -> token across all of them, which is as high-frequency
a function as the exact criterion was.

This script builds orderings that put perceptually similar codes next to each
other and measures what the acceptable set looks like afterwards:

  * runs          how many maximal contiguous blocks the j neighbours form
  * largest run   the mass of the biggest single interval, which is the width
                  of the widest target the student can actually aim at
  * total mass    unchanged by reordering, shown as the ceiling

Orderings compared: the codebook's own order, a random permutation (control),
a greedy nearest-neighbour chain, and a spectral (Fiedler) ordering.

Usage:
    python codebook_reorder.py --llamagen-root ~/LlamaGen \
        --gpt-ckpt ~/LlamaGen/pretrained_models/c2i_B_256.pt \
        --vq-ckpt ~/LlamaGen/pretrained_models/vq_ds16_c2i.pt \
        --out ~/ptp-image-results/reorder.json
"""
import argparse
import json
import sys
from pathlib import Path

import torch


NEIGHBOURHOODS = [4, 8, 16, 32, 64]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--llamagen-root", type=str, required=True)
    parser.add_argument("--gpt-model", type=str, default="GPT-B")
    parser.add_argument("--gpt-ckpt", type=str, required=True)
    parser.add_argument("--vq-model", type=str, default="VQ-16")
    parser.add_argument("--vq-ckpt", type=str, required=True)
    parser.add_argument("--precision", type=str, default="bf16", choices=["none", "fp16", "bf16"])
    parser.add_argument("--codebook-size", type=int, default=16384)
    parser.add_argument("--codebook-embed-dim", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--downsample-size", type=int, default=16)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cls-token-num", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--knn-graph", type=int, default=16, help="k for the spectral graph")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, required=True)
    return parser.parse_args()


def nearest_neighbours(embedding, max_j, chunk=1024):
    v = embedding.shape[0]
    idx = torch.empty((v, max_j), dtype=torch.long, device=embedding.device)
    for start in range(0, v, chunk):
        stop = min(start + chunk, v)
        d = torch.cdist(embedding[start:stop], embedding)
        idx[start:stop] = torch.topk(d, max_j, dim=1, largest=False).indices
    return idx


def greedy_chain_order(embedding, chunk=2048):
    """Walk the codebook by repeatedly stepping to the nearest unvisited code."""
    v = embedding.shape[0]
    device = embedding.device
    visited = torch.zeros(v, dtype=torch.bool, device=device)
    order = torch.empty(v, dtype=torch.long, device=device)
    cur = 0
    for i in range(v):
        order[i] = cur
        visited[cur] = True
        if i == v - 1:
            break
        d = torch.cdist(embedding[cur:cur + 1], embedding).squeeze(0)
        d[visited] = float("inf")
        cur = int(torch.argmin(d))
    return order


def spectral_order(embedding, k):
    """Sort by the Fiedler vector of a symmetric kNN graph over the codebook."""
    import numpy as np
    from scipy.sparse import coo_matrix, csgraph
    from scipy.sparse.linalg import eigsh

    v = embedding.shape[0]
    nn = nearest_neighbours(embedding, k + 1)[:, 1:].cpu().numpy()
    rows = np.repeat(np.arange(v), k)
    cols = nn.reshape(-1)
    data = np.ones(rows.shape[0], dtype=np.float64)
    adj = coo_matrix((data, (rows, cols)), shape=(v, v))
    adj = adj.maximum(adj.T).tocsr()
    lap = csgraph.laplacian(adj, normed=True)
    # Shift-invert around zero pulls out the smallest eigenpairs reliably.
    vals, vecs = eigsh(lap, k=2, sigma=-1e-3, which="LM")
    fiedler = vecs[:, np.argsort(vals)[1]]
    return torch.from_numpy(np.argsort(fiedler).copy()).to(embedding.device)


@torch.no_grad()
def collect(model, sample_fn, cond, seq_len, cfg_scale, **sampling_kwargs):
    if cfg_scale > 1.0:
        cond_combined = torch.cat([cond, torch.ones_like(cond) * model.num_classes])
    else:
        cond_combined = cond
    device = cond.device
    with torch.device(device):
        model.setup_caches(
            max_batch_size=cond.shape[0] * (2 if cfg_scale > 1.0 else 1),
            max_seq_length=1 + seq_len,
            dtype=model.tok_embeddings.weight.dtype,
        )

    def combine(logits):
        if cfg_scale > 1.0:
            c, u = torch.split(logits, len(logits) // 2, dim=0)
            return u + (c - u) * cfg_scale
        return logits

    tokens, dists = [], []
    logits, _ = model(None, cond_combined, torch.arange(0, 1, device=device))
    tok, probs = sample_fn(combine(logits), **sampling_kwargs)
    tokens.append(tok.squeeze(1)); dists.append(probs.float())
    cur = tok.view(-1, 1)
    pos = torch.tensor([1], device=device, dtype=torch.int)
    for _ in range(seq_len - 1):
        inp = torch.cat([cur, cur]) if cfg_scale > 1.0 else cur
        logits, _ = model(inp, cond_idx=None, input_pos=pos)
        tok, probs = sample_fn(combine(logits), **sampling_kwargs)
        tokens.append(tok.squeeze(1)); dists.append(probs.float())
        pos += 1
        cur = tok.view(-1, 1)
    return torch.stack(tokens, 1), torch.stack(dists, 1)


def contiguity(ranks, masses):
    """Given per-neighbour ranks and masses, return (num_runs, largest_run_mass).

    ranks/masses are (P, j); ranks are positions in the reordered codebook.
    """
    order = ranks.argsort(dim=1)
    sorted_ranks = ranks.gather(1, order)
    sorted_mass = masses.gather(1, order)
    # A run breaks wherever consecutive ranks are not adjacent integers.
    breaks = (sorted_ranks[:, 1:] - sorted_ranks[:, :-1]) != 1        # (P, j-1)
    runs = breaks.sum(dim=1) + 1
    # Segment id per element, then scatter-add mass into segments.
    seg = torch.cat([torch.zeros_like(breaks[:, :1]), breaks], dim=1).cumsum(dim=1)
    largest = torch.zeros_like(sorted_mass).scatter_add_(1, seg, sorted_mass).max(dim=1).values
    return runs, largest


def main():
    args = parse_args()
    root = Path(args.llamagen_root).expanduser()
    sys.path.insert(0, str(root))
    from autoregressive.models.gpt import GPT_models
    from autoregressive.models.generate import sample as sample_fn
    from tokenizer.tokenizer_image.vq_model import VQ_models

    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    precision = {"none": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[args.precision]

    vq_model = VQ_models[args.vq_model](
        codebook_size=args.codebook_size,
        codebook_embed_dim=args.codebook_embed_dim,
    ).to(device).eval()
    vq_ckpt = torch.load(args.vq_ckpt, map_location="cpu")
    vq_model.load_state_dict(vq_ckpt["model"])
    del vq_ckpt
    embedding = vq_model.quantize.embedding.weight.detach().float()
    if getattr(vq_model.quantize, "l2_norm", False):
        embedding = torch.nn.functional.normalize(embedding, p=2, dim=-1)
    del vq_model

    v = embedding.shape[0]
    nn_idx = nearest_neighbours(embedding, max(NEIGHBOURHOODS))

    orderings = {"identity": torch.arange(v, device=device)}
    g = torch.Generator(device="cpu").manual_seed(args.seed)
    orderings["random"] = torch.randperm(v, generator=g).to(device)
    print("building greedy chain ordering ...")
    orderings["greedy_chain"] = greedy_chain_order(embedding)
    try:
        print("building spectral ordering ...")
        orderings["spectral"] = spectral_order(embedding, args.knn_graph)
    except Exception as exc:  # scipy missing or eigensolver failed
        print(f"spectral ordering unavailable: {exc}")

    # rank[code] = where that code sits in the ordering
    ranks = {}
    for name, order in orderings.items():
        r = torch.empty(v, dtype=torch.long, device=device)
        r[order] = torch.arange(v, device=device)
        ranks[name] = r

    latent_size = args.image_size // args.downsample_size
    seq_len = latent_size ** 2
    model = GPT_models[args.gpt_model](
        vocab_size=args.codebook_size, block_size=seq_len,
        num_classes=args.num_classes, cls_token_num=args.cls_token_num,
        model_type="c2i",
    ).to(device=device, dtype=precision)
    ckpt = torch.load(args.gpt_ckpt, map_location="cpu")
    for key in ("model", "module", "state_dict"):
        if key in ckpt:
            ckpt = ckpt[key]
            break
    model.load_state_dict(ckpt, strict=False)
    model.eval()
    del ckpt

    cond = torch.randint(0, args.num_classes, (args.batch_size,), device=device)
    tokens, dists = collect(
        model, sample_fn, cond, seq_len, cfg_scale=args.cfg_scale,
        temperature=args.temperature, top_k=args.top_k, top_p=args.top_p,
    )
    flat_tokens = tokens.reshape(-1).long()
    flat_dists = dists.reshape(-1, v)
    print(f"collected {flat_tokens.numel()} positions")

    result = {
        "config": {
            "cfg_scale": args.cfg_scale, "top_k": args.top_k, "top_p": args.top_p,
            "temperature": args.temperature, "positions": int(flat_tokens.numel()),
            "gpt_model": args.gpt_model, "codebook_size": v,
        },
        "orderings": {},
    }

    for j in NEIGHBOURHOODS:
        nbrs = nn_idx[flat_tokens][:, :j]
        masses = flat_dists.gather(1, nbrs)
        total = masses.sum(dim=1)
        for name, rank in ranks.items():
            runs, largest = contiguity(rank[nbrs], masses)
            entry = result["orderings"].setdefault(name, [])
            entry.append({
                "j": j,
                "total_mass_median": total.median().item(),
                "largest_run_mass_median": largest.median().item(),
                "largest_run_frac_of_total": (largest / total.clamp(min=1e-9)).median().item(),
                "runs_median": runs.float().median().item(),
                "runs_mean": runs.float().mean().item(),
                "frac_largest_run_above_0.1": (largest > 0.1).float().mean().item(),
            })
        print(f"j={j:3d} total mass median={total.median().item():.4f}")
        for name in ranks:
            e = result["orderings"][name][-1]
            print(f"      {name:>13s} | runs={e['runs_median']:.0f} "
                  f"largest run={e['largest_run_mass_median']:.4f} "
                  f"({e['largest_run_frac_of_total'] * 100:.0f}% of mass) "
                  f"| >0.1: {e['frac_largest_run_above_0.1']:.3f}")

    Path(args.out).expanduser().write_text(json.dumps(result, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
