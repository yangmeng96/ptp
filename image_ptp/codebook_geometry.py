"""How much of the teacher's uncertainty is perceptual redundancy?

The exact O-PTP acceptance test needs the student to hit one codebook index.
If the teacher's probability mass instead spreads over codes that decode to
near-identical patches, a relaxed test -- accept anything within a small
neighbourhood in codebook space -- recovers most of that mass, and the
effective bin width grows accordingly.

This script measures, at each generated position:
  * exact bin width          Q(sampled)
  * relaxed bin width        sum of Q over the j nearest codes to the sampled one
  * index-space contiguity   how scattered those j codes are in codebook order,
                             which is what a codebook reordering would fix

Usage:
    python codebook_geometry.py --llamagen-root ~/LlamaGen \
        --gpt-ckpt ~/LlamaGen/pretrained_models/c2i_B_256.pt \
        --vq-ckpt ~/LlamaGen/pretrained_models/vq_ds16_c2i.pt
"""
import argparse
import json
import sys
from pathlib import Path

import torch


NEIGHBOURHOODS = [1, 2, 4, 8, 16, 32, 64]


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
    parser.add_argument("--top-k", type=int, default=2000)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, required=True)
    return parser.parse_args()


def codebook_neighbours(embedding: torch.Tensor, max_j: int, chunk: int = 1024):
    """Return (V, max_j) indices of the nearest codes to each code, self first."""
    v = embedding.shape[0]
    idx_out = torch.empty((v, max_j), dtype=torch.long, device=embedding.device)
    dist_out = torch.empty((v, max_j), dtype=torch.float32, device=embedding.device)
    for start in range(0, v, chunk):
        stop = min(start + chunk, v)
        d = torch.cdist(embedding[start:stop], embedding)  # (chunk, V)
        vals, idx = torch.topk(d, max_j, dim=1, largest=False)
        idx_out[start:stop] = idx
        dist_out[start:stop] = vals
    return idx_out, dist_out


@torch.no_grad()
def collect_distributions(model, sample_fn, cond, seq_len, cfg_scale, **sampling_kwargs):
    """Run the c2i loop, returning sampled tokens and the sampling distributions."""
    if cfg_scale > 1.0:
        cond_null = torch.ones_like(cond) * model.num_classes
        cond_combined = torch.cat([cond, cond_null])
    else:
        cond_combined = cond
    t_cls = 1
    device = cond.device
    with torch.device(device):
        model.setup_caches(
            max_batch_size=cond.shape[0] * (2 if cfg_scale > 1.0 else 1),
            max_seq_length=t_cls + seq_len,
            dtype=model.tok_embeddings.weight.dtype,
        )

    def combine(logits):
        if cfg_scale > 1.0:
            cond_logits, uncond_logits = torch.split(logits, len(logits) // 2, dim=0)
            return uncond_logits + (cond_logits - uncond_logits) * cfg_scale
        return logits

    tokens, dists = [], []
    input_pos = torch.arange(0, t_cls, device=device)
    logits, _ = model(None, cond_combined, input_pos)
    next_token, probs = sample_fn(combine(logits), **sampling_kwargs)
    tokens.append(next_token.squeeze(1))
    dists.append(probs.float())

    cur_token = next_token.view(-1, 1)
    input_pos = torch.tensor([t_cls], device=device, dtype=torch.int)
    for _ in range(seq_len - 1):
        if cfg_scale > 1.0:
            logits, _ = model(torch.cat([cur_token, cur_token]), cond_idx=None, input_pos=input_pos)
        else:
            logits, _ = model(cur_token, cond_idx=None, input_pos=input_pos)
        next_token, probs = sample_fn(combine(logits), **sampling_kwargs)
        tokens.append(next_token.squeeze(1))
        dists.append(probs.float())
        input_pos += 1
        cur_token = next_token.view(-1, 1)

    return torch.stack(tokens, dim=1), torch.stack(dists, dim=1)


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
    print(f"codebook {tuple(embedding.shape)}, l2_norm={getattr(vq_model.quantize, 'l2_norm', False)}")

    max_j = max(NEIGHBOURHOODS)
    nn_idx, nn_dist = codebook_neighbours(embedding, max_j)
    # Distance to the nearest distinct code, as the natural scale for "close".
    print(f"nearest-neighbour distance: mean={nn_dist[:, 1].mean():.4f} "
          f"median={nn_dist[:, 1].median():.4f}")
    print(f"distance to {max_j}th neighbour: mean={nn_dist[:, max_j - 1].mean():.4f}")

    latent_size = args.image_size // args.downsample_size
    seq_len = latent_size ** 2
    model = GPT_models[args.gpt_model](
        vocab_size=args.codebook_size,
        block_size=seq_len,
        num_classes=args.num_classes,
        cls_token_num=args.cls_token_num,
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
    tokens, dists = collect_distributions(
        model, sample_fn, cond, seq_len,
        cfg_scale=args.cfg_scale,
        temperature=args.temperature, top_k=args.top_k, top_p=args.top_p,
    )
    flat_tokens = tokens.reshape(-1).long()
    flat_dists = dists.reshape(-1, dists.shape[-1])
    print(f"collected {flat_tokens.numel()} positions")

    exact = flat_dists.gather(1, flat_tokens[:, None]).squeeze(1)
    result = {
        "config": {
            "cfg_scale": args.cfg_scale, "top_k": args.top_k, "top_p": args.top_p,
            "temperature": args.temperature, "batch_size": args.batch_size,
            "gpt_model": args.gpt_model, "positions": int(flat_tokens.numel()),
        },
        "codebook": {
            "nn_dist_mean": nn_dist[:, 1].mean().item(),
            "nn_dist_median": nn_dist[:, 1].median().item(),
            f"dist_to_{max_j}th_mean": nn_dist[:, max_j - 1].mean().item(),
        },
        "neighbourhoods": [],
    }

    for j in NEIGHBOURHOODS:
        nbrs = nn_idx[flat_tokens][:, :j]                       # (P, j)
        mass = flat_dists.gather(1, nbrs).sum(dim=1)            # relaxed bin width
        # How far apart are these codes in raw index order? A reordering that made
        # them adjacent would turn the relaxed bin into one contiguous interval.
        spread = (nbrs.max(dim=1).values - nbrs.min(dim=1).values).float()
        result["neighbourhoods"].append({
            "j": j,
            "relaxed_width_mean": mass.mean().item(),
            "relaxed_width_median": mass.median().item(),
            "gain_over_exact_median": (mass.median() / exact.median()).item(),
            "frac_above_0.1": (mass > 0.1).float().mean().item(),
            "frac_above_0.5": (mass > 0.5).float().mean().item(),
            "index_spread_median": spread.median().item(),
            "embed_dist_mean": nn_dist[flat_tokens][:, :j].mean().item(),
        })
        r = result["neighbourhoods"][-1]
        print(f"j={j:3d} | relaxed width mean={r['relaxed_width_mean']:.4f} "
              f"median={r['relaxed_width_median']:.4f} "
              f"(x{r['gain_over_exact_median']:.1f}) | "
              f">0.1: {r['frac_above_0.1']:.3f} | "
              f"index spread median={r['index_spread_median']:.0f}")

    result["exact_width_mean"] = exact.mean().item()
    result["exact_width_median"] = exact.median().item()
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
