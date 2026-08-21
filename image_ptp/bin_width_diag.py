"""Measure teacher bin widths for an image AR model.

For O-PTP, the auxiliary u_k for token t_k must land in
    [F_{k,t_k-1}, F_{k,t_k}),
whose width equals the teacher probability of the sampled token. Narrow bins
make the u -> token map hard to fit and leave little margin for acceptance
during speculative decoding, so the width distribution bounds how well PTP can
work on this model.

Widths are measured on the distribution actually used for sampling, i.e. after
CFG, temperature, top-k and top-p.

Usage:
    python bin_width_diag.py --llamagen-root ~/LlamaGen \
        --gpt-ckpt ~/LlamaGen/pretrained_models/c2i_B_256.pt
"""
import argparse
import json
import sys
from pathlib import Path

import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--llamagen-root", type=str, required=True)
    parser.add_argument("--gpt-model", type=str, default="GPT-B")
    parser.add_argument("--gpt-ckpt", type=str, required=True)
    parser.add_argument("--precision", type=str, default="bf16", choices=["none", "fp16", "bf16"])
    parser.add_argument("--codebook-size", type=int, default=16384)
    parser.add_argument("--image-size", type=int, default=256, choices=[256, 384, 512])
    parser.add_argument("--downsample-size", type=int, default=16)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cls-token-num", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16, help="images per sweep config")
    parser.add_argument("--uncond", action="store_true",
                        help="condition on the null class, i.e. sample unconditionally")
    parser.add_argument("--configs", type=str, default=None,
                        help="override the sweep, as cfg:top_k:top_p:temp entries "
                             "separated by commas, e.g. 1.0:100:1.0:1.0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="bin_width_stats.json")
    return parser.parse_args()


# Sweep grid: (cfg_scale, top_k, top_p, temperature).
# top_k=0 and top_p=1.0 mean no truncation.
SWEEP = [
    (1.0, 0, 1.0, 1.0),
    (2.0, 0, 1.0, 1.0),
    (4.0, 0, 1.0, 1.0),
    (4.0, 2000, 1.0, 1.0),
    (2.0, 2000, 1.0, 1.0),
    (4.0, 100, 1.0, 1.0),
    (4.0, 2000, 1.0, 0.9),
    (4.0, 2000, 0.9, 1.0),
]


def parse_configs(spec):
    out = []
    for chunk in spec.split(","):
        cfg, top_k, top_p, temp = chunk.split(":")
        out.append((float(cfg), int(top_k), float(top_p), float(temp)))
    return out


@torch.no_grad()
def generate_with_stats(model, sample_fn, cond, max_new_tokens, cfg_scale, cfg_interval,
                        **sampling_kwargs):
    """Mirror LlamaGen's c2i generate loop, keeping the sampling distribution.

    Returns (bin_widths, eff_candidates, support_sizes), each (B, max_new_tokens).
    """
    if cfg_scale > 1.0:
        cond_null = torch.ones_like(cond) * model.num_classes
        cond_combined = torch.cat([cond, cond_null])
    else:
        cond_combined = cond
    T = 1

    max_batch_size = cond.shape[0]
    device = cond.device
    with torch.device(device):
        max_batch_size_cfg = max_batch_size * 2 if cfg_scale > 1.0 else max_batch_size
        model.setup_caches(max_batch_size=max_batch_size_cfg,
                           max_seq_length=T + max_new_tokens,
                           dtype=model.tok_embeddings.weight.dtype)

    widths, eff_cands, supports = [], [], []

    def record(idx, probs):
        probs = probs.float()
        widths.append(probs.gather(1, idx).squeeze(1))
        eff_cands.append(1.0 / probs.pow(2).sum(dim=1))
        supports.append((probs > 0).sum(dim=1).float())

    def combine(logits):
        if cfg_scale > 1.0:
            cond_logits, uncond_logits = torch.split(logits, len(logits) // 2, dim=0)
            return uncond_logits + (cond_logits - uncond_logits) * cfg_scale
        return logits

    # Prefill on the class token.
    input_pos = torch.arange(0, T, device=device)
    logits, _ = model(None, cond_combined, input_pos)
    next_token, probs = sample_fn(combine(logits), **sampling_kwargs)
    record(next_token, probs)

    # Decode the rest of the raster sequence.
    cur_token = next_token.view(-1, 1)
    input_pos = torch.tensor([T], device=device, dtype=torch.int)
    cfg_flag = True
    for i in range(max_new_tokens - 1):
        if cfg_interval > -1 and i > cfg_interval:
            cfg_flag = False
        if cfg_scale > 1.0:
            x_combined = torch.cat([cur_token, cur_token])
            logits, _ = model(x_combined, cond_idx=None, input_pos=input_pos)
            cond_logits, uncond_logits = torch.split(logits, len(logits) // 2, dim=0)
            logits = uncond_logits + (cond_logits - uncond_logits) * cfg_scale if cfg_flag \
                else cond_logits
        else:
            logits, _ = model(cur_token, cond_idx=None, input_pos=input_pos)
        next_token, probs = sample_fn(logits, **sampling_kwargs)
        record(next_token, probs)
        input_pos += 1
        cur_token = next_token.view(-1, 1)

    return (torch.stack(widths, dim=1),
            torch.stack(eff_cands, dim=1),
            torch.stack(supports, dim=1))


def summarize(widths, eff_cands, supports):
    w = widths.flatten()
    quantiles = torch.tensor([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99], device=w.device)
    # Split raster order in half to see whether later tokens get easier.
    half = widths.shape[1] // 2
    return {
        "bin_width_mean": w.mean().item(),
        "bin_width_quantiles": dict(zip(
            [f"p{round(q * 100)}" for q in quantiles.tolist()],
            torch.quantile(w, quantiles).tolist(),
        )),
        "frac_width_below_1e-2": (w < 1e-2).float().mean().item(),
        "frac_width_below_1e-3": (w < 1e-3).float().mean().item(),
        "frac_width_above_0.5": (w > 0.5).float().mean().item(),
        "eff_candidates_mean": eff_cands.mean().item(),
        "eff_candidates_median": eff_cands.flatten().median().item(),
        "support_mean": supports.mean().item(),
        "bin_width_mean_first_half": widths[:, :half].mean().item(),
        "bin_width_mean_second_half": widths[:, half:].mean().item(),
    }


def main():
    args = parse_args()
    root = Path(args.llamagen_root).expanduser()
    sys.path.insert(0, str(root))
    from autoregressive.models.gpt import GPT_models
    from autoregressive.models.generate import sample as sample_fn

    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    precision = {"none": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[args.precision]

    latent_size = args.image_size // args.downsample_size
    seq_len = latent_size ** 2
    model = GPT_models[args.gpt_model](
        vocab_size=args.codebook_size,
        block_size=seq_len,
        num_classes=args.num_classes,
        cls_token_num=args.cls_token_num,
        model_type="c2i",
    ).to(device=device, dtype=precision)

    checkpoint = torch.load(args.gpt_ckpt, map_location="cpu")
    for key in ("model", "module", "state_dict"):
        if key in checkpoint:
            checkpoint = checkpoint[key]
            break
    model.load_state_dict(checkpoint, strict=False)
    model.eval()
    del checkpoint
    print(f"loaded {args.gpt_model} from {args.gpt_ckpt}, seq_len={seq_len}")

    sweep = parse_configs(args.configs) if args.configs else SWEEP
    if args.uncond:
        assert all(c[0] == 1.0 for c in sweep), \
            "unconditional sampling is only meaningful without guidance (cfg 1.0)"

    results = []
    for cfg_scale, top_k, top_p, temperature in sweep:
        torch.manual_seed(args.seed)
        if args.uncond:
            # The null class is the token LlamaGen uses for the unguided branch.
            cond = torch.full((args.batch_size,), args.num_classes, device=device)
        else:
            cond = torch.randint(0, args.num_classes, (args.batch_size,), device=device)
        widths, eff_cands, supports = generate_with_stats(
            model, sample_fn, cond, seq_len,
            cfg_scale=cfg_scale, cfg_interval=-1,
            temperature=temperature, top_k=top_k, top_p=top_p,
        )
        entry = {
            "uncond": args.uncond,
            "cfg_scale": cfg_scale, "top_k": top_k, "top_p": top_p,
            "temperature": temperature,
            **summarize(widths, eff_cands, supports),
        }
        results.append(entry)
        print(f"{'uncond' if args.uncond else 'cond'} "
              f"cfg={cfg_scale} top_k={top_k} top_p={top_p} temp={temperature} | "
              f"width mean={entry['bin_width_mean']:.4f} "
              f"median={entry['bin_width_quantiles']['p50']:.4f} | "
              f"eff_cand mean={entry['eff_candidates_mean']:.1f} | "
              f"frac<1e-2={entry['frac_width_below_1e-2']:.3f}")

    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
