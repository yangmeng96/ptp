"""Sample training sequences from the teacher and cache their bin edges.

The PTP paper finds teacher samples the lowest-variance source of training data,
and here there is no alternative anyway -- no ImageNet on this machine. Since the
teacher is frozen and the sequences are fixed, the intervals [F_{k,t_k-1},
F_{k,t_k}) are constants: computing them once removes two teacher forwards
(guidance doubles them) from every training step, for 17MB of storage.

The guidance scale, truncation and temperature are baked in here. Changing any
of them means regenerating, because they change which distribution `u` inverts.

Usage:
    python pregenerate.py --llamagen-root ~/LlamaGen \
        --gpt-ckpt ~/LlamaGen/pretrained_models/c2i_B_256.pt \
        --num-images 8192 --out ~/ptp-image-results/pregen_cfg4_k100.pt
"""
import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from llamagen_ptp import LlamaGenPTP, load_llamagen  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--llamagen-root", type=str, required=True)
    p.add_argument("--gpt-model", type=str, default="GPT-B")
    p.add_argument("--gpt-ckpt", type=str, required=True)
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--num-classes", type=int, default=1000)
    p.add_argument("--num-images", type=int, default=8192)
    p.add_argument("--gen-batch", type=int, default=32)
    p.add_argument("--edge-batch", type=int, default=8, help="smaller: the logits are (B, S, V)")
    p.add_argument("--cfg-scale", type=float, default=4.0)
    p.add_argument("--top-k", type=int, default=100)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, required=True)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    root = Path(args.llamagen_root).expanduser()

    gpt, seq_len = load_llamagen(root, args.gpt_model, args.gpt_ckpt,
                                 image_size=args.image_size,
                                 num_classes=args.num_classes, device=device)
    from autoregressive.models.generate import generate
    # lora_rank is irrelevant here; only the frozen base is used.
    model = LlamaGenPTP(gpt, lora_rank=4).to(device).eval()

    all_tokens, all_labels = [], []
    started = time.time()
    done = 0
    while done < args.num_images:
        batch = min(args.gen_batch, args.num_images - done)
        labels = torch.randint(0, args.num_classes, (batch,), device=device)
        tokens = generate(
            model.base, labels, seq_len,
            cfg_scale=args.cfg_scale, cfg_interval=-1,
            temperature=args.temperature, top_k=args.top_k, top_p=args.top_p,
        ).long()
        all_tokens.append(tokens.cpu())
        all_labels.append(labels.cpu())
        done += batch
        if done % (args.gen_batch * 10) == 0 or done == args.num_images:
            rate = done / (time.time() - started)
            eta = (args.num_images - done) / max(rate, 1e-6)
            print(f"generated {done}/{args.num_images}  {rate:.1f} img/s  eta {eta / 60:.1f} min",
                  flush=True)

    tokens = torch.cat(all_tokens)
    labels = torch.cat(all_labels)

    # generate() leaves preallocated caches behind; the edge pass runs cache-free.
    for block in model.base.layers:
        block.attention.kv_cache = None

    lefts, rights = [], []
    started = time.time()
    for start in range(0, tokens.shape[0], args.edge_batch):
        stop = min(start + args.edge_batch, tokens.shape[0])
        left, right = model.bin_edges(
            labels[start:stop].to(device), tokens[start:stop].to(device),
            0, seq_len - 1,
            cfg_scale=args.cfg_scale, top_k=args.top_k,
            top_p=args.top_p, temperature=args.temperature,
        )
        lefts.append(left.cpu())
        rights.append(right.cpu())
        if (stop // args.edge_batch) % 50 == 0 or stop == tokens.shape[0]:
            rate = stop / (time.time() - started)
            print(f"bin edges {stop}/{tokens.shape[0]}  {rate:.1f} img/s", flush=True)

    left = torch.cat(lefts)
    right = torch.cat(rights)
    width = right - left
    print(f"bin width: mean={width.mean():.4f} median={width.median():.4f} "
          f"min={width.min():.2e}")
    assert (width >= -1e-6).all(), "negative bin width means the edges are misaligned"

    payload = {
        "tokens": tokens.to(torch.int32),
        "labels": labels.to(torch.int32),
        "left_bin_edges": left,
        "right_bin_edges": right,
        "config": {
            "cfg_scale": args.cfg_scale, "top_k": args.top_k, "top_p": args.top_p,
            "temperature": args.temperature, "gpt_model": args.gpt_model,
            "seq_len": seq_len, "num_images": int(tokens.shape[0]),
        },
    }
    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
