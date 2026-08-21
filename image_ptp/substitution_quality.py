"""What does relaxed acceptance actually cost in image quality?

A relaxed O-PTP acceptance test would let the student's proposal stand whenever
it decodes to something close enough to what the base model picked. This script
prices that: it generates images, swaps every token for a neighbouring code, and
measures how far the decoded image moves.

Codebook-space neighbours are compared against random codes as a control. If
the two degrade quality equally, latent distance carries no perceptual
information and a relaxed test would have to be defined some other way.

Usage:
    python substitution_quality.py --llamagen-root ~/LlamaGen \
        --gpt-ckpt ~/LlamaGen/pretrained_models/c2i_B_256.pt \
        --vq-ckpt ~/LlamaGen/pretrained_models/vq_ds16_c2i.pt \
        --out-dir ~/ptp-image-results/substitution
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
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--top-k", type=int, default=2000)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--swap-frac", type=float, default=1.0,
                        help="fraction of positions to substitute")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=str, required=True)
    return parser.parse_args()


def codebook_neighbours(embedding, max_j, chunk=1024):
    v = embedding.shape[0]
    idx_out = torch.empty((v, max_j), dtype=torch.long, device=embedding.device)
    for start in range(0, v, chunk):
        stop = min(start + chunk, v)
        d = torch.cdist(embedding[start:stop], embedding)
        idx_out[start:stop] = torch.topk(d, max_j, dim=1, largest=False).indices
    return idx_out


def psnr(a, b):
    """Both in [-1, 1], shape (B, C, H, W). Returns per-image PSNR in dB."""
    mse = ((a - b) / 2.0).pow(2).flatten(1).mean(dim=1)
    return -10.0 * torch.log10(mse.clamp(min=1e-12))


def main():
    args = parse_args()
    root = Path(args.llamagen_root).expanduser()
    sys.path.insert(0, str(root))
    from autoregressive.models.gpt import GPT_models
    from autoregressive.models.generate import generate
    from tokenizer.tokenizer_image.vq_model import VQ_models
    from torchvision.utils import save_image

    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

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
    nn_idx = codebook_neighbours(embedding, max(NEIGHBOURHOODS))

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
    tokens = generate(
        model, cond, seq_len,
        cfg_scale=args.cfg_scale, cfg_interval=-1,
        temperature=args.temperature, top_k=args.top_k, top_p=args.top_p,
    ).long()
    qzshape = [args.batch_size, args.codebook_embed_dim, latent_size, latent_size]
    reference = vq_model.decode_code(tokens, qzshape).float()
    save_image(reference, out_dir / "reference.png", nrow=4,
               normalize=True, value_range=(-1, 1))

    swap_mask = torch.rand(tokens.shape, device=device) < args.swap_frac
    results = []
    for j in NEIGHBOURHOODS:
        row = {"j": j}
        for mode in ("codebook", "random"):
            if mode == "codebook":
                if j == 1:
                    swapped = tokens.clone()  # self only, no change
                else:
                    pick = torch.randint(1, j, tokens.shape, device=device)
                    swapped = nn_idx[tokens].gather(2, pick[..., None]).squeeze(2)
            else:
                swapped = torch.randint(0, args.codebook_size, tokens.shape, device=device)
            swapped = torch.where(swap_mask, swapped, tokens)
            image = vq_model.decode_code(swapped, qzshape).float()
            row[f"{mode}_psnr"] = psnr(image, reference).mean().item()
            row[f"{mode}_l1"] = (image - reference).abs().mean().item()
            if mode == "codebook":
                save_image(image, out_dir / f"swap_codebook_j{j}.png", nrow=4,
                           normalize=True, value_range=(-1, 1))
            elif j == NEIGHBOURHOODS[-1]:
                save_image(image, out_dir / "swap_random.png", nrow=4,
                           normalize=True, value_range=(-1, 1))
        results.append(row)
        print(f"j={j:3d} | codebook PSNR={row['codebook_psnr']:6.2f} dB  "
              f"L1={row['codebook_l1']:.4f} | "
              f"random PSNR={row['random_psnr']:6.2f} dB  L1={row['random_l1']:.4f}")

    payload = {
        "config": vars(args),
        "results": results,
        "note": "j=1 codebook is the identity swap and should be infinite PSNR",
    }
    (out_dir / "substitution_stats.json").write_text(json.dumps(payload, indent=2, default=str))
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
