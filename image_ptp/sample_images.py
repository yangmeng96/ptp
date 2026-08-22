"""Sample images from an AR teacher and from O-PTP students, side by side.

No speculative decoding: the students generate on their own, a block at a time,
drawing u ~ U[0,1] and taking the argmax. That shows what the student actually
learned rather than what a verifier can rescue -- if it has not learned to read
u, the samples are what a marginal predictor produces.

The teacher samples one token at a time from its own softmax, which is the
process the students are distilled to reproduce.

Usage:
    PYTHONPATH=src:. python image_ptp/sample_images.py \
        --ar-ckpt ~/ptp-vqvae/checkpoints/ar_mnist_raster.pt \
        --students full=~/ptp-image-exp/mnist-full/last.ckpt \
                   lora=~/ptp-image-exp/mnist-lora/last.ckpt \
        --out-dir ~/ptp-image-results/samples
"""
import argparse
import os
import sys
from pathlib import Path

import torch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", type=str, default="/home/mengy13/ptp-vqvae")
    p.add_argument("--ar-ckpt", type=str, required=True)
    p.add_argument("--students", type=str, nargs="*", default=[],
                   help="name=path entries; the checkpoint tells us LoRA from full")
    p.add_argument("--lora-rank", type=int, default=32)
    p.add_argument("--num-layers", type=int, default=None,
                   help="student depth, when it differs from the teacher's")
    p.add_argument("--num-images", type=int, default=10)
    p.add_argument("--block-len", type=int, default=7)
    p.add_argument("--temperature", type=float, default=1.0,
                   help="teacher sampling temperature")
    p.add_argument("--student-temperature", type=float, default=0.0,
                   help="0 = argmax, which is what O-PTP prescribes")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", type=str, required=True)
    return p.parse_args()


@torch.no_grad()
def sample_teacher(model, batch, seq_len, bos, temperature, device):
    """Plain autoregressive sampling, one token at a time."""
    seq = torch.full((batch, 1), bos, dtype=torch.long, device=device)
    for _ in range(seq_len):
        logits = model(input_ids=seq).logits[:, -1, :] / max(temperature, 1e-5)
        probs = torch.softmax(logits.float(), dim=-1)
        probs[:, bos] = 0  # never emit BOS mid-sequence
        seq = torch.cat([seq, torch.multinomial(probs, 1)], dim=1)
    return seq[:, 1:]


@torch.no_grad()
def sample_ptp(mixed, batch, seq_len, bos, block_len, device, temperature=0.0):
    """Block-wise O-PTP sampling: draw u, run one forward, read off the block.

    `MixedTransformerModel.forward` runs the prefix without adapters to build the
    cache and the auxiliary block with them, and derives the shifted auxiliary
    positions itself -- the same path training uses.

    `temperature=0` takes the argmax, which is what O-PTP prescribes: the output
    is meant to be one-hot and u is meant to carry all the randomness. Positive
    temperatures sample from the logits instead, as `ptp-vqvae`'s sampler does by
    default. That adds randomness the formulation does not have, so it can make
    a student that reads u poorly still look diverse.
    """
    seq = torch.full((batch, 1), bos, dtype=torch.long, device=device)
    while seq.shape[1] - 1 < seq_len:
        span = min(block_len, seq_len - (seq.shape[1] - 1))
        u = torch.rand(batch, span, device=device)
        with mixed.enable_adapters(enabled=True):
            _, completion = mixed(input_ids=seq, auxiliaries=u)
        logits = completion.logits[:, :span, :].float()
        if temperature > 0:
            probs = torch.softmax(logits / temperature, dim=-1)
            proposed = torch.multinomial(probs.reshape(-1, probs.shape[-1]), 1)
            proposed = proposed.view(batch, span)
        else:
            proposed = logits.argmax(dim=-1)
        proposed = torch.where(proposed == bos, torch.zeros_like(proposed), proposed)
        seq = torch.cat([seq, proposed], dim=1)
    return seq[:, 1:]


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    sys.path.insert(0, args.repo)
    repo_cwd = os.getcwd()
    os.chdir(args.repo)
    from models.ar import seq_to_codes_grid
    from utils.helper import load_vqvae
    from torchvision.utils import save_image

    # load_vqvae reads its weights by a path relative to the repo, so stay here
    # until it has them.
    vqvae, _ = load_vqvae("mnist", device)
    os.chdir(repo_cwd)

    from image_ptp.vqvae_ar_hf import build, build_module
    from ptp.transformer import MixedTransformerModel

    teacher, meta = build(args.ar_ckpt, device=device, dtype=torch.float32)
    h, w, bos = meta["h"], meta["w"], meta["num_codes"]
    seq_len = h * w
    inv = torch.argsort(meta["perm"]).to(device)
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    def save(seq, name):
        grid = seq_to_codes_grid(seq, inv, h, w)
        images = vqvae.decode(grid).float()
        path = out_dir / f"{name}.png"
        save_image(images, path, nrow=5, normalize=True, value_range=(-1, 1))
        print(f"  wrote {path}")
        return path

    torch.manual_seed(args.seed)
    save(sample_teacher(teacher, args.num_images, seq_len, bos,
                        args.temperature, device), "teacher")

    targets = ["q_proj", "k_proj", "v_proj", "o_proj", "linear1", "linear2"]
    for entry in args.students:
        name, _, path = entry.partition("=")
        state = torch.load(Path(path).expanduser(), map_location="cpu",
                           weights_only=False)["state_dict"]
        has_lora = any("lora_" in k for k in state)
        inner = build_module(args.ar_ckpt, device=device, dtype=torch.float32,
                             num_layers=args.num_layers)
        mixed = MixedTransformerModel(
            model_id=inner, dtype=torch.float32,
            lora_config={"r": args.lora_rank, "target_modules": targets} if has_lora else None,
            attn_implementation="flex_attention",
        ).to(device).eval()
        # The checkpoint is keyed for the LightningModule, one level up.
        stripped = {k[len("model."):]: v for k, v in state.items() if k.startswith("model.")}
        missing, unexpected = mixed.load_state_dict(stripped, strict=False)
        print(f"{name}: {'LoRA' if has_lora else 'full'}, "
              f"{len(missing)} missing, {len(unexpected)} unexpected")
        for temp, suffix in ((0.0, "argmax"), (1.0, "temp1")):
            torch.manual_seed(args.seed)
            save(sample_ptp(mixed, args.num_images, seq_len, bos, args.block_len,
                            device, temperature=temp), f"ptp_{name}_{suffix}")
        del mixed, inner
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
