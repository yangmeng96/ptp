"""Locate where a trained O-PTP student fails.

Block-level acceptance conflates two very different failures. The student may be
unable to invert the CDF from u at all, or it may invert it fine and then lose
the thread because later positions in the block have to infer the realised
prefix from earlier auxiliaries. These have opposite implications, so:

  per position   accuracy at each offset in the block. Offset 0 is the easy
                 case -- the prefix is known exactly and only one inversion is
                 needed. If offset 0 is already poor, u inversion is the problem.
  shuffled u     the same evaluation with auxiliaries taken from a different
                 sequence. If accuracy barely moves, the student is ignoring u
                 and has only learned the teacher's marginal.

Usage:
    python diagnose_student.py --llamagen-root ~/LlamaGen \
        --gpt-ckpt ~/LlamaGen/pretrained_models/c2i_B_256.pt \
        --data ~/ptp-image-results/pregen_cfg4_k100.pt \
        --student ~/ptp-image-results/run_cfg4_k100_r128/student.pt
"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from llamagen_ptp import LlamaGenPTP, load_llamagen, sample_auxiliaries  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--llamagen-root", type=str, required=True)
    p.add_argument("--gpt-model", type=str, default="GPT-B")
    p.add_argument("--gpt-ckpt", type=str, required=True)
    p.add_argument("--data", type=str, required=True)
    p.add_argument("--student", type=str, required=True)
    p.add_argument("--images", type=int, default=256)
    p.add_argument("--starts", type=int, default=4)
    p.add_argument("--chunk", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default=None)
    return p.parse_args()


@torch.no_grad()
def accuracy_by_offset(model, data, idx, starts, block_len, device, shuffle_u=False):
    hits = torch.zeros(block_len)
    count = 0
    for start in starts:
        stop = start + block_len - 1
        for chunk in torch.split(idx, 32):
            tok = data["tokens"][chunk].long().to(device)
            lab = data["labels"][chunk].long().to(device)
            left = data["left_bin_edges"][chunk][:, start:stop + 1].to(device)
            right = data["right_bin_edges"][chunk][:, start:stop + 1].to(device)
            u = sample_auxiliaries(left, right, uniform=True)
            if shuffle_u:
                # Auxiliaries from other sequences: same marginal, wrong content.
                u = u[torch.randperm(u.shape[0], device=device)]
            proposed = model.student_logits(lab, tok, u, start).argmax(dim=-1)
            hits += (proposed == tok[:, start:stop + 1]).float().sum(dim=0).cpu()
            count += tok.shape[0]
    return hits / count


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    root = Path(args.llamagen_root).expanduser()

    ckpt = torch.load(Path(args.student).expanduser(), map_location="cpu")
    train_args = ckpt["args"]
    block_len = train_args["block_len"]

    data = torch.load(Path(args.data).expanduser(), map_location="cpu")
    seq_len = data["config"]["seq_len"]

    gpt, _ = load_llamagen(root, args.gpt_model, args.gpt_ckpt,
                           dtype=torch.float32, device=device)
    for block in gpt.layers:
        block.attention.kv_cache = None
    model = LlamaGenPTP(gpt, lora_rank=train_args["lora_rank"]).to(device).eval()
    missing, unexpected = model.load_state_dict(ckpt["lora"], strict=False)
    assert not unexpected, f"unexpected keys in checkpoint: {unexpected[:3]}"
    model.u_embed.load_state_dict(ckpt["u_embed"])
    print(f"loaded student, block_len={block_len}, lora_rank={train_args['lora_rank']}")

    generator = torch.Generator().manual_seed(train_args["seed"])
    perm = torch.randperm(data["tokens"].shape[0], generator=generator)
    holdout = min(train_args["holdout"], data["tokens"].shape[0] // 4)
    idx = perm[:holdout][:args.images]
    starts = torch.linspace(0, seq_len - block_len, args.starts).long().tolist()
    print(f"evaluating on {len(idx)} held-out sequences at starts {starts}")

    real = accuracy_by_offset(model, data, idx, starts, block_len, device)
    shuffled = accuracy_by_offset(model, data, idx, starts, block_len, device, shuffle_u=True)

    print("\noffset  accuracy  shuffled-u  lift")
    for k in range(block_len):
        print(f"{k:5d}   {real[k]:.4f}    {shuffled[k]:.4f}     {real[k] / max(shuffled[k], 1e-9):.2f}x")
    print(f"\nmean    {real.mean():.4f}    {shuffled.mean():.4f}     "
          f"{real.mean() / max(shuffled.mean(), 1e-9):.2f}x")
    print(f"offset 0 is the easy case: prefix known exactly, one inversion needed")

    if args.out:
        Path(args.out).expanduser().write_text(json.dumps({
            "accuracy_by_offset": real.tolist(),
            "shuffled_u_by_offset": shuffled.tolist(),
        }, indent=2))


if __name__ == "__main__":
    main()
