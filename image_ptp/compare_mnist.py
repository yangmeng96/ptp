"""Why did O-PTP work on MNIST but not on LlamaGen?

Two candidate explanations, and they call for different responses:

  bin width   MNIST has a 512-code book on an easy dataset, so the teacher is
              confident and the intervals are wide, the way they are for text.
              LlamaGen's are ~350x narrower.

  the bar     the MNIST run was scored by FID and Inception score on the
              student's own samples. A student that ignores u entirely still
              draws decent digits, so that score never tested the mechanism.
              The LlamaGen run was scored by exact agreement with the teacher,
              which is what speculative decoding actually requires.

This measures both on the existing MNIST checkpoints: the teacher's bin widths,
and whether the trained student's predictions depend on u at all.

Run from inside the ptp-vqvae checkout.
"""
import argparse
import sys

import torch
import torch.nn.functional as F


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", type=str, default="/home/mengy13/ptp-vqvae")
    p.add_argument("--student-ckpt", type=str, default="checkpoints/ptp_o_mnist_raster_row.pt")
    p.add_argument("--images", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    sys.path.insert(0, args.repo)
    import os
    os.chdir(args.repo)

    from models.ar import ARTransformer, codes_flat_to_seq
    from models.ptp import PTPStudent
    from utils.helper import load_vqvae
    from torchvision import datasets

    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ck = torch.load(args.student_ckpt, map_location=device, weights_only=False)
    num_codes, block_len = ck["num_codes"], ck["block_len"]
    perm = ck["perm"].to(device)
    bos = num_codes

    vqvae, tfm = load_vqvae("mnist", device)
    t_ck = torch.load(ck["teacher_ckpt"], map_location=device, weights_only=False)
    teacher = ARTransformer(
        num_codes=num_codes, d_model=t_ck["d_model"], nhead=t_ck["nhead"],
        num_layers=t_ck["num_layers"], dim_feedforward=t_ck["dim_feedforward"],
        max_seq_len=t_ck["max_seq_len"],
    ).to(device).eval()
    teacher.load_state_dict(t_ck["ar"])

    student = PTPStudent(
        num_codes=num_codes, d_model=ck["d_model"], nhead=ck["nhead"],
        num_layers=ck["num_layers"], dim_feedforward=ck["dim_feedforward"],
        max_seq_len=ck["max_seq_len"],
    ).to(device).eval()
    student.load_state_dict(ck["student"])
    print(f"MNIST: {num_codes} codes, block_len={block_len}, epoch={ck['epoch']}")

    ds = datasets.MNIST(root="data/mnist", train=False, download=False, transform=tfm)
    xs = torch.stack([ds[i][0] for i in range(args.images)]).to(device)
    codes = vqvae.encode(xs)
    seq = codes_flat_to_seq(codes, perm)
    B, T = seq.shape

    # Teacher distribution over the whole sequence, exactly as training does it.
    inp = torch.cat([torch.full((B, 1), bos, dtype=torch.long, device=device),
                     seq[:, :-1]], dim=1)
    logits = teacher(inp)[:, :, :num_codes]
    probs = F.softmax(logits, dim=-1)
    cdf = probs.cumsum(-1)
    idx = seq.unsqueeze(-1)
    right = cdf.gather(2, idx).squeeze(2)
    width = probs.gather(2, idx).squeeze(2)
    left = right - width
    eff = 1.0 / probs.pow(2).sum(-1)

    print("\n--- teacher bin widths (MNIST) ---")
    print(f"median={width.median():.4f}  mean={width.mean():.4f}  "
          f"eff candidates median={eff.median():.2f}")
    print(f"share > 0.5: {(width > 0.5).float().mean():.3f}   "
          f"share < 0.01: {(width < 0.01).float().mean():.3f}")
    print("for comparison: text 0.837 / 1.33 candidates, LlamaGen 0.0024 / ~300")

    # Does the trained student actually use u?
    starts = list(range(0, T - block_len + 1, block_len))
    hits = shuffled_hits = total = 0
    for start in starts:
        stop = min(start + block_len - 1, T - 1)
        span = stop - start + 1
        u = left[:, start:stop + 1] + width[:, start:stop + 1] * torch.rand(
            B, span, device=device)
        prefix = torch.cat([torch.full((B, 1), bos, dtype=torch.long, device=device),
                            seq[:, :start]], dim=1)
        truth = seq[:, start:stop + 1]
        for shuffled in (False, True):
            uu = u[torch.randperm(B, device=device)] if shuffled else u
            pred = student(prefix, uu)[:, start + 1:start + 1 + span, :num_codes].argmax(-1)
            if shuffled:
                shuffled_hits += int((pred == truth).sum())
            else:
                hits += int((pred == truth).sum())
        total += truth.numel()

    print("\n--- does the student use u? ---")
    print(f"accuracy with the real u:     {hits / total:.4f}")
    print(f"accuracy with u shuffled:     {shuffled_hits / total:.4f}")
    print(f"lift:                         {hits / max(shuffled_hits, 1):.2f}x")
    print("LlamaGen student for comparison: 0.0583 vs 0.0560, lift 1.04x")


if __name__ == "__main__":
    main()
