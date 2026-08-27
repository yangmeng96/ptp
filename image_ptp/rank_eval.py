"""How far down the student's ranking does the teacher's token sit?

O-PTP takes the argmax, so the default scoring asks whether the true token is
ranked first. But argmax only needs the ordering to be right, not the logits to
be numerically accurate, and a speculative decoder can accept anything a cheap
verifier confirms. Scoring at top-j says how much is recoverable by relaxing the
acceptance rule rather than by training a better student.

The second question here is slot 0. It is the one position in a block whose
prefix is entirely real tokens, so the teacher's own distribution for it is
computable -- and in a gated or LoRA student the prefix pass already returns it
at no extra cost. Inverting that CDF at u_0 recovers the token by construction,
which is what `--oracle-slot0` substitutes in. Since slot 0's errors propagate
to every later slot, the gap between the two runs is what the student's failure
at one easy position costs the whole block.

Usage:
    PYTHONPATH=src:. python image_ptp/rank_eval.py \
        --ar-ckpt ~/ptp-vqvae/checkpoints/ar_mnist_raster.pt \
        --data ~/ptp-image-results/mnist_tokens.pt \
        --ckpt ~/ptp-image-exp/mnist-S64/last.ckpt --gated --block-len 7
"""
import argparse
import json
from pathlib import Path

import torch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ar-ckpt", type=str, default=None)
    p.add_argument("--data", type=str, required=True)
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--gated", action="store_true")
    p.add_argument("--num-layers", type=int, default=8)
    p.add_argument("--backbone", type=str, default="vqvae_ar",
                   choices=["vqvae_ar", "llamagen"])
    p.add_argument("--llamagen-root", type=str, default="/home/mengy13/LlamaGen")
    p.add_argument("--gpt-model", type=str, default="GPT-B")
    p.add_argument("--gpt-ckpt", type=str, default=None)
    p.add_argument("--adapter-name", type=str, default="linear_interpolation")
    p.add_argument("--adapter-kwargs", type=str, default=None)
    p.add_argument("--block-len", type=int, default=7)
    p.add_argument("--split", type=str, default="val", choices=["val", "all"])
    p.add_argument("--val-split", type=int, default=256)
    p.add_argument("--images", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--topj", type=int, nargs="*", default=[1, 2, 4, 8, 16])
    p.add_argument("--oracle-slot0", action="store_true",
                   help="decide slot 0 by inverting the frozen teacher's CDF "
                        "at u_0 instead of taking the student's argmax")
    p.add_argument("--cfg-scale", type=float, default=4.0,
                   help="llamagen only: must match the value the bin edges were "
                        "built with, since it changes the CDF u inverts")
    p.add_argument("--top-k", type=int, default=100)
    p.add_argument("--top-p", type=float, default=0.999)
    p.add_argument("--teacher-dtype", type=str, default="bfloat16",
                   choices=["bfloat16", "float32"],
                   help="llamagen only: pregenerate.py takes load_llamagen's "
                        "bfloat16 default, so edges cached by it describe a "
                        "bf16 teacher. Inverting an fp32 one recovered 0.64. "
                        "Note the RoPE table is also built per device, so this "
                        "has to run where the edges were generated")
    p.add_argument("--bos", type=int, default=512)
    p.add_argument("--code-vocab", type=int, default=16384)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def leading_run(ok):
    return ((~ok).float().cumsum(dim=1) == 0).sum(dim=1)


@torch.no_grad()
def run(model, teacher, tokens, first, left, right, block_len, topj, oracle0,
        device, batch_size, seed, lg_teacher=None):
    torch.manual_seed(seed)
    seq_len = tokens.shape[1]
    starts = list(range(1, seq_len - block_len + 1, block_len))
    runs = {j: [] for j in topj}
    hits = {j: 0 for j in topj}
    total = 0
    slot0_hit = 0
    slot0_n = 0

    for s in range(0, tokens.shape[0], batch_size):
        t = tokens[s:s + batch_size].to(device)
        l = left[s:s + batch_size].to(device)
        r = right[s:s + batch_size].to(device)
        ids = torch.cat([first[s:s + batch_size].to(device)[:, None], t], 1)
        for start in starts:
            span = min(block_len, seq_len - start + 1)
            lo = l[:, start - 1:start - 1 + span]
            hi = r[:, start - 1:start - 1 + span]
            u = lo + (hi - lo) * torch.rand(lo.shape, device=device)
            _, comp = model(input_ids=ids[:, :start], auxiliaries=u)
            logits = comp.logits[:, :span].float()
            truth = ids[:, start:start + span]

            # Rank of the true token inside each slot's ordering: 0 means argmax.
            order = logits.argsort(dim=-1, descending=True)
            rank = (order == truth.unsqueeze(-1)).float().argmax(dim=-1)

            if oracle0:
                # The teacher's distribution for slot 0 is conditioned only on
                # real tokens, so inverting it at u_0 returns the token exactly --
                # but only if it is the same distribution the edges came from.
                # LlamaGen's were built under guidance and truncation, and the
                # raw logits invert to something else entirely (0.157 recovered).
                if lg_teacher is not None:
                    raw = lg_teacher["model"].teacher_logits(
                        first[s:s + batch_size].to(device) - lg_teacher["code_vocab"],
                        t, lg_teacher["cfg"])[:, start - 1]
                    tl = lg_teacher["filter"](
                        raw.clone(), lg_teacher["top_k"], lg_teacher["top_p"]).float()
                else:
                    tl = teacher(input_ids=ids[:, :start]).logits[:, -1].float()
                cdf = torch.softmax(tl, -1).cumsum(-1)
                pick = torch.searchsorted(cdf.contiguous(),
                                          u[:, :1].contiguous()).squeeze(-1)
                pick = pick.clamp(max=tl.shape[-1] - 1)
                slot0_hit += int((pick == truth[:, 0]).sum())
                slot0_n += pick.shape[0]
                rank = rank.clone()
                rank[:, 0] = torch.where(pick == truth[:, 0],
                                         torch.zeros_like(rank[:, 0]),
                                         torch.full_like(rank[:, 0], 10 ** 6))

            for j in topj:
                ok = rank < j
                runs[j].append(leading_run(ok).cpu())
                hits[j] += int(ok.sum())
            total += truth.numel()

    out = {}
    for j in topj:
        out[j] = (torch.cat(runs[j]).float().mean().item(), hits[j] / total)
    if slot0_n:
        out["slot0"] = slot0_hit / slot0_n
    return out



def first_positions(payload, n, args):
    """The token that opens each sequence.

    The VQ-VAE arms prepend one shared BOS. LlamaGen instead carries the class
    label in the extended vocabulary, so the opening token differs per sample and
    a constant would condition every sequence on class 0.
    """
    if args.backbone == "llamagen":
        labels = payload["labels"][:n].long()
        return labels + args.code_vocab
    return torch.full((n,), args.bos, dtype=torch.long)


def main():
    args = parse_args()
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    from image_ptp.vqvae_ar_hf import build
    from ptp.transformer import MixedTransformerModel
    from image_ptp.gated_full import GatedFullTransformerModel

    build_lg = None
    if args.backbone == "llamagen":
        from image_ptp.llamagen_hf import build as build_lg
        root = Path(args.llamagen_root).expanduser()
        inner, _ = build_lg(root, args.gpt_model,
                            args.gpt_ckpt or str(root / "pretrained_models/c2i_B_256.pt"),
                            dtype=torch.float32, device=device)
    else:
        from image_ptp.vqvae_ar_hf import build_module
        inner = build_module(args.ar_ckpt, device=device, dtype=torch.float32,
                             num_layers=args.num_layers)
    cls = GatedFullTransformerModel if args.gated else MixedTransformerModel
    model = cls(model_id=inner, dtype=torch.float32,
                adapter_name=args.adapter_name,
                adapter_kwargs=(json.loads(args.adapter_kwargs)
                                if args.adapter_kwargs else None),
                attn_implementation="flex_attention").to(device).eval()
    state = torch.load(Path(args.ckpt).expanduser(), map_location="cpu",
                       weights_only=False)["state_dict"]
    missing, unexpected = model.load_state_dict(
        {k[len("model."):]: v for k, v in state.items() if k.startswith("model.")},
        strict=False)
    stale = [k for k in list(missing) + list(unexpected) if "u_embed" in k]
    assert not stale, f"auxiliary embedding did not load: {stale[:3]}"

    teacher, lg_teacher = None, None
    if args.oracle_slot0:
        if args.backbone == "llamagen":
            import sys as _sys
            _sys.path.insert(0, str(Path(__file__).resolve().parent))
            from llamagen_ptp import (LlamaGenPTP, load_llamagen,
                                      top_k_top_p_filter)
            root = Path(args.llamagen_root).expanduser()
            gpt, _ = load_llamagen(root, args.gpt_model,
                                   args.gpt_ckpt or str(root / "pretrained_models/c2i_B_256.pt"),
                                   dtype=getattr(torch, args.teacher_dtype),
                                   device=device)
            lg_teacher = {"model": LlamaGenPTP(gpt, lora_rank=4).to(device).eval(),
                          "filter": top_k_top_p_filter, "cfg": args.cfg_scale,
                          "top_k": args.top_k, "top_p": args.top_p,
                          "code_vocab": args.code_vocab}
        else:
            teacher, _ = build(args.ar_ckpt, device=device, dtype=torch.float32)
            teacher.eval()

    payload = torch.load(Path(args.data).expanduser(), map_location="cpu")
    total = payload["tokens"].shape[0]
    split = total if args.split == "all" else min(args.val_split, total // 4)
    hi = min(split, args.images)
    tokens = payload["tokens"][:hi].long()
    left, right = payload["left_bin_edges"][:hi], payload["right_bin_edges"][:hi]
    first = first_positions(payload, hi, args)

    res = run(model, teacher, tokens, first, left, right, args.block_len,
              args.topj, args.oracle_slot0, device, args.batch_size, args.seed,
              lg_teacher=lg_teacher)
    tag = Path(args.ckpt).parent.name + (" +oracle-slot0" if args.oracle_slot0 else "")
    print(f"{tag}   {hi} sequences, block {args.block_len}")
    if "slot0" in res:
        print(f"  oracle slot 0 recovers the token: {res['slot0']:.4f}")
        # By construction the edges describe the teacher, so an fp32 teacher
        # recovers exactly 1.0 -- MNIST and CIFAR both do. bfloat16 matmuls are
        # not bit-reproducible, which costs LlamaGen about 2%. The threshold has
        # to allow that while still catching the real failures: inverting an
        # fp32 teacher when the edges came from a bf16 one recovers 0.64, and
        # skipping guidance and truncation altogether recovers 0.16.
        assert res["slot0"] > 0.97, (
            f"oracle recovered only {res['slot0']:.4f}: the CDF being inverted is "
            "not the one the bin edges describe -- check cfg/top_k/top_p, the "
            "teacher dtype, and whether this is the machine that generated them")
    print("  top-j   correct    accuracy")
    for j in args.topj:
        run_len, acc = res[j]
        print(f"   {j:3d}    {run_len:6.3f}    {acc:.4f}")


if __name__ == "__main__":
    main()
