"""Is the frozen teacher's slot-0 distribution really free in a gated student?

Deciding slot 0 by inverting the teacher's CDF at u_0 is the largest single gain
measured so far -- CIFAR 1.024 to 1.697, LlamaGen 0.286 to 1.203 -- and the whole
claim that it costs nothing rests on one unverified assertion: that a gated
student's prefix pass already returns those logits, because it runs on frozen
weights. Every measurement so far got them from a *separate* teacher forward,
which would not be free at inference.

Three checks:

  A  the gated model's prefix logits equal the frozen teacher's, elementwise
  B  inverting those prefix logits at u_0 recovers the true token
  C  an ungated student's prefix logits do NOT -- if they did, gating would be
     buying nothing and the claim would be about something else
"""
import argparse
import json
from pathlib import Path

import torch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ar-ckpt", type=str,
                   default="/home/mengy13/ptp-vqvae/checkpoints/ar_cifar10_raster.pt")
    p.add_argument("--data", type=str,
                   default="/home/mengy13/ptp-image-results/cifar_tokens_test.pt")
    p.add_argument("--gated-ckpt", type=str,
                   default="/home/mengy13/ptp-image-exp/cifar-G513/last.ckpt")
    p.add_argument("--ungated-ckpt", type=str,
                   default="/home/mengy13/ptp-image-exp/cifar-L513/last.ckpt")
    p.add_argument("--adapter-name", type=str, default="linear_interpolation")
    p.add_argument("--adapter-kwargs", type=str, default='{"num_embeddings":513}')
    p.add_argument("--block-len", type=int, default=8)
    p.add_argument("--num-layers", type=int, default=8)
    p.add_argument("--images", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=32)
    return p.parse_args()


def load(ckpt, gated, args, device):
    from image_ptp.vqvae_ar_hf import build_module
    from ptp.transformer import MixedTransformerModel
    from image_ptp.gated_full import GatedFullTransformerModel
    inner = build_module(args.ar_ckpt, device=device, dtype=torch.float32,
                         num_layers=args.num_layers)
    cls = GatedFullTransformerModel if gated else MixedTransformerModel
    m = cls(model_id=inner, dtype=torch.float32, adapter_name=args.adapter_name,
            adapter_kwargs=json.loads(args.adapter_kwargs),
            attn_implementation="flex_attention").to(device).eval()
    state = torch.load(Path(ckpt).expanduser(), map_location="cpu",
                       weights_only=False)["state_dict"]
    missing, unexpected = m.load_state_dict(
        {k[len("model."):]: v for k, v in state.items() if k.startswith("model.")},
        strict=False)
    stale = [k for k in list(missing) + list(unexpected) if "u_embed" in k]
    assert not stale, f"{ckpt}: u embedding did not load: {stale[:3]}"
    return m


@torch.no_grad()
def probe(model, teacher, tokens, left, right, bos, block_len, device, batch_size):
    """Returns (max |prefix - teacher| logit gap, recovery rate from prefix logits)."""
    seq_len = tokens.shape[1]
    starts = list(range(1, seq_len - block_len + 1, block_len))
    torch.manual_seed(0)
    gap = 0.0
    hit = n = 0
    for s in range(0, tokens.shape[0], batch_size):
        t = tokens[s:s + batch_size].to(device)
        l = left[s:s + batch_size].to(device)
        r = right[s:s + batch_size].to(device)
        ids = torch.cat([torch.full((t.shape[0], 1), bos, device=device), t], 1)
        tl_all = teacher(input_ids=ids).logits[:, :-1].float()
        for start in starts:
            span = min(block_len, seq_len - start + 1)
            lo = l[:, start - 1:start - 1 + span]
            hi = r[:, start - 1:start - 1 + span]
            u = lo + (hi - lo) * torch.rand(lo.shape, device=device)
            prefix_out, _ = model(input_ids=ids[:, :start], auxiliaries=u)
            # The prefix ends at ids[start-1]; its last position predicts ids[start].
            pl = prefix_out.logits[:, -1].float()
            tl = tl_all[:, start - 1]
            gap = max(gap, float((pl - tl).abs().max()))
            cdf = torch.softmax(pl, -1).cumsum(-1)
            pred = torch.searchsorted(cdf.contiguous(),
                                      u[:, :1].contiguous()).squeeze(-1)
            pred = pred.clamp(max=pl.shape[-1] - 1)
            hit += int((pred == ids[:, start]).sum())
            n += pred.shape[0]
    return gap, hit / n


def main():
    args = parse_args()
    torch.set_grad_enabled(False)
    device = "cuda"
    from image_ptp.vqvae_ar_hf import build

    teacher, meta = build(args.ar_ckpt, device=device, dtype=torch.float32)
    teacher.eval()
    bos = meta["num_codes"]
    payload = torch.load(Path(args.data).expanduser(), map_location="cpu")
    n = min(args.images, payload["tokens"].shape[0])
    tokens = payload["tokens"][:n].long()
    left, right = payload["left_bin_edges"][:n], payload["right_bin_edges"][:n]

    print("== A/B: gated student ==")
    m = load(args.gated_ckpt, True, args, device)
    gap, rate = probe(m, teacher, tokens, left, right, bos, args.block_len,
                      device, args.batch_size)
    print(f"  max |prefix logit - teacher logit| = {gap:.3e}")
    print(f"  inverting the PREFIX logits at u_0 recovers the truth: {rate:.4f}")
    del m
    torch.cuda.empty_cache()

    print("\n== C: ungated student, same probe ==")
    m2 = load(args.ungated_ckpt, False, args, device)
    gap2, rate2 = probe(m2, teacher, tokens, left, right, bos, args.block_len,
                        device, args.batch_size)
    print(f"  max |prefix logit - teacher logit| = {gap2:.3e}")
    print(f"  inverting the PREFIX logits at u_0 recovers the truth: {rate2:.4f}")

    print()
    assert gap < 1e-3, (
        f"gated prefix logits differ from the teacher by {gap:.3e}: the oracle is "
        "not free, it needs a separate teacher forward")
    assert rate > 0.99, (
        f"gated prefix logits invert to the truth only {rate:.4f} of the time")
    if gap2 < 1e-3:
        print("NOTE: the ungated prefix matches the teacher too -- gating is not "
              "what makes this work, and the free-oracle claim needs restating")
    else:
        print(f"ungated drifts by {gap2:.3e} and recovers only {rate2:.4f}: "
              "gating is what keeps the prefix pass the teacher")
    print("FREE_ORACLE_CONFIRMED", flush=True)


if __name__ == "__main__":
    main()
