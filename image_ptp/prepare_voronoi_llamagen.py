"""Quota-assigned Voronoi auxiliaries for LlamaGen.

The CIFAR port took the scheme from 1.024 to 1.447 correct at K=4096, so the
question is whether it survives the setting it was meant for: a 16384-entry
codebook, bins a tenth as wide, and a distribution that only exists under
guidance.

Three things differ from the CIFAR version and each is a place the earlier work
went wrong once already:

  the distribution   the edges in pregen_cfg4_k100_big.pt were built by
                     LlamaGenPTP.bin_edges at cfg 4, top-k 100, top-p 0.999, in
                     bfloat16, on lucy. Cells cut from anything else describe a
                     different teacher: an fp32 pass recovers 0.64 of the tokens
                     and an unguided one 0.16. Same call, same dtype, here.
  the candidates     top-k leaves 100 tokens with any mass at all, so the
                     shortlist is exact rather than an approximation and the
                     assignment is small.
  the geometry       token positions are the rows of the model's output head,
                     768-dimensional, which is also where the sphere lives, so
                     an auxiliary vector can be read straight off as an input
                     embedding.

The sphere is stored in the payload; the trainer reads it from there rather than
rebuilding it from a seed.
"""
import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, "/home/mengy13/ptp")
sys.path.insert(0, "/home/mengy13/ptp/image_ptp")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--llamagen-root", type=str, default="/home/mengy13/LlamaGen")
    p.add_argument("--gpt-model", type=str, default="GPT-B")
    p.add_argument("--gpt-ckpt", type=str, default=None)
    p.add_argument("--tokens", type=str, required=True)
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--K", type=int, default=4096)
    p.add_argument("--cfg-scale", type=float, default=4.0)
    p.add_argument("--top-k", type=int, default=100)
    p.add_argument("--top-p", type=float, default=0.999)
    p.add_argument("--teacher-dtype", type=str, default="bfloat16",
                   choices=["bfloat16", "float32"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--upstream", type=str, default="/home/mengy13/upstream_ptp")
    p.add_argument("--limit", type=int, default=0, help="0 uses every sequence")
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda"
    torch.set_grad_enabled(False)
    from llamagen_ptp import LlamaGenPTP, load_llamagen, top_k_top_p_filter
    from image_ptp.prepare_voronoi_tokens import quotas_from_probs
    sys.path.insert(0, args.upstream)
    from proportional_assignment import assign as ref_assign

    root = Path(args.llamagen_root).expanduser()
    gpt, seq_len = load_llamagen(
        root, args.gpt_model,
        args.gpt_ckpt or str(root / "pretrained_models/c2i_B_256.pt"),
        dtype=getattr(torch, args.teacher_dtype), device=device)
    model = LlamaGenPTP(gpt, lora_rank=4).to(device).eval()

    W = gpt.output.weight.data
    V, dim = W.shape
    positions = F.normalize(W.float(), dim=-1)
    g = torch.Generator(device=device).manual_seed(args.seed)
    sphere = F.normalize(torch.randn(args.K, dim, generator=g, device=device),
                         dim=-1)
    S = (positions @ sphere.T).contiguous()                  # (V, K)
    print(f"vocab {V}, dim {dim}, K {args.K}, cfg {args.cfg_scale}, "
          f"top_k {args.top_k}, dtype {args.teacher_dtype}", flush=True)

    payload = torch.load(Path(args.tokens).expanduser(), map_location="cpu")
    tokens = payload["tokens"].long()
    labels = payload["labels"].long()
    if args.limit:
        tokens, labels = tokens[:args.limit], labels[:args.limit]
    n_seq = tokens.shape[0]
    M = args.top_k
    aux_ids = torch.zeros(n_seq, seq_len, dtype=torch.int32)
    masked = 0
    started = time.time()
    gen = torch.Generator(device=device).manual_seed(args.seed + 1)

    for s in range(0, n_seq, args.batch_size):
        t = tokens[s:s + args.batch_size].to(device)
        lab = labels[s:s + args.batch_size].to(device)
        logits = model.teacher_logits(lab, t, args.cfg_scale)
        logits = top_k_top_p_filter(logits.clone(), args.top_k, args.top_p)
        Q = torch.softmax(logits.float(), dim=-1).reshape(-1, V)
        truth = t.reshape(-1)
        P = Q.shape[0]

        # Truncation leaves at most top_k tokens with mass, so the shortlist is
        # the support itself rather than an approximation of it.
        cand = Q.topk(M, dim=-1).indices                      # (P, M)
        qk = quotas_from_probs(Q.gather(1, cand)
                              / Q.gather(1, cand).sum(-1, keepdim=True), args.K)
        short = args.K - qk.sum(dim=1)
        qk[:, 0] += short
        cost = (-S[cand]).to(torch.bfloat16).contiguous()      # (P, M, K)
        sigma = ref_assign(cost, counts=qk.to(torch.int32), k=8, rounds=3)
        owner = cand.gather(1, sigma.clamp(0, M - 1))
        owner = torch.where(sigma < M, owner, torch.full_like(owner, -1))

        mine = owner == truth[:, None]
        count = mine.sum(dim=1)
        r = torch.rand(P, generator=gen, device=device)
        pick = (r * count.clamp_min(1)).long()
        csum = mine.long().cumsum(dim=1)
        chosen = ((csum == (pick + 1)[:, None]) & mine).float().argmax(dim=1)
        chosen = torch.where(count > 0, chosen,
                             torch.full_like(chosen, args.K))
        masked += int((count == 0).sum())
        aux_ids[s:s + args.batch_size] = chosen.reshape(t.shape).to(
            torch.int32).cpu()

        done = min(s + args.batch_size, n_seq)
        if (s // args.batch_size) % 200 == 0 or done == n_seq:
            rate = done / max(time.time() - started, 1e-9)
            print(f"  {done}/{n_seq}  {rate:.1f} seq/s  eta "
                  f"{(n_seq - done) / max(rate, 1e-9) / 60:.0f} min", flush=True)

    total = n_seq * seq_len
    print(f"mask rate: {masked / total:.4f} ({masked} of {total})")
    u = (aux_ids.float() + 0.5) / (args.K + 1)
    payload = {k: v for k, v in payload.items()}
    payload["tokens"] = tokens.to(torch.int32)
    payload["labels"] = labels.to(torch.int32)
    payload["left_bin_edges"] = u
    payload["right_bin_edges"] = u.clone()
    payload["aux_ids"] = aux_ids
    payload["sphere"] = sphere.cpu()
    payload["config"] = dict(payload.get("config", {}))
    payload["config"].update({
        "auxiliary": "voronoi-proportional", "K": args.K,
        "sphere_seed": args.seed, "cfg_scale": args.cfg_scale,
        "top_k": args.top_k, "top_p": args.top_p,
        "teacher_dtype": args.teacher_dtype, "mask_rate": masked / total})
    out = Path(args.out).expanduser()
    torch.save(payload, out)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
