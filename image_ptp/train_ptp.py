"""Distil an O-PTP student from a LlamaGen teacher, and measure what it accepts.

Everything measured so far has been a property of the teacher. Two numbers need
a trained student, and this script exists to produce them:

  acceptance    how many proposed tokens survive verification, which is what a
                speedup is made of
  error locality when the student is wrong, is the correct code a codebook
                neighbour of what it proposed? A relaxed acceptance test pays
                off if errors are local and does nothing if they are arbitrary.
                Codebook reordering was supposed to answer this and did not.

Usage:
    python train_ptp.py --llamagen-root ~/LlamaGen \
        --gpt-ckpt ~/LlamaGen/pretrained_models/c2i_B_256.pt \
        --data ~/ptp-image-results/pregen_cfg4_k100.pt \
        --vq-ckpt ~/LlamaGen/pretrained_models/vq_ds16_c2i.pt \
        --steps 4000 --out-dir ~/ptp-image-results/run1
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from llamagen_ptp import LlamaGenPTP, load_llamagen, sample_auxiliaries  # noqa: E402

RELAXED_J = [2, 4, 8, 16, 32, 64]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--llamagen-root", type=str, required=True)
    p.add_argument("--gpt-model", type=str, default="GPT-B")
    p.add_argument("--gpt-ckpt", type=str, required=True)
    p.add_argument("--vq-ckpt", type=str, default=None,
                   help="enables the relaxed-acceptance and error-locality metrics")
    p.add_argument("--data", type=str, required=True)
    p.add_argument("--codebook-size", type=int, default=16384)
    p.add_argument("--codebook-embed-dim", type=int, default=8)
    p.add_argument("--lora-rank", type=int, default=128)
    p.add_argument("--block-len", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--eval-images", type=int, default=256)
    p.add_argument("--eval-starts", type=int, default=4)
    p.add_argument("--holdout", type=int, default=512)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", type=str, required=True)
    return p.parse_args()


def codebook_neighbours(embedding, max_j, chunk=1024):
    v = embedding.shape[0]
    idx = torch.empty((v, max_j), dtype=torch.long, device=embedding.device)
    for start in range(0, v, chunk):
        stop = min(start + chunk, v)
        idx[start:stop] = torch.topk(
            torch.cdist(embedding[start:stop], embedding), max_j, dim=1, largest=False).indices
    return idx


def load_codebook(vq_ckpt, llamagen_root, codebook_size, embed_dim, device):
    sys.path.insert(0, str(llamagen_root))
    from tokenizer.tokenizer_image.vq_model import VQ_models
    vq = VQ_models["VQ-16"](codebook_size=codebook_size, codebook_embed_dim=embed_dim)
    ckpt = torch.load(vq_ckpt, map_location="cpu")
    vq.load_state_dict(ckpt["model"])
    emb = vq.quantize.embedding.weight.detach().float().to(device)
    if getattr(vq.quantize, "l2_norm", False):
        emb = F.normalize(emb, p=2, dim=-1)
    del vq, ckpt
    return emb


def error_ranks(embedding, proposed, truth):
    """Rank of the true code among the proposal's neighbours, for each pair.

    Computed on the fly: a full 16384 x 16384 rank table would cost 2GB, while
    only the mismatched positions ever need one.
    """
    a = embedding[proposed]                                  # (M, D)
    d_truth = (a - embedding[truth]).pow(2).sum(dim=1, keepdim=True)
    d_all = torch.cdist(a, embedding).pow(2)                 # (M, V)
    return (d_all < d_truth).sum(dim=1)


def leading_run(correct):
    """Length of the leading True run in each row of a (B, L) bool tensor."""
    wrong = (~correct).float()
    # cumsum of mistakes is 0 exactly over the leading correct run
    return (wrong.cumsum(dim=1) == 0).sum(dim=1)


@torch.no_grad()
def evaluate(model, data, indices, starts, block_len, device, nn_idx=None, embedding=None):
    model.eval()
    tokens, labels = data["tokens"], data["labels"]
    left_all, right_all = data["left_bin_edges"], data["right_bin_edges"]

    correct_runs, hits, totals = [], 0, 0
    relaxed_runs = {j: [] for j in RELAXED_J} if nn_idx is not None else {}
    err_rank = []

    for start in starts:
        stop = start + block_len - 1
        for chunk in torch.split(indices, 32):
            tok = tokens[chunk].long().to(device)
            lab = labels[chunk].long().to(device)
            left = left_all[chunk][:, start:stop + 1].to(device)
            right = right_all[chunk][:, start:stop + 1].to(device)
            u = sample_auxiliaries(left, right, uniform=True)

            logits = model.student_logits(lab, tok, u, start)
            proposed = logits.argmax(dim=-1)
            truth = tok[:, start:stop + 1]
            correct = proposed == truth

            correct_runs.append(leading_run(correct).cpu())
            hits += int(correct.sum())
            totals += correct.numel()

            if nn_idx is not None:
                for j in RELAXED_J:
                    near = (nn_idx[truth.reshape(-1), :j]
                            == proposed.reshape(-1, 1)).any(dim=1).view(truth.shape)
                    relaxed_runs[j].append(leading_run(near).cpu())
                miss = ~correct
                if miss.any():
                    # Where the student erred, how far is the truth from what it
                    # proposed, in codebook rank? Low ranks mean local errors.
                    err_rank.append(error_ranks(
                        embedding, proposed[miss], truth[miss]).cpu())

    model.train()
    runs = torch.cat(correct_runs).float()
    out = {
        "token_accuracy": hits / max(totals, 1),
        "correct_per_block_mean": runs.mean().item(),
        "accepted_per_step_mean": (runs + 1).clamp(max=block_len).mean().item(),
        "frac_zero_correct": (runs == 0).float().mean().item(),
    }
    for j, chunks in relaxed_runs.items():
        r = torch.cat(chunks).float()
        out[f"relaxed_j{j}_correct_mean"] = r.mean().item()
        out[f"relaxed_j{j}_accepted_mean"] = (r + 1).clamp(max=block_len).mean().item()
    if err_rank:
        e = torch.cat(err_rank).float()
        out["error_rank_median"] = e.median().item()
        for j in (8, 64, 512):
            out[f"error_within_{j}"] = (e < j).float().mean().item()
    return out


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    root = Path(args.llamagen_root).expanduser()
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    data = torch.load(Path(args.data).expanduser(), map_location="cpu")
    cfg = data["config"]
    seq_len = cfg["seq_len"]
    print(f"data: {cfg['num_images']} images, cfg={cfg['cfg_scale']} top_k={cfg['top_k']}")
    assert args.block_len < seq_len

    gpt, model_seq_len = load_llamagen(root, args.gpt_model, args.gpt_ckpt,
                                       dtype=torch.float32, device=device)
    assert model_seq_len == seq_len, "checkpoint and data disagree on sequence length"
    for block in gpt.layers:
        block.attention.kv_cache = None
    model = LlamaGenPTP(gpt, lora_rank=args.lora_rank).to(device)
    model.train()
    print(model.parameter_summary())

    nn_idx = emb = None
    if args.vq_ckpt:
        emb = load_codebook(args.vq_ckpt, root, args.codebook_size,
                            args.codebook_embed_dim, device)
        nn_idx = codebook_neighbours(emb, max(RELAXED_J))
        print("codebook neighbour table ready")

    total = data["tokens"].shape[0]
    generator = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(total, generator=generator)
    eval_idx = perm[:min(args.holdout, total // 4)][:args.eval_images]
    train_idx = perm[len(eval_idx):]
    print(f"train {len(train_idx)} / eval {len(eval_idx)}")

    eval_starts = torch.linspace(0, seq_len - args.block_len, args.eval_starts).long().tolist()
    params = model.trainable_parameters()
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0, betas=(0.9, 0.95))

    def lr_at(step):
        if step < args.warmup:
            return args.lr * (step + 1) / args.warmup
        progress = (step - args.warmup) / max(1, args.steps - args.warmup)
        return args.lr * (0.1 + 0.9 * 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)).item()))

    history, running, started = [], 0.0, time.time()
    for step in range(args.steps):
        for group in opt.param_groups:
            group["lr"] = lr_at(step)

        batch = train_idx[torch.randint(0, len(train_idx), (args.batch_size,),
                                        generator=generator)]
        start = int(torch.randint(0, seq_len - args.block_len + 1, (1,),
                                  generator=generator))
        stop = start + args.block_len - 1

        tok = data["tokens"][batch].long().to(device)
        lab = data["labels"][batch].long().to(device)
        left = data["left_bin_edges"][batch][:, start:stop + 1].to(device)
        right = data["right_bin_edges"][batch][:, start:stop + 1].to(device)
        u = sample_auxiliaries(left, right)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model.student_logits(lab, tok, u, start)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]).float(),
                tok[:, start:stop + 1].reshape(-1))

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        running += loss.item()

        if (step + 1) % 50 == 0:
            rate = (step + 1) / (time.time() - started)
            print(f"step {step + 1}/{args.steps}  loss {running / 50:.4f}  "
                  f"lr {lr_at(step):.2e}  {rate:.1f} it/s", flush=True)
            running = 0.0

        if (step + 1) % args.eval_every == 0 or step + 1 == args.steps:
            metrics = evaluate(model, data, eval_idx, eval_starts, args.block_len,
                               device, nn_idx, emb)
            metrics["step"] = step + 1
            history.append(metrics)
            print(f"  eval @ {step + 1}: acc={metrics['token_accuracy']:.4f} "
                  f"correct/block={metrics['correct_per_block_mean']:.2f} "
                  f"accepted/step={metrics['accepted_per_step_mean']:.2f}"
                  + (f" | relaxed j8={metrics.get('relaxed_j8_correct_mean', 0):.2f} "
                     f"j64={metrics.get('relaxed_j64_correct_mean', 0):.2f} "
                     f"| err rank med={metrics.get('error_rank_median', 0):.0f} "
                     f"within64={metrics.get('error_within_64', 0):.3f}"
                     if nn_idx is not None else ""), flush=True)
            (out_dir / "history.json").write_text(json.dumps(history, indent=2))

    torch.save({
        "lora": {k: v for k, v in model.state_dict().items() if "lora" in k},
        "u_embed": model.u_embed.state_dict(),
        "args": vars(args), "data_config": cfg,
    }, out_dir / "student.pt")
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
