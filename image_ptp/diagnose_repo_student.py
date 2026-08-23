"""Does the student trained through the PTP repo actually use the auxiliaries?

`val/correct` cannot answer this. A student that ignores u entirely and predicts
the teacher's marginal still scores above zero on it, and that is exactly the
failure mode every earlier LlamaGen attempt fell into. The test that separates
the two is to re-run the same evaluation with auxiliaries taken from other
sequences: if accuracy barely moves, u is not being read.

The reference point is the MNIST run in `~/ptp-vqvae`, which scores 1.70x on
this test. Every LlamaGen run so far has scored 1.0x.

Usage:
    PYTHONPATH=src:. python image_ptp/diagnose_repo_student.py \
        --llamagen-root ~/LlamaGen --data <pregen.pt> \
        --ckpt ~/ptp-image-exp/llamagen-b-k100/last.ckpt
"""
import argparse
from pathlib import Path

import torch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", type=str, default="llamagen",
                   choices=["llamagen", "vqvae_ar"])
    p.add_argument("--llamagen-root", type=str, default=None)
    p.add_argument("--ar-ckpt", type=str, default=None,
                   help="ptp-vqvae AR checkpoint, for backbone=vqvae_ar")
    p.add_argument("--code-vocab", type=int, default=16384)
    p.add_argument("--prepend-label", type=int, default=1)
    p.add_argument("--gated", action="store_true",
                   help="checkpoint was trained with a frozen prefix pass; "
                        "evaluating it ungated runs a configuration it never saw")
    p.add_argument("--num-layers", type=int, default=None,
                   help="student depth, when it differs from the teacher's")
    p.add_argument("--adapter-name", type=str, default="linear_interpolation",
                   help="must match the trained checkpoint, or its u embedding "
                        "silently fails to load and the lift reads as 1.0")
    p.add_argument("--gpt-model", type=str, default="GPT-B")
    p.add_argument("--gpt-ckpt", type=str, default=None)
    p.add_argument("--data", type=str, required=True)
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--lora-rank", type=int, default=128)
    p.add_argument("--top-k", type=int, default=100)
    p.add_argument("--top-p", type=float, default=0.999)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-completions", type=int, default=32)
    p.add_argument("--completion-len", type=int, default=16)
    p.add_argument("--batches", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    from image_ptp.image_data import ImageTokenDataModule
    from ptp.transformer import MixedTransformerModel
    from ptp.lit import ParallelSamplingLightningModule
    if args.gated:
        from image_ptp.gated_full import GatedFullTransformerModel

    if args.backbone == "llamagen":
        from image_ptp.llamagen_hf import build
        root = Path(args.llamagen_root).expanduser()
        gpt_ckpt = args.gpt_ckpt or str(root / "pretrained_models/c2i_B_256.pt")
        inner, _ = build(root, args.gpt_model, gpt_ckpt,
                         dtype=torch.float32, device=device)
        targets = ["wqkv", "wo", "w1", "w2", "w3"]
    else:
        from image_ptp.vqvae_ar_hf import build_module
        inner = build_module(args.ar_ckpt, device=device, dtype=torch.float32,
                             num_layers=args.num_layers)
        targets = ["q_proj", "k_proj", "v_proj", "o_proj", "linear1", "linear2"]

    # A full-finetune checkpoint has no adapter tensors; adding LoRA here would
    # leave the loaded weights sitting under the wrong keys.
    state_peek = torch.load(Path(args.ckpt).expanduser(), map_location="cpu",
                            weights_only=False)["state_dict"]
    has_lora = any("lora_" in k for k in state_peek)
    model_cls = GatedFullTransformerModel if args.gated else MixedTransformerModel
    model = model_cls(
        model_id=inner, dtype=torch.float32,
        lora_config={"r": args.lora_rank, "target_modules": targets} if has_lora else None,
        adapter_name=args.adapter_name,
        attn_implementation="flex_attention",
    ).to(device)
    print(f"checkpoint is {'LoRA' if has_lora else 'full finetune'}")
    lit = ParallelSamplingLightningModule(
        model=model, optim_cfg={"lr": 1e-4, "lr_warmup": 10},
        top_k=args.top_k, top_p=args.top_p, temperature=1.0,
    ).to(device)

    # Load into the inner model, not the LightningModule. Its `load_state_dict`
    # rewrites any ".u_adapter." in a key to ".u_embed.", which is presumably
    # legacy-name handling -- but BinaryFloatEmbedding's parameter is literally
    # named `u_adapter`, so the rewrite renames it to something the model does
    # not have and the auxiliary embedding is silently dropped. A checkpoint
    # loaded that way reports a lift of exactly 1.0 while looking healthy.
    state = {k[len("model."):]: v for k, v in state_peek.items() if k.startswith("model.")}
    missing, unexpected = model.load_state_dict(state, strict=False)
    trained = [k for k in state if "lora_" in k or "u_embed" in k]
    print(f"loaded {len(state)} tensors ({len(trained)} adapter/auxiliary), "
          f"{len(missing)} missing, {len(unexpected)} unexpected")
    # A mismatched adapter_name leaves the auxiliary embedding at its random
    # initialisation, which reads as a lift of exactly 1.0 -- the same signature
    # as a student that ignores u. Refuse to report that as a result.
    stale = [k for k in missing if "u_embed" in k] + [k for k in unexpected if "u_embed" in k]
    assert not stale, (f"u embedding did not load: {stale[:3]}. "
                       f"Pass --adapter-name matching the checkpoint.")
    lit.eval()

    dm = ImageTokenDataModule(args.data, args.completion_len,
                              args.num_completions, args.batch_size,
                              code_vocab=args.code_vocab,
                              prepend_label=bool(args.prepend_label))
    dm.setup()

    def run(shuffle_u):
        hits = total = 0
        runs = []
        loader = dm.val_dataloader()
        for i, batch in zip(range(args.batches), loader):
            batch = {k: (v.to(device) if torch.is_tensor(v) else v)
                     for k, v in batch.items()}
            torch.manual_seed(args.seed + i)  # same u draw for both arms
            with torch.autocast("cuda", dtype=torch.float16):
                metrics = lit.forward(batch, eval=True, return_outputs=True)
            # `forward` returns the sequence metrics the repo computes; the
            # per-sample correct counts are what the acceptance number is made of.
            counts = metrics.get("correct_counts")
            if counts is not None:
                runs.append(counts.float().cpu())
            hits += float(metrics["accuracy"]) * 1.0
            total += 1
        acc = hits / max(total, 1)
        run_len = torch.cat(runs).mean().item() if runs else float("nan")
        return acc, run_len

    # The shuffle is applied inside the model by permuting the auxiliaries, so
    # patch the sampler rather than the batch: the repo derives u from the
    # teacher's bin edges internally.
    original = ParallelSamplingLightningModule.sample_auxiliaries

    def shuffled(self, left, right, eval):
        u = original(self, left, right, eval)
        return u[torch.randperm(u.shape[0], device=u.device)]

    real_acc, real_run = run(False)
    ParallelSamplingLightningModule.sample_auxiliaries = shuffled
    try:
        shuf_acc, shuf_run = run(True)
    finally:
        ParallelSamplingLightningModule.sample_auxiliaries = original

    print("\n--- does the student use u? ---")
    print(f"accuracy   real u {real_acc:.4f}   shuffled {shuf_acc:.4f}   "
          f"lift {real_acc / max(shuf_acc, 1e-9):.2f}x")
    print(f"correct    real u {real_run:.4f}   shuffled {shuf_run:.4f}   "
          f"lift {real_run / max(shuf_run, 1e-9):.2f}x")
    print("\nMNIST reference (ptp-vqvae, works): 1.70x")
    print("every earlier LlamaGen attempt:      1.0x")


if __name__ == "__main__":
    main()
