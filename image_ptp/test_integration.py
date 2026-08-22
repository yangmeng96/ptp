"""Drive one PTP training step over LlamaGen using only the repo's own code.

The wrapper and the data module are ours; everything between them --
`MixedTransformerModel`, `ParallelSamplingLightningModule`, the bin-edge
inversion, the nested completion masks, the Beta-sampled auxiliaries, the loss
-- is the PTP repo, unmodified. If a step runs and the loss is finite, the
integration holds and no PTP-specific logic needs reimplementing.

Usage:
    PYTHONPATH=src:. python image_ptp/test_integration.py \
        --llamagen-root ~/LlamaGen --data <pregen.pt>
"""
import argparse
import math
from pathlib import Path

import torch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--llamagen-root", type=str, required=True)
    p.add_argument("--gpt-model", type=str, default="GPT-B")
    p.add_argument("--gpt-ckpt", type=str, default=None)
    p.add_argument("--data", type=str, required=True)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--num-completions", type=int, default=32)
    p.add_argument("--completion-len", type=int, default=16)
    p.add_argument("--lora-rank", type=int, default=32)
    p.add_argument("--steps", type=int, default=3)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    from image_ptp.llamagen_hf import build
    from image_ptp.image_data import ImageTokenDataModule
    from ptp.transformer import MixedTransformerModel
    from ptp.lit import ParallelSamplingLightningModule

    ckpt = args.gpt_ckpt or str(
        Path(args.llamagen_root).expanduser() / "pretrained_models/c2i_B_256.pt")
    inner, seq_len = build(Path(args.llamagen_root).expanduser(), args.gpt_model,
                           ckpt, dtype=torch.float32, device=device)
    print(f"wrapped LlamaGen: seq_len={seq_len}, hidden={inner.config.hidden_size}")

    # Master weights in fp32 with autocast around the step, which is what the
    # repo's `precision: bf16-mixed` gives it. Passing a module skips the dtype
    # handling in TransformerModel, so the auxiliary embedding would otherwise
    # stay fp32 against bf16 weights.
    model = MixedTransformerModel(
        model_id=inner,
        dtype=torch.float32,
        lora_config={"r": args.lora_rank,
                     "target_modules": ["wqkv", "wo", "w1", "w2", "w3"]},
        attn_implementation="flex_attention",
    ).to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"{trainable / 1e6:.1f}M trainable of {total / 1e6:.1f}M")

    lit = ParallelSamplingLightningModule(
        model=model,
        optim_cfg={"lr": 1e-4, "lr_warmup": 10},
        top_k=100, top_p=0.999, temperature=1.0,
    ).to(device)

    dm = ImageTokenDataModule(args.data, args.completion_len, args.num_completions,
                              args.batch_size)
    dm.setup()
    loader = dm.train_dataloader()

    params = [p for p in lit.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=1e-4)
    losses = []
    for step, batch in zip(range(args.steps), loader):
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            metrics = lit.forward(batch)
        loss = metrics["loss"]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(float(loss))
        print(f"step {step}: loss={float(loss):.4f} "
              f"l_completion={float(metrics['l_completion']):.4f} "
              f"n_completions={metrics['num_completions']}")

    ok = all(math.isfinite(v) for v in losses)
    grads = [p.grad for p in params if p.grad is not None]
    print(f"\n[{'PASS' if ok else 'FAIL'}] losses finite")
    print(f"[{'PASS' if grads else 'FAIL'}] gradients reached "
          f"{len(grads)}/{len(params)} trainable tensors")
    print(f"supervised positions per step: "
          f"{args.batch_size * args.num_completions * args.completion_len}")
    return 0 if ok and grads else 1


if __name__ == "__main__":
    raise SystemExit(main())
