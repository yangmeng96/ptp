"""Correctness checks for the LlamaGen O-PTP wrapper.

Two things can be wrong in ways that do not raise: the teacher path may not
reproduce LlamaGen's own logits, and the auxiliary block may sit at the wrong
RoPE positions. Both are checked against the upstream model rather than against
expectations.

Usage:
    python test_llamagen_ptp.py --llamagen-root ~/LlamaGen \
        --gpt-ckpt ~/LlamaGen/pretrained_models/c2i_B_256.pt
"""
import argparse
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
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--lora-rank", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def report(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}{'  ' + detail if detail else ''}")
    return ok


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    root = Path(args.llamagen_root).expanduser()

    gpt, seq_len = load_llamagen(
        root, args.gpt_model, args.gpt_ckpt, device=device, dtype=torch.float32)
    # The reference path must not write into a preallocated cache.
    for block in gpt.layers:
        block.attention.kv_cache = None

    model = LlamaGenPTP(gpt, lora_rank=args.lora_rank).to(device).eval()
    print(model.parameter_summary())

    b = args.batch_size
    cond = torch.randint(0, 1000, (b,), device=device)
    # Real samples, not random ids: the teacher assigns near-zero probability to
    # random tokens, which would make the bin-width check vacuous.
    from autoregressive.models.generate import generate
    tokens = generate(model.base, cond, seq_len, cfg_scale=4.0, cfg_interval=-1,
                      temperature=1.0, top_k=100, top_p=1.0).long()
    for block in model.base.layers:
        block.attention.kv_cache = None
    passed = []

    # 1. The teacher path must match LlamaGen's own forward over the same input.
    ref_logits, _ = model.base(
        idx=tokens[:, :-1], cond_idx=cond,
        input_pos=torch.arange(seq_len, device=device))
    with model.adapters(False):
        ours = model.teacher_logits(cond, tokens, cfg_scale=1.0)
    delta = (ours - ref_logits).abs().max().item()
    passed.append(report("teacher matches LlamaGen forward", delta < 2e-3, f"max|d|={delta:.2e}"))

    # 2. Hand the block real token embeddings and hold back the prefix token
    #    they duplicate: the concatenation is then ordinary autoregression, so
    #    the block logits must equal the teacher's. This is what pins down
    #    positions and masking -- a shifted RoPE index or a wrong mask shows up
    #    here and nowhere else.
    start, block_len = 100, 16
    stop = start + block_len - 1
    block_tokens = tokens[:, start - 1:stop]  # t_{start-1} .. t_{stop-1}
    with model.adapters(False):
        block_logits = model.block_forward(
            cond, tokens[:, :start - 1],
            model.base.tok_embeddings(block_tokens), start)
    expect = ref_logits[:, start:stop + 1]
    delta = (block_logits - expect).abs().max().item()
    passed.append(report("auxiliary block layout reproduces AR logits",
                         delta < 2e-3, f"max|d|={delta:.2e}"))

    # 3. At initialisation LoRA is a no-op, so enabling adapters must change
    #    nothing. If it does, the adapters are not initialised as identity.
    with model.adapters(False):
        off = model.teacher_logits(cond, tokens, cfg_scale=1.0)
    on = model.teacher_logits(cond, tokens, cfg_scale=1.0)
    delta = (on - off).abs().max().item()
    passed.append(report("LoRA is identity at init", delta < 1e-5, f"max|d|={delta:.2e}"))

    # 4. Bin edges must bracket a valid interval whose width is the teacher's
    #    probability for the true token, and u must land inside it.
    left, right = model.bin_edges(cond, tokens, start, stop, cfg_scale=4.0, top_k=100)
    widths = right - left
    u = sample_auxiliaries(left, right)
    ok = bool((widths >= 0).all() and (left >= -1e-4).all() and (right <= 1 + 1e-4).all()
              and (u >= left - 1e-6).all() and (u <= right + 1e-6).all())
    passed.append(report("bin edges and u are consistent", ok,
                         f"median width={widths.median().item():.4f}"))

    # 5. Shapes line up for the training step.
    student = model.student_logits(cond, tokens, u, start)
    ok = student.shape == (b, block_len, model.vocab_size)
    passed.append(report("student logits shape", ok, str(tuple(student.shape))))

    # 6. Guidance must actually change the teacher, and only through the class.
    guided = model.teacher_logits(cond, tokens, cfg_scale=4.0)
    plain = model.teacher_logits(cond, tokens, cfg_scale=1.0)
    delta = (guided - plain).abs().max().item()
    passed.append(report("cfg changes the teacher distribution", delta > 1e-2,
                         f"max|d|={delta:.2e}"))

    print(f"\n{sum(passed)}/{len(passed)} checks passed")
    return 0 if all(passed) else 1


if __name__ == "__main__":
    raise SystemExit(main())
