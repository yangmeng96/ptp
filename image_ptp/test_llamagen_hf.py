"""Check the HuggingFace-shaped LlamaGen wrapper against LlamaGen itself.

The wrapper reimplements the attention and layer loop so it can take a
`BlockMask` and a `DynamicCache`. That reimplementation has to produce the same
numbers as the original, and it has to satisfy the contract `ptp.transformer`
and `ptp.lit` call it through. Both are checked here rather than assumed.

Usage:
    python test_llamagen_hf.py --llamagen-root ~/LlamaGen \
        --gpt-ckpt ~/LlamaGen/pretrained_models/c2i_B_256.pt
"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from llamagen_hf import build  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--llamagen-root", type=str, required=True)
    p.add_argument("--gpt-model", type=str, default="GPT-B")
    p.add_argument("--gpt-ckpt", type=str, required=True)
    p.add_argument("--batch-size", type=int, default=2)
    return p.parse_args()


def report(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}{'  ' + detail if detail else ''}")
    return ok


def main():
    args = parse_args()
    torch.manual_seed(0)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, seq_len = build(Path(args.llamagen_root).expanduser(), args.gpt_model,
                           args.gpt_ckpt, dtype=torch.float32, device=device)
    gpt = model.gpt
    b, code_vocab = args.batch_size, model.config.code_vocab
    passed = []

    labels = torch.randint(0, 1000, (b,), device=device)
    codes = torch.randint(0, code_vocab, (b, seq_len - 1), device=device)
    # [class, t_0, ... ] in the wrapper's extended vocabulary
    input_ids = torch.cat([(labels + code_vocab)[:, None], codes], dim=1)

    # 1. Same numbers as LlamaGen's own forward over the same content.
    ref, _ = gpt(idx=codes, cond_idx=labels,
                 input_pos=torch.arange(seq_len, device=device))
    ours = model(input_ids=input_ids).logits
    delta = (ours - ref).abs().max().item()
    passed.append(report("matches LlamaGen forward", delta < 2e-3, f"max|d|={delta:.2e}"))

    # 2. The class must actually be read from the sequence, not ignored.
    other = input_ids.clone()
    other[:, 0] = ((labels + 500) % 1000) + code_vocab
    delta = (model(input_ids=other).logits - ours).abs().max().item()
    passed.append(report("class token changes the output", delta > 1e-2, f"max|d|={delta:.2e}"))

    # 3. A DynamicCache prefill followed by a cached step must equal one pass.
    from transformers import DynamicCache
    cache = DynamicCache()
    split = seq_len // 2
    model(input_ids=input_ids[:, :split], past_key_values=cache, use_cache=True)
    stepped = model(input_ids=input_ids[:, split:], past_key_values=cache,
                    use_cache=True).logits
    delta = (stepped - ours[:, split:]).abs().max().item()
    passed.append(report("cached continuation matches one pass", delta < 2e-3,
                         f"max|d|={delta:.2e}"))

    # 4. inputs_embeds with explicit positions, the path PTP uses for the
    #    auxiliary block: feeding token embeddings back in must be a no-op.
    cache = DynamicCache()
    model(input_ids=input_ids[:, :split], past_key_values=cache, use_cache=True)
    embeds = model.embed(input_ids[:, split:])
    via_embeds = model(
        inputs_embeds=embeds,
        position_ids=torch.arange(split, seq_len, device=device),
        past_key_values=cache, use_cache=False).logits
    delta = (via_embeds - ours[:, split:]).abs().max().item()
    passed.append(report("inputs_embeds path matches", delta < 2e-3, f"max|d|={delta:.2e}"))

    # 5. A BlockMask must route through flex_attention and reproduce causal.
    from torch.nn.attention.flex_attention import create_block_mask

    def causal(bi, hi, q, kv):
        return q >= kv

    block_mask = create_block_mask(causal, B=None, H=None, Q_LEN=seq_len,
                                   KV_LEN=seq_len, device=device)
    flex = model(input_ids=input_ids, attention_mask=block_mask).logits
    delta = (flex - ours).abs().max().item()
    passed.append(report("BlockMask reproduces causal attention", delta < 5e-3,
                         f"max|d|={delta:.2e}"))

    # 6. The contract `ptp.transformer.MixedTransformerModel` reads.
    ok = (model.config.hidden_size == gpt.config.dim
          and hasattr(model, "get_input_embeddings")
          and hasattr(model, "gradient_checkpointing_enable"))
    passed.append(report("exposes the config PTP reads", ok,
                         f"hidden_size={model.config.hidden_size}"))

    # 7. The linears PEFT will target must still be reachable by name.
    from torch import nn
    names = {n.split(".")[-1] for n, m in model.named_modules() if isinstance(m, nn.Linear)}
    wanted = {"wqkv", "wo", "w1", "w2", "w3"}
    passed.append(report("LoRA target modules present", wanted <= names,
                         f"missing={sorted(wanted - names)}"))

    print(f"\n{sum(passed)}/{len(passed)} checks passed")
    return 0 if all(passed) else 1


if __name__ == "__main__":
    raise SystemExit(main())
