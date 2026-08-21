"""Teacher bin widths for a text model, for comparison with the image results.

Same measurement as bin_width_diag.py, run on the model the PTP paper distils
(TinyLlama-1.1B-Chat), so the image numbers can be read against a text baseline
rather than against an estimate.

Usage:
    python text_baseline.py --out text_bin_width.json
"""
import argparse
import json
from pathlib import Path

import torch


PROMPTS = [
    "def factorial(n):",
    "import numpy as np\n\ndef normalize(x):",
    "class BinaryTree:\n    def __init__(self):",
    "The capital of France is",
    "Explain in one sentence why the sky is blue.",
    "Write a short poem about the ocean.",
    "def quicksort(arr):\n    if len(arr) <= 1:",
    "The main difference between a list and a tuple in Python is",
]

# (top_k, top_p, temperature); top_k=0 and top_p=1.0 mean no truncation.
SWEEP = [
    (0, 1.0, 1.0),
    (2000, 1.0, 1.0),
    (100, 1.0, 1.0),
    (0, 1.0, 0.9),
    (0, 0.9, 1.0),
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, required=True)
    return parser.parse_args()


def filter_logits(logits, top_k, top_p):
    """Match LlamaGen's top_k_top_p_filtering so both sides measure the same thing."""
    logits = logits.clone()
    if top_k > 0:
        top_k = min(top_k, logits.shape[-1])
        threshold = torch.topk(logits, top_k, dim=-1).values[..., -1, None]
        logits[logits < threshold] = -float("inf")
    if top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        cumulative = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
        remove = cumulative > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        logits[remove.scatter(-1, sorted_idx, remove)] = -float("inf")
    return logits


@torch.no_grad()
def run_prompt(model, input_ids, max_new_tokens, top_k, top_p, temperature, eos_id):
    widths, eff_cands, supports = [], [], []
    past = None
    cur = input_ids
    for _ in range(max_new_tokens):
        out = model(input_ids=cur, past_key_values=past, use_cache=True)
        past = out.past_key_values
        logits = out.logits[:, -1, :].float() / max(temperature, 1e-5)
        probs = torch.softmax(filter_logits(logits, top_k, top_p), dim=-1)
        token = torch.multinomial(probs, num_samples=1)
        widths.append(probs.gather(1, token).squeeze(1).item())
        eff_cands.append((1.0 / probs.pow(2).sum(dim=1)).item())
        supports.append((probs > 0).sum(dim=1).float().item())
        if token.item() == eos_id:
            break
        cur = token
    return widths, eff_cands, supports


def summarize(widths, eff_cands, supports):
    w = torch.tensor(widths)
    quantiles = torch.tensor([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    return {
        "positions": w.numel(),
        "bin_width_mean": w.mean().item(),
        "bin_width_quantiles": dict(zip(
            [f"p{round(q * 100)}" for q in quantiles.tolist()],
            torch.quantile(w, quantiles).tolist(),
        )),
        "frac_width_below_1e-2": (w < 1e-2).float().mean().item(),
        "frac_width_below_1e-3": (w < 1e-3).float().mean().item(),
        "frac_width_above_0.5": (w > 0.5).float().mean().item(),
        "eff_candidates_mean": torch.tensor(eff_cands).mean().item(),
        "eff_candidates_median": torch.tensor(eff_cands).median().item(),
        "support_mean": torch.tensor(supports).mean().item(),
    }


def main():
    args = parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16
    ).to(device).eval()
    print(f"loaded {args.model}, vocab={model.config.vocab_size}")

    results = []
    for top_k, top_p, temperature in SWEEP:
        torch.manual_seed(args.seed)
        widths, eff_cands, supports = [], [], []
        for prompt in PROMPTS:
            ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
            w, e, s = run_prompt(model, ids, args.max_new_tokens, top_k, top_p,
                                 temperature, tokenizer.eos_token_id)
            widths += w
            eff_cands += e
            supports += s
        entry = {
            "top_k": top_k, "top_p": top_p, "temperature": temperature,
            **summarize(widths, eff_cands, supports),
        }
        results.append(entry)
        print(f"top_k={top_k} top_p={top_p} temp={temperature} | "
              f"width mean={entry['bin_width_mean']:.4f} "
              f"median={entry['bin_width_quantiles']['p50']:.4f} | "
              f"eff_cand mean={entry['eff_candidates_mean']:.1f} "
              f"median={entry['eff_candidates_median']:.1f} | "
              f"frac<1e-2={entry['frac_width_below_1e-2']:.3f}")

    Path(args.out).expanduser().write_text(json.dumps(results, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
