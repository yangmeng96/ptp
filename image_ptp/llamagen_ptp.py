"""O-PTP on top of a LlamaGen autoregressive image model.

LlamaGen's transformer is not a HuggingFace model: its KV cache lives inside the
attention layers, its RoPE table is indexed explicitly, and `forward` returns a
tuple. Rather than emulating the HuggingFace cache API, this module drives the
layers directly. Training never needs the cache -- one pass over
`[class | prefix tokens | auxiliary block]` with a causal mask is enough.

Position alignment is the part that has to be right. In LlamaGen the output at
sequence index k predicts token t_k, and index k carries RoPE entry k. The
auxiliary u_k stands in for t_k, so it is given RoPE index k -- the same entry
the prefix's last token would hold. Getting this wrong does not crash; it
quietly discards the spatial prior in the pretrained weights.
"""
from contextlib import contextmanager

import torch
import torch.nn.functional as F
from torch import nn

LORA_TARGETS = ["wqkv", "wo", "w1", "w2", "w3"]


class BinaryFloatEmbedding(nn.Module):
    """embed(u) = W binary(u) + b, with binary(u) the 32 bits of the float32.

    This is the scheme the PTP paper reports (Eq. 16). Bit-level input gives the
    network access to arbitrarily fine distinctions between nearby u, which
    matters here because image bins are narrow.
    """

    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Linear(32, dim)

    def forward(self, u):
        bits = u.to(torch.float32).view(torch.int32)
        masks = 2 ** torch.arange(31, -1, -1, device=u.device, dtype=torch.int32)
        expanded = ((bits[..., None] & masks) != 0).to(self.proj.weight.dtype)
        return self.proj(expanded)


def top_k_top_p_filter(logits, top_k=0, top_p=1.0):
    """Match LlamaGen's own truncation so bin edges describe the sampled distribution."""
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


class LlamaGenPTP(nn.Module):
    """Wraps a LlamaGen `Transformer` as an O-PTP student plus its own teacher.

    The base weights are frozen and serve as the teacher; LoRA adapters and the
    auxiliary embedding are the only trainable parts, so a single set of weights
    plays both roles -- disable the adapters and the model is the teacher again.
    """

    def __init__(self, gpt, lora_rank=128, lora_alpha=None, lora_dropout=0.0):
        super().__init__()
        from peft import LoraConfig, get_peft_model

        self.config = gpt.config
        self.cls_token_num = gpt.cls_token_num
        self.num_classes = gpt.num_classes
        self.vocab_size = gpt.vocab_size

        for p in gpt.parameters():
            p.requires_grad_(False)
        self.gpt = get_peft_model(gpt, LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha if lora_alpha is not None else lora_rank,
            lora_dropout=lora_dropout,
            target_modules=LORA_TARGETS,
            bias="none",
        ))
        self.u_embed = BinaryFloatEmbedding(self.config.dim)

    @property
    def base(self):
        """The underlying LlamaGen module, past the PEFT wrapper."""
        return self.gpt.base_model.model

    @contextmanager
    def adapters(self, enabled):
        from peft.tuners.tuners_utils import BaseTunerLayer
        for module in self.gpt.modules():
            if isinstance(module, BaseTunerLayer):
                module.enable_adapters(enabled)
        try:
            yield
        finally:
            for module in self.gpt.modules():
                if isinstance(module, BaseTunerLayer):
                    module.enable_adapters(True)

    def _freqs(self, positions):
        base = self.base
        table = base.freqs_cis
        if table.device != positions.device:
            table = base.freqs_cis = table.to(positions.device)
        return table[positions]

    def _run(self, h, positions, mask):
        """Push embeddings through the transformer stack with explicit positions."""
        base = self.base
        freqs = self._freqs(positions)
        for layer in base.layers:
            h = layer(h, freqs, None, mask)
        return base.output(base.norm(h)).float()

    def _class_embed(self, cond):
        # train=False keeps LabelEmbedder from randomly dropping the class; the
        # caller decides when to use the null class.
        return self.base.cls_embedding(cond, train=False)[:, :self.cls_token_num]

    @staticmethod
    def _causal_mask(length, batch, device):
        m = torch.tril(torch.ones(length, length, dtype=torch.bool, device=device))
        return m[None, None].expand(batch, 1, length, length)

    def teacher_logits(self, cond, tokens, cfg_scale=1.0):
        """Unadapted next-token logits for a full sequence.

        Returns (B, S, V) where entry j is the teacher's distribution for token
        j, already combined across the guided and unguided branches.
        """
        batch = tokens.shape[0]
        guided = cfg_scale > 1.0
        if guided:
            cond = torch.cat([cond, torch.full_like(cond, self.num_classes)])
            tokens = torch.cat([tokens, tokens])

        embeds = torch.cat([self._class_embed(cond), self.base.tok_embeddings(tokens)], dim=1)
        length = embeds.shape[1]
        positions = torch.arange(length, device=tokens.device)
        with self.adapters(False), torch.no_grad():
            logits = self._run(embeds, positions,
                               self._causal_mask(length, embeds.shape[0], tokens.device))
        # Drop the trailing slot: it would predict one past the sequence.
        logits = logits[:, :-1]
        if guided:
            cond_logits, uncond_logits = torch.split(logits, batch, dim=0)
            logits = uncond_logits + (cond_logits - uncond_logits) * cfg_scale
        return logits

    def bin_edges(self, cond, tokens, start, stop, cfg_scale=1.0,
                  top_k=0, top_p=1.0, temperature=1.0):
        """Interval [left, right) that the teacher assigns to each true token.

        `start`/`stop` bound the supervised block, inclusive of both ends.
        Returns two (B, stop - start + 1) tensors.
        """
        logits = self.teacher_logits(cond, tokens, cfg_scale)[:, start:stop + 1]
        logits = top_k_top_p_filter(logits / max(temperature, 1e-5), top_k, top_p)
        probs = torch.softmax(logits.float(), dim=-1)
        cdf = probs.cumsum(dim=-1)
        target = tokens[:, start:stop + 1, None]
        right = cdf.gather(2, target).squeeze(2)
        return (right - probs.gather(2, target).squeeze(2)), right

    def block_forward(self, cond, prefix, block_embeds, start):
        """Run [class | prefix | block] and return the logits over the block.

        `start` is the RoPE index of the block's first slot, independent of how
        many prefix tokens are supplied. Split out from `student_logits` so the
        layout can be tested: hand it one fewer prefix token and a block of real
        token embeddings and the concatenation becomes plain autoregression, so
        the output must match the teacher over the same span.
        """
        device = prefix.device
        block_len = block_embeds.shape[1]
        embeds = torch.cat([
            self._class_embed(cond),
            self.base.tok_embeddings(prefix),
            block_embeds,
        ], dim=1)
        prefix_len = self.cls_token_num + prefix.shape[1]

        # The prefix runs 0..start; u_k takes RoPE index k, so the block runs
        # start..stop and deliberately reuses index `start`.
        positions = torch.cat([
            torch.arange(prefix_len, device=device),
            torch.arange(start, start + block_len, device=device),
        ])
        length = prefix_len + block_len
        logits = self._run(embeds, positions,
                           self._causal_mask(length, embeds.shape[0], device))
        return logits[:, prefix_len:]

    def student_logits(self, cond, tokens, u_block, start):
        """Predict the block t_start .. t_stop from the prefix and the auxiliaries.

        `tokens` is the full sequence; only t_0 .. t_{start-1} is read. Returns
        (B, len(u_block), V), aligned so entry m predicts t_{start + m}.
        """
        return self.block_forward(cond, tokens[:, :start],
                                  self.u_embed(u_block), start)

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def parameter_summary(self):
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        return f"{trainable / 1e6:.1f}M trainable of {total / 1e6:.1f}M ({trainable / total:.1%})"


def sample_auxiliaries(left, right, beta=0.3, uniform=False):
    """Draw u inside each bin, biased toward the edges during training.

    Bin boundaries are where the student's decision flips, so edge-heavy samples
    concentrate supervision on the cases that decide acceptance. Evaluation uses
    a uniform draw, which is what inference actually does.
    """
    if uniform:
        z = torch.rand(left.shape, device=left.device, dtype=torch.float32)
    else:
        conc = torch.full((), beta, device=left.device, dtype=torch.float32)
        z = torch.distributions.Beta(conc, conc).sample(left.shape).to(left.device)
    return left + (right - left) * z


def load_llamagen(llamagen_root, gpt_model, gpt_ckpt, image_size=256, downsample=16,
                  codebook_size=16384, num_classes=1000, cls_token_num=1,
                  dtype=torch.bfloat16, device="cuda"):
    import sys
    sys.path.insert(0, str(llamagen_root))
    from autoregressive.models.gpt import GPT_models

    seq_len = (image_size // downsample) ** 2
    gpt = GPT_models[gpt_model](
        vocab_size=codebook_size, block_size=seq_len,
        num_classes=num_classes, cls_token_num=cls_token_num, model_type="c2i",
    ).to(device=device, dtype=dtype)
    ckpt = torch.load(gpt_ckpt, map_location="cpu")
    for key in ("model", "module", "state_dict"):
        if key in ckpt:
            ckpt = ckpt[key]
            break
    missing, unexpected = gpt.load_state_dict(ckpt, strict=False)
    if missing:
        print(f"missing keys: {len(missing)} (first: {missing[:3]})")
    gpt.eval()
    return gpt, seq_len
