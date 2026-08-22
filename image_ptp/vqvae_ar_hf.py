"""A HuggingFace-shaped causal LM backed by the MNIST VQ-VAE AR teacher.

This exists to give the LlamaGen work a positive control. The `ptp-vqvae`
project already trains an O-PTP student on these weights and gets a 1.70x lift
from the auxiliaries, so running the same teacher through the PTP repo's
training code should reproduce that. If it does not, the fault is in this
pipeline rather than in anything about images or LoRA -- which is the question
the LlamaGen runs cannot answer on their own.

Two things are deliberately not faithful to `models/ar.py`:

  dropout      forced to zero. PTP evaluates the teacher through `ar_forward`
               while the module sits in train mode, so the original 0.1 would
               put noise into the bin edges that define `u`. HuggingFace decoder
               models default to no dropout, which is why the repo never trips
               on this.

  attention    `nn.MultiheadAttention` packs q, k and v into a single
               `in_proj_weight` Parameter, which PEFT cannot target and
               flex_attention cannot take apart. The projections are split into
               named Linears and the checkpoint is sliced into them, leaving the
               computation identical but the modules addressable.
"""
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.attention.flex_attention import BlockMask, flex_attention

_COMPILED_FLEX = None
_COMPILE_FAILED = False


def _flex(query, key, value, block_mask):
    """Compiled flex_attention where the backend accepts it, eager where it does not.

    Inductor raises `IndexError: map::at` on this model's 32-wide heads. The
    sequences here are 49 tokens against LlamaGen's 256, so the eager path costs
    little -- unlike on the larger model, where compiling took a step from 11.5s
    to 2.9s and is worth insisting on.
    """
    global _COMPILED_FLEX, _COMPILE_FAILED
    if not _COMPILE_FAILED:
        try:
            if _COMPILED_FLEX is None:
                _COMPILED_FLEX = torch.compile(flex_attention, dynamic=False)
            return _COMPILED_FLEX(query, key, value, block_mask=block_mask)
        except Exception:
            _COMPILE_FAILED = True
            # Print the whole traceback. Reporting only the exception type hides
            # exactly what is needed to tell a backend limitation apart from a
            # bug in the caller, and the two call for opposite responses.
            import traceback
            print("flex_attention compile failed; falling back to eager. "
                  "Full traceback follows:")
            traceback.print_exc()
    return flex_attention(query, key, value, block_mask=block_mask)


@dataclass
class ARHFConfig:
    hidden_size: int
    vocab_size: int
    num_attention_heads: int
    num_hidden_layers: int
    max_position_embeddings: int
    model_type: str = "llama"
    tie_word_embeddings: bool = False


@dataclass
class CausalOutput:
    logits: torch.Tensor
    past_key_values: object = None


class ARBlock(nn.Module):
    """One `nn.TransformerEncoderLayer(norm_first=True)` with split projections."""

    def __init__(self, d_model, n_head, dim_feedforward):
        super().__init__()
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def load_encoder_layer(self, state, prefix):
        """Slice a MultiheadAttention layer's packed weights into the projections."""
        d = self.q_proj.out_features
        w = state[f"{prefix}.self_attn.in_proj_weight"]
        b = state[f"{prefix}.self_attn.in_proj_bias"]
        for i, proj in enumerate((self.q_proj, self.k_proj, self.v_proj)):
            proj.weight.data.copy_(w[i * d:(i + 1) * d])
            proj.bias.data.copy_(b[i * d:(i + 1) * d])
        self.o_proj.weight.data.copy_(state[f"{prefix}.self_attn.out_proj.weight"])
        self.o_proj.bias.data.copy_(state[f"{prefix}.self_attn.out_proj.bias"])
        for name in ("linear1", "linear2", "norm1", "norm2"):
            getattr(self, name).weight.data.copy_(state[f"{prefix}.{name}.weight"])
            getattr(self, name).bias.data.copy_(state[f"{prefix}.{name}.bias"])

    def attend(self, x, mask, past_key_values, layer_idx):
        bsz, seqlen, _ = x.shape
        shape = (bsz, seqlen, self.n_head, self.head_dim)
        q = self.q_proj(x).view(shape).transpose(1, 2)
        k = self.k_proj(x).view(shape).transpose(1, 2)
        v = self.v_proj(x).view(shape).transpose(1, 2)
        if past_key_values is not None:
            k, v = past_key_values.update(k, v, layer_idx)
        if isinstance(mask, BlockMask):
            out = _flex(q, k, v, mask)
        else:
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=mask,
                is_causal=mask is None and k.shape[2] == q.shape[2])
        out = out.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.o_proj(out)

    def forward(self, x, mask, past_key_values, layer_idx):
        x = x + self.attend(self.norm1(x), mask, past_key_values, layer_idx)
        return x + self.linear2(F.relu(self.linear1(self.norm2(x))))


class ARForCausalLM(nn.Module):
    def __init__(self, num_codes, d_model, n_head, num_layers, dim_feedforward,
                 max_seq_len):
        super().__init__()
        vocab = num_codes + 1  # the last id is BOS
        self.config = ARHFConfig(
            hidden_size=d_model, vocab_size=vocab, num_attention_heads=n_head,
            num_hidden_layers=num_layers, max_position_embeddings=max_seq_len,
        )
        self.num_codes = num_codes
        self.tok_emb = nn.Embedding(vocab, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.layers = nn.ModuleList(
            ARBlock(d_model, n_head, dim_feedforward) for _ in range(num_layers))
        self.lm_head = nn.Linear(d_model, vocab)

    def get_input_embeddings(self):
        return self.tok_emb

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        pass

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      attention_mask=None, inputs_embeds=None, **kwargs):
        return {"input_ids": input_ids, "past_key_values": past_key_values,
                "attention_mask": attention_mask, "use_cache": True}

    def forward(self, input_ids=None, inputs_embeds=None, attention_mask=None,
                position_ids=None, past_key_values=None, use_cache=False,
                auxiliaries=None, **kwargs):
        if use_cache and past_key_values is None:
            from transformers import DynamicCache
            past_key_values = DynamicCache()

        if inputs_embeds is None:
            inputs_embeds = self.tok_emb(input_ids)
        bsz, seqlen, _ = inputs_embeds.shape
        device = inputs_embeds.device

        if position_ids is None:
            offset = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(offset, offset + seqlen, device=device)
        if position_ids.dim() == 2:
            position_ids = position_ids[0]
        # PTP shifts auxiliary positions to -1 for the very first completion slot.
        h = inputs_embeds + self.pos_emb(position_ids.clamp(min=0))

        if attention_mask is None and past_key_values is not None:
            cached = past_key_values.get_seq_length()
            if cached and seqlen > 1:
                q_pos = torch.arange(seqlen, device=device) + cached
                k_pos = torch.arange(cached + seqlen, device=device)
                attention_mask = (k_pos[None, :] <= q_pos[:, None])[None, None]

        for idx, layer in enumerate(self.layers):
            h = layer(h, attention_mask, past_key_values, idx)
        return CausalOutput(logits=self.lm_head(h).float(),
                            past_key_values=past_key_values)


def build(ar_ckpt, device="cuda", dtype=torch.float32, num_layers=None):
    """Load an `ar_*.pt` checkpoint from ptp-vqvae into the wrapper.

    `num_layers` may exceed the checkpoint's depth to give the student more
    capacity than the teacher, which it needs: the student has to reproduce the
    teacher's distribution *and* invert it against u, for several positions at
    once without seeing the tokens in between. The extra layers are copies of
    the teacher's, cycled, so the student starts out able to model the sequence
    and spends its new capacity on the inversion rather than relearning the
    task. Only meaningful when the bin edges come from a separate frozen
    teacher -- otherwise the student is its own teacher and depth is moot.
    """
    if isinstance(dtype, str):
        dtype = getattr(torch, dtype)
    ckpt = torch.load(ar_ckpt, map_location="cpu", weights_only=False)
    depth = num_layers or ckpt["num_layers"]
    model = ARForCausalLM(
        num_codes=ckpt["num_codes"], d_model=ckpt["d_model"], n_head=ckpt["nhead"],
        num_layers=depth, dim_feedforward=ckpt["dim_feedforward"],
        max_seq_len=ckpt["max_seq_len"],
    )
    state = ckpt["ar"]
    model.tok_emb.weight.data.copy_(state["tok_emb.weight"])
    model.pos_emb.weight.data.copy_(state["pos_emb.weight"])
    model.lm_head.weight.data.copy_(state["lm_head.weight"])
    model.lm_head.bias.data.copy_(state["lm_head.bias"])
    for i, layer in enumerate(model.layers):
        layer.load_encoder_layer(state, f"encoder.layers.{i % ckpt['num_layers']}")
    if depth != ckpt["num_layers"]:
        print(f"student depth {depth} from a {ckpt['num_layers']}-layer teacher "
              f"(layers cycled)")
    return model.to(device=device, dtype=dtype).eval(), ckpt


def build_module(ar_ckpt, device="cuda", dtype=torch.float32, num_layers=None):
    return build(ar_ckpt, device=device, dtype=dtype, num_layers=num_layers)[0]
