"""A HuggingFace-shaped causal LM backed by LlamaGen weights.

The point of this file is that nothing downstream of it needs to change. The
PTP repo drives its model through the HuggingFace contract -- keyword arguments
`input_ids` / `inputs_embeds` / `position_ids` / `attention_mask` /
`past_key_values`, an output object carrying `.logits` and `.past_key_values`,
a `BlockMask` for the nested-completion attention, and PEFT-wrappable linears.
LlamaGen satisfies none of that: its KV cache is preallocated inside each
attention module, its RoPE table is indexed by hand, and `forward` returns a
tuple. Adapting it here lets `ptp.lit`, `ptp.transformer` and `ptp_train` run
unmodified, so every PTP-specific detail comes from the tested implementation.

The class label rides in the sequence as a token: ids below `code_vocab` index
the codebook, ids at or above it index LlamaGen's class embedding. A sequence
is therefore `[class, t_0, ... t_255]`, which puts token t_j at position j+1 --
exactly the layout LlamaGen's own RoPE table assumes, and exactly the offset
the HuggingFace convention expects.
"""
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn.attention.flex_attention import BlockMask, flex_attention


@dataclass
class LlamaGenHFConfig:
    """The fields the PTP code and PEFT read off `model.config`."""
    hidden_size: int
    vocab_size: int
    num_attention_heads: int
    num_hidden_layers: int
    code_vocab: int
    num_classes: int
    cls_token_num: int
    # PEFT branches on this to special-case a few architectures; naming one that
    # needs no special-casing keeps it on the generic path.
    model_type: str = "llama"
    tie_word_embeddings: bool = False


@dataclass
class CausalOutput:
    logits: torch.Tensor
    past_key_values: object = None


class LlamaGenForCausalLM(nn.Module):
    """LlamaGen behind the HuggingFace causal-LM surface the PTP repo expects."""

    def __init__(self, gpt, apply_rotary_emb):
        super().__init__()
        self.gpt = gpt
        self._apply_rotary = apply_rotary_emb
        code_vocab = gpt.vocab_size
        self.config = LlamaGenHFConfig(
            hidden_size=gpt.config.dim,
            vocab_size=code_vocab,
            num_attention_heads=gpt.config.n_head,
            num_hidden_layers=gpt.config.n_layer,
            code_vocab=code_vocab,
            num_classes=gpt.num_classes,
            cls_token_num=gpt.cls_token_num,
        )
        # LlamaGen keeps its own preallocated caches; this path never uses them.
        for block in self.gpt.layers:
            block.attention.kv_cache = None

    # -- embeddings -------------------------------------------------------

    def get_input_embeddings(self):
        return self.gpt.tok_embeddings

    def embed(self, input_ids):
        """Route codebook ids to the token table and class ids to the label table."""
        code_vocab = self.config.code_vocab
        is_class = input_ids >= code_vocab
        codes = self.gpt.tok_embeddings(input_ids.clamp(max=code_vocab - 1))
        if not bool(is_class.any()):
            return codes
        labels = (input_ids - code_vocab).clamp(min=0, max=self.config.num_classes)
        # LabelEmbedder adds a length-1 axis and would drop the class at train
        # time; index the table directly and keep every position.
        class_embeds = self.gpt.cls_embedding.embedding_table(labels)
        return torch.where(is_class[..., None], class_embeds, codes)

    # -- attention --------------------------------------------------------

    def _attend(self, attn, x, freqs, mask, past_key_values, layer_idx, use_cache):
        bsz, seqlen, _ = x.shape
        head_dim = attn.head_dim
        kv_size = attn.n_kv_head * head_dim
        q, k, v = attn.wqkv(x).split([attn.dim, kv_size, kv_size], dim=-1)
        q = self._apply_rotary(q.view(bsz, seqlen, attn.n_head, head_dim), freqs)
        k = self._apply_rotary(k.view(bsz, seqlen, attn.n_kv_head, head_dim), freqs)
        v = v.view(bsz, seqlen, attn.n_kv_head, head_dim)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))

        if past_key_values is not None:
            k, v = past_key_values.update(k, v, layer_idx)
        repeat = attn.n_head // attn.n_kv_head
        k = k.repeat_interleave(repeat, dim=1)
        v = v.repeat_interleave(repeat, dim=1)

        if isinstance(mask, BlockMask):
            out = flex_attention(q, k, v, block_mask=mask)
        else:
            out = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=mask,
                is_causal=mask is None and k.shape[2] == q.shape[2],
            )
        out = out.transpose(1, 2).contiguous().view(bsz, seqlen, attn.dim)
        return attn.wo(out)

    # -- forward ----------------------------------------------------------

    def forward(self, input_ids=None, inputs_embeds=None, attention_mask=None,
                position_ids=None, past_key_values=None, use_cache=False,
                auxiliaries=None, **kwargs):
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("pass input_ids or inputs_embeds")
            inputs_embeds = self.embed(input_ids)
        h = inputs_embeds
        bsz, seqlen, _ = h.shape
        device = h.device

        # HuggingFace models allocate a cache when asked to use one and given
        # none, and hand it back on the output. PTP relies on that: `ar_forward`
        # calls with `use_cache=True` and no cache, then feeds the returned
        # object to the completion pass.
        if use_cache and past_key_values is None:
            from transformers import DynamicCache
            past_key_values = DynamicCache()

        if position_ids is None:
            offset = 0
            if past_key_values is not None:
                offset = past_key_values.get_seq_length()
            position_ids = torch.arange(offset, offset + seqlen, device=device)
        if position_ids.dim() == 2:
            # The PTP code sends per-sample positions; one row is enough here
            # because RoPE is applied per position, not per sample.
            position_ids = position_ids[0]

        table = self.gpt.freqs_cis
        if table.device != device:
            table = self.gpt.freqs_cis = table.to(device)
        freqs = table[position_ids.clamp(min=0, max=table.shape[0] - 1)]

        # A cached continuation of more than one position needs its causal mask
        # spelled out: the queries are shorter than the keys, so SDPA's
        # `is_causal` aligns to the wrong corner and the new positions would see
        # each other's future. Callers that pass a mask already handle this.
        if attention_mask is None and past_key_values is not None:
            cached = past_key_values.get_seq_length()
            if cached and seqlen > 1:
                q_pos = torch.arange(seqlen, device=device) + cached
                k_pos = torch.arange(cached + seqlen, device=device)
                attention_mask = (k_pos[None, :] <= q_pos[:, None])[None, None]

        for layer_idx, block in enumerate(self.gpt.layers):
            h = h + block.drop_path(self._attend(
                block.attention, block.attention_norm(h), freqs, attention_mask,
                past_key_values, layer_idx, use_cache))
            h = h + block.drop_path(block.feed_forward(block.ffn_norm(h)))

        logits = self.gpt.output(self.gpt.norm(h)).float()
        return CausalOutput(logits=logits, past_key_values=past_key_values)

    # -- knobs the PTP code may reach for --------------------------------

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        pass  # the image sequences are short; nothing to trade here

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      attention_mask=None, inputs_embeds=None, **kwargs):
        # PEFT's causal-LM wrapper stores a reference to this at construction.
        # Nothing here calls it -- PTP drives the model directly -- but it has
        # to exist for `get_peft_model` to accept the module.
        if past_key_values is not None:
            cached = past_key_values.get_seq_length()
            if cached:
                input_ids = input_ids[:, cached:]
        return {"input_ids": input_ids, "past_key_values": past_key_values,
                "attention_mask": attention_mask, "use_cache": True}

    @property
    def device(self):
        return self.gpt.tok_embeddings.weight.device


def build(llamagen_root, gpt_model="GPT-B", gpt_ckpt=None, image_size=256,
          downsample=16, codebook_size=16384, num_classes=1000, cls_token_num=1,
          dtype=torch.bfloat16, device="cuda"):
    """Load LlamaGen weights and wrap them. Returns (model, seq_len)."""
    import sys
    sys.path.insert(0, str(llamagen_root))
    from autoregressive.models.gpt import GPT_models, precompute_freqs_cis_2d, apply_rotary_emb

    seq_len = (image_size // downsample) ** 2
    gpt = GPT_models[gpt_model](
        vocab_size=codebook_size, block_size=seq_len,
        num_classes=num_classes, cls_token_num=cls_token_num, model_type="c2i",
    ).to(device=device, dtype=dtype)
    if gpt_ckpt is not None:
        ckpt = torch.load(gpt_ckpt, map_location="cpu")
        for key in ("model", "module", "state_dict"):
            if key in ckpt:
                ckpt = ckpt[key]
                break
        missing, unexpected = gpt.load_state_dict(ckpt, strict=False)
        if missing or unexpected:
            print(f"load_state_dict: {len(missing)} missing, {len(unexpected)} unexpected")

    # freqs_cis is a plain attribute, so `.to()` skips it and `setup_caches`
    # rebuilds it on whatever device happens to be current. The CPU and GPU
    # tables differ by one float32 ULP, which is enough to move a third of the
    # narrow bins here. Pin one canonical version.
    with torch.device(device):
        grid = int(seq_len ** 0.5)
        gpt.freqs_cis = precompute_freqs_cis_2d(
            grid, gpt.config.dim // gpt.config.n_head,
            gpt.config.rope_base, cls_token_num)
    gpt.eval()
    return LlamaGenForCausalLM(gpt, apply_rotary_emb), seq_len


def build_module(llamagen_root, gpt_model="GPT-B", gpt_ckpt=None, image_size=256,
                 **kwargs):
    """Return just the model, for configs that instantiate it by target."""
    model, _ = build(llamagen_root, gpt_model=gpt_model, gpt_ckpt=gpt_ckpt,
                     image_size=image_size, **kwargs)
    return model
