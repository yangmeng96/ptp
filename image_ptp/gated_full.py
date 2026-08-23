"""Full finetuning restricted to the auxiliary positions.

The repo's LoRA setup is really a bolt-on: `ar_forward` runs the prefix with the
adapters disabled and only the auxiliary block sees them, and at inference
`GatedLinearLoraMerged` subtracts the update back off the token positions. The
prefix representation is therefore bit-identical to the base model's, which is
what lets the draft and the verifier share a KV cache -- the speedup mechanism.

Plain full finetuning throws that away: every position goes through the changed
weights, so the prefix representation moves too. Measured on MNIST, the
student's cumulative probability for the true token drifts a median of 0.076
from the teacher's, and feeding its own logits to an external inverse-CDF pick
then scores 0.536 against the teacher's 1.000.

This class keeps the two-pass structure and replaces the low-rank update with a
full trainable copy: frozen weights for the prefix, trained weights for the
auxiliaries. Zero prefix drift, no rank ceiling.
"""
import copy

import torch

from ptp.transformer import MixedTransformerModel


class GatedFullTransformerModel(MixedTransformerModel):
    def __init__(self, **kwargs):
        if kwargs.get("lora_config") is not None:
            raise ValueError("gated full finetuning replaces LoRA; drop lora_config")
        super().__init__(**kwargs)
        frozen = copy.deepcopy(self.model)
        for p in frozen.parameters():
            p.requires_grad_(False)
        # Held in a list so it stays out of the module tree: it must not be
        # trained, saved, or counted among the parameters the optimiser sees.
        self._frozen = [frozen.eval()]

    def frozen_model(self, device=None):
        model = self._frozen[0]
        if device is not None and next(model.parameters()).device != device:
            model = self._frozen[0] = model.to(device)
        return model

    def ar_forward(self, input_ids, attention_mask=None):
        """The prefix pass, on the untouched weights."""
        input_ids = self._replace_ignore_index(input_ids)
        with torch.no_grad():
            return self.frozen_model(input_ids.device)(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
                auxiliaries=None,
            )
