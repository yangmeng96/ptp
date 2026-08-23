"""Categorical Parallel Token Prediction on top of the repo's O-PTP machinery.

O-PTP hands slot k its own auxiliary u_k, so the slot's output collapses to the
single token u_k selects and the network has to perform the inverse-CDF lookup
internally. Measured on MNIST that lookup is where the remaining error lives:
the teacher's logits with an *external* pick score 1.000, while the best
internalised student reaches 0.873.

C-PTP withholds u_k. Slot k sees only u_i..u_{k-1}, so its output stays a full
distribution -- by Theorem 2 exactly Q(t_k | t_<k) -- and the pick happens in
code afterwards. The operation that costs 12.7% is simply no longer the
network's job.

Withholding cannot be done with the attention mask alone: u_k would be the
slot's own input embedding, sitting in its residual stream where no mask
reaches. The auxiliaries are shifted right by one instead, so the slot
predicting t_k carries u_{k-1}. With that shift the repo's mask and loss are
already correct -- `kv_pos <= q_pos` now means "up to u_{k-1}", and the
cross-entropy against t_k is the paper's Eq. 11.

The first slot of a block has no predecessor inside the block. Its target is
Q(t_i | t_<i), which the prefix alone determines, so it receives a constant
placeholder rather than an auxiliary.
"""
import torch

from image_ptp.lit_variants import ConfigurableAuxSampling


class CategoricalPTP(ConfigurableAuxSampling):
    def __init__(self, *args, first_slot_value: float = 0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.first_slot_value = first_slot_value

    def sample_auxiliaries(self, left_bin_edges, right_bin_edges, eval):
        u = super().sample_auxiliaries(left_bin_edges, right_bin_edges, eval)
        # (B, N, L): shift along the within-block axis so slot l carries u_{l-1}.
        shifted = torch.empty_like(u)
        shifted[..., 0] = self.first_slot_value
        shifted[..., 1:] = u[..., :-1]
        return shifted
