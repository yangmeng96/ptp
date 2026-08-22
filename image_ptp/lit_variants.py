"""A LightningModule subclass that exposes the auxiliary sampling distribution.

`ptp.lit` hard-codes Beta(0.3, 0.3) for training draws of u -- a U-shaped
distribution that concentrates near the bin edges, where the student's decision
flips and supervision is worth most. The `ptp-vqvae` implementation that this
work is calibrated against instead draws uniformly inside the bin, and reaches a
comparable lift, so which is better is an open question rather than a settled
one.

Subclassing keeps the repo untouched: point a config's `_target_` here and set
`aux_sampling`.
"""
import torch

from ptp.lit import ParallelSamplingLightningModule


class ConfigurableAuxSampling(ParallelSamplingLightningModule):
    def __init__(self, *args, aux_sampling: str = "beta",
                 beta_concentration: float = 0.3, **kwargs):
        super().__init__(*args, **kwargs)
        if aux_sampling not in ("beta", "uniform"):
            raise ValueError(f"aux_sampling must be beta or uniform, got {aux_sampling}")
        self.aux_sampling = aux_sampling
        self.beta_concentration = beta_concentration

    def sample_auxiliaries(self, left_bin_edges, right_bin_edges, eval):
        # Evaluation always draws uniformly, because that is what inference does.
        if eval or self.aux_sampling == "uniform":
            z = torch.rand(left_bin_edges.shape, device=left_bin_edges.device,
                           dtype=torch.float32)
        else:
            conc = torch.tensor(self.beta_concentration, dtype=torch.float32)
            if left_bin_edges.device.type == "mps":
                sample_on = conc
            else:
                sample_on = conc.to(left_bin_edges.device)
            z = torch.distributions.Beta(sample_on, sample_on).sample(
                left_bin_edges.shape).to(left_bin_edges.device)
        return left_bin_edges + (right_bin_edges - left_bin_edges) * z
