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

class SphereInitAuxSampling(ConfigurableAuxSampling):
    """ConfigurableAuxSampling with the u embedding table set to the sphere.

    The Voronoi cells make neighbouring auxiliary ids select geometrically
    related tokens, but a randomly initialised RoundingEmbedding table does not
    know two ids are neighbours -- the geometry lives in the assignment and
    never reaches the model. The upstream scheme has the model read the frozen
    sphere vectors themselves. This rebuilds the same sphere (same seed, same
    construction as prepare_voronoi_tokens.py) and writes it into the table;
    the mask row keeps its random initialisation, as upstream's mask_auxiliary
    is learned. `freeze_sphere` pins the cell rows the way upstream freezes the
    sphere; the mask row stays trainable either way.
    """

    def __init__(self, *args, sphere_K: int = 512, sphere_seed: int = 0,
                 freeze_sphere: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        import torch.nn.functional as F
        table = self.model.u_embed.embedding.weight
        K, dim = sphere_K, table.shape[1]
        assert table.shape[0] == K + 1, (
            f"expected {K}+1 bins (cells + mask), got {table.shape[0]}")
        g = torch.Generator(device="cpu").manual_seed(sphere_seed)
        sphere = F.normalize(torch.randn(K, dim, generator=g), dim=-1)
        with torch.no_grad():
            table[:K] = sphere.to(table.device, table.dtype)
        if freeze_sphere:
            # One parameter cannot be half-frozen; a hook zeroes the cell rows'
            # gradient instead, leaving the mask row trainable.
            def clamp(grad):
                grad = grad.clone()
                grad[:K] = 0
                return grad
            table.register_hook(clamp)
        print(f"u embedding: {K} rows set to the seed-{sphere_seed} sphere, "
              f"mask row learned, frozen={freeze_sphere}")

