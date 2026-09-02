"""Feature extractors for Frechet distance, including the standard one.

Everything in this project has been scored with a small CNN trained here, whose
128-d penultimate layer defined the space the distance was measured in. That has
two costs. It is not comparable to any published number, so "is 1200 bad?" had no
answer. And it is unstable: retraining the classifier once reordered two arms
that differ by 8%, because the metric itself moved.

InceptionV3's ImageNet weights are already cached on this machine, so the
standard 2048-d pool3 feature is available offline. Values still will not match
the TensorFlow reference exactly -- torchvision's port differs slightly, as every
PyTorch FID implementation notes -- but they are on the usual scale and can be
read against the literature.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)


class InceptionFeatures(nn.Module):
    """2048-d pool3 features, the standard FID representation.

    Takes images in [-1, 1] with 1 or 3 channels at any size; grayscale is
    repeated to three channels and everything is resized to 299 bilinearly,
    which is what the reference implementations do.
    """

    def __init__(self, device="cuda"):
        super().__init__()
        from torchvision.models import inception_v3, Inception_V3_Weights
        net = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1,
                           transform_input=False)
        net.fc = nn.Identity()
        self.net = net.eval().to(device)
        for p in self.net.parameters():
            p.requires_grad_(False)
        self.register_buffer("mean", torch.tensor(_MEAN).view(1, 3, 1, 1).to(device))
        self.register_buffer("std", torch.tensor(_STD).view(1, 3, 1, 1).to(device))
        self.dim = 2048

    @torch.no_grad()
    def forward(self, x):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        x = (x.clamp(-1, 1) + 1) / 2
        x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)
        return self.net((x - self.mean) / self.std)


def frechet(a, b):
    """Frechet distance between Gaussians fitted to two feature sets."""
    mu_a, mu_b = a.mean(0), b.mean(0)
    ca, cb = torch.cov(a.T.double()), torch.cov(b.T.double())
    diff = (mu_a - mu_b).double()
    ea, va = torch.linalg.eigh(ca)
    half = va @ torch.diag(ea.clamp_min(0).sqrt()) @ va.T
    em = torch.linalg.eigvalsh(half @ cb @ half).clamp_min(0)
    return float(diff.dot(diff) + ca.trace() + cb.trace() - 2 * em.sqrt().sum())


def check_sample_count(n, dim):
    """Frechet distance needs more samples than feature dimensions.

    With 2048-d Inception features and n=2000 the covariance is rank-deficient,
    and the estimate is not merely biased: the same tokeniser and ladder scored
    9.579 at n=2000 and 2.251 at n=10000. Anything at or below the feature
    dimension is refused rather than reported.
    """
    if n <= dim:
        raise ValueError(
            f"{n} samples for {dim}-d features: the covariance is singular and "
            f"the distance is meaningless. Use at least ~5x the dimension.")
    if n < 5 * dim:
        print(f"  WARNING: {n} samples for {dim}-d features is thin; "
              f"~{5 * dim} would be safer", flush=True)
