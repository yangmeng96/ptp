"""O-PTP with the student's own next-token distribution pinned to the teacher's.

The student is trained to answer "given u, which token would the teacher have
sampled", which means inverting the teacher's CDF at u. Nothing in that loss
constrains the student's own autoregressive head, and under a full finetune the
shared weights drift: measured on MNIST, the student's cumulative probability
for the true token moves a median of 0.076-0.136 away from the teacher's, and
35-52% of positions move further than half their own bin. A student inverting
against its own drifted CDF lands on the neighbouring token exactly there.

Whether that drift *causes* the errors is a hypothesis, not an established
fact -- the supporting evidence is a three-point correlation across runs that
differ in several ways. This class exists to test it by intervention rather
than by more observation: add a KL term pulling the student's AR distribution
back onto the teacher's and see whether acceptance improves.

`drift_weight: 0` reproduces the unmodified objective, so the arms differ in
one term only.
"""
import torch
import torch.nn.functional as F

from image_ptp.lit_variants import ConfigurableAuxSampling


class DriftRegularised(ConfigurableAuxSampling):
    def __init__(self, *args, drift_weight: float = 0.0, teacher_ckpt: str | None = None,
                 teacher_layers: int | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.drift_weight = drift_weight
        self._teacher = None
        self._teacher_ckpt = teacher_ckpt
        self._teacher_layers = teacher_layers
        if drift_weight > 0 and teacher_ckpt is None:
            raise ValueError("drift_weight > 0 needs a teacher_ckpt to pull towards")

    def _frozen_teacher(self, device):
        if self._teacher is None:
            from image_ptp.vqvae_ar_hf import build_module
            teacher = build_module(self._teacher_ckpt, device=device, dtype=torch.float32,
                                   num_layers=self._teacher_layers)
            for p in teacher.parameters():
                p.requires_grad_(False)
            # Held outside the module tree so it never lands in the checkpoint or
            # the optimiser.
            self._teacher = [teacher.eval()]
        return self._teacher[0]

    def forward(self, batch, batch_idx=None, eval=False, return_outputs=False):
        metrics = super().forward(batch, batch_idx, eval=eval, return_outputs=return_outputs)
        if self.drift_weight <= 0:
            return metrics

        input_ids = batch["input_ids"]
        teacher = self._frozen_teacher(input_ids.device)
        with self.model.enable_adapters(enabled=True):
            student_logits = self.model.model(input_ids=input_ids).logits[:, :-1]
        with torch.no_grad():
            teacher_logits = teacher(input_ids=input_ids).logits[:, :-1]

        # KL(teacher || student): the teacher is the reference the auxiliaries
        # were derived from, so it belongs in the first argument.
        kl = F.kl_div(
            student_logits.float().log_softmax(-1),
            teacher_logits.float().log_softmax(-1),
            log_target=True, reduction="batchmean",
        )
        metrics["l_drift"] = kl
        metrics["loss"] = metrics["loss"] + self.drift_weight * kl
        return metrics
