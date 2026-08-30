"""Three samplers on the MNIST ladder, same labels, same seed.

The PTP student's 1.4233 is a VAE reconstruction term: it is scored with u drawn
from the interval the true token owns, so it already carries the answer. That is
not the same quantity as the 2.0618 and 1.4206 it was put beside, which are
model likelihoods. Whether two forwards actually reach four forwards' quality is
a question about samples, so take samples.

  (1,8) parallel      2 forwards, every token of a scale drawn independently
  (1,8) + PTP         2 forwards, every token drawn from its own u
  (1,2,4,8) parallel  4 forwards, the quality target

The student reproduces the unguided teacher -- its edges were built with
cond_drop_rate at 0 -- so all three run at cfg 1.0 or the comparison is between
different distributions.
"""
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, "/home/mengy13/ptp")
os.environ.setdefault("VAR_ROOT", "/home/mengy13/VAR")
device = "cuda"
torch.set_grad_enabled(False)
N = 100
PER_CLASS = 10
SEED = 0


def load_ladder(patch, out_dir):
    """Return the module, and the vae the VAR itself holds -- loaded.

    VAR reaches its codebook through vae_quant_proxy, a reference to the vae it
    was constructed with. Building a second vae alongside and loading only that
    one leaves the sampler decoding through a randomly initialised codebook,
    which comes out as noise.
    """
    os.environ["PATCH"] = patch
    os.environ["OUT"] = out_dir
    for m in [k for k in sys.modules if k.startswith("image_ptp.mnist_var")]:
        del sys.modules[m]
    import importlib
    mv = importlib.import_module("image_ptp.mnist_var")
    importlib.reload(mv)
    mv.ensure_process_group()
    vae, var = mv.build_var()
    vae.load_state_dict(torch.load(Path(out_dir) / "vqvae.pt", map_location="cpu"))
    var.load_state_dict(torch.load(Path(out_dir) / "var.pt", map_location="cpu"))
    vae.eval()
    var.eval()
    var.cond_drop_rate = 0.0
    return mv, vae, var


@torch.no_grad()
def sample_ptp(mv, vae, student, labels):
    """One forward per scale; each slot reads its own u and takes the argmax.

    Scale 0's positions cannot attend to scale 1 under the within-scale causal
    mask, so the first pass may hand scale 1 whatever it likes -- its own output
    is unaffected. The second pass gets the real input, built from the token
    scale 0 just produced.
    """
    B = labels.shape[0]
    L = student.L
    bounds, cur = [], 0
    for pn in mv.PATCH:
        bounds.append((cur, cur + pn * pn, pn))
        cur += pn * pn
    g = torch.Generator(device=device).manual_seed(SEED)
    u = torch.rand(B, L, generator=g, device=device)

    tokens = torch.zeros(B, L, dtype=torch.long, device=device)
    for si, (a, b, pn) in enumerate(bounds):
        if si == 0:
            x_in = torch.zeros(B, L - student.first_l, vae.Cvae, device=device)
        else:
            gt = [tokens[:, s0:s1] for s0, s1, _ in bounds[:si]]
            x_in = vae.quantize.idxBl_to_var_input(gt)
            if x_in is None:
                x_in = torch.zeros(B, 0, vae.Cvae, device=device)
        logits = student(labels, x_in, u=u).float()
        tokens[:, a:b] = logits[:, a:b].argmax(-1)
    return [tokens[:, a:b] for a, b, _ in bounds]


@torch.no_grad()
def decode(mv, vae, per_scale):
    """Tokens per scale -> image, the way the quantiser accumulates them."""
    import torch.nn.functional as F
    B = per_scale[0].shape[0]
    H = W = mv.PATCH[-1]
    q = vae.quantize
    f_hat = torch.zeros(B, vae.Cvae, H, W, device=device)
    SN = len(mv.PATCH)
    for si, pn in enumerate(mv.PATCH):
        h = q.embedding(per_scale[si]).transpose(1, 2).view(B, vae.Cvae, pn, pn)
        if pn != H:
            h = F.interpolate(h, size=(H, W), mode="bicubic")
        f_hat = f_hat + q.quant_resi[si / (SN - 1)](h)
    return vae.fhat_to_img(f_hat)


def main():
    from torchvision.utils import save_image
    OUT = Path("/home/mengy13/ptp-image-results/var_ptp_samples")
    OUT.mkdir(parents=True, exist_ok=True)
    labels = torch.arange(10, device=device).repeat_interleave(PER_CLASS)

    # --- (1,8) parallel and (1,8) + PTP ---
    mv8, vae8, var8 = load_ladder("1,8", "/home/mengy13/ptp-image-results/mnist_var_r8")
    torch.manual_seed(SEED)
    save_image(var8.autoregressive_infer_cfg(B=N, label_B=labels, cfg=1.0, top_k=100,
                                             top_p=0.95),
               OUT / "a_parallel_1_8.png", nrow=PER_CLASS)
    print("wrote a_parallel_1_8.png (2 forwards, independent within a scale)", flush=True)
    del var8
    torch.cuda.empty_cache()

    from image_ptp.within_scale_var import build_ptp_student
    student = build_ptp_student(vae8, mv8.PATCH, adapter="binary",
                                num_classes=mv8.NUM_CLASSES, device=device)
    student.load_state_dict(torch.load(Path(mv8.OUT) / "var_ptp_student.pt",
                                       map_location="cpu"))
    student.eval()
    student.cond_drop_rate = 0.0
    save_image(decode(mv8, vae8, sample_ptp(mv8, vae8, student, labels)),
               OUT / "b_ptp_1_8.png", nrow=PER_CLASS)
    print("wrote b_ptp_1_8.png (2 forwards, each slot reads its own u)", flush=True)
    del student, vae8
    torch.cuda.empty_cache()

    # --- (1,2,4,8) parallel, the target ---
    mv2, vae2, var2 = load_ladder("1,2,4,8", "/home/mengy13/ptp-image-results/mnist_var_r2")
    torch.manual_seed(SEED)
    save_image(var2.autoregressive_infer_cfg(B=N, label_B=labels, cfg=1.0, top_k=100,
                                             top_p=0.95),
               OUT / "c_parallel_1_2_4_8.png", nrow=PER_CLASS)
    print("wrote c_parallel_1_2_4_8.png (4 forwards, the target)", flush=True)

    # A correctly loaded codebook reconstructs; a random one does not. Check rather
    # than eyeballing three grids for noise.
    xs, _ = next(iter(mv2.mnist_loader(False, 64, 0)))
    rec, _, _ = vae2(xs.to(device))
    psnr = 10 * torch.log10(4.0 / torch.nn.functional.mse_loss(rec, xs.to(device)))
    print(f"sanity: the vae VAR samples through reconstructs at {float(psnr):.1f} dB")
    assert float(psnr) > 15, "the codebook in use is not the trained one"
    print("VARPTP_SAMPLES_DONE", flush=True)


if __name__ == "__main__":
    main()
