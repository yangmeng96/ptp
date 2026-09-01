"""Is the Voronoi cell assignment a deterministic function of the teacher's Q?

The PTP student has to learn "given this cell id and this prefix, which code owns
the cell". That is only learnable if the assignment is a fixed function of Q --
the prefix determines Q, and Q must then determine the cells. If the solver
returns a different assignment for the same Q on a second call, or depends on how
many positions happen to be batched with it, then the training file and the test
file were cut under two different rules and no student can transfer between them.

Regenerating cifar_voro4k at batch 8 rather than 32 produced an assignment that
agrees with the original on 0.14% of positions -- barely above the 0.024% two
random assignments would share -- while reproducing its quotas to seven figures.
That is the symptom this checks for.
"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", default="/home/mengy13/ptp-image-results/cifar_tokens.pt")
    ap.add_argument("--ar-ckpt", default="/home/mengy13/ptp-vqvae/checkpoints/ar_cifar10_raster.pt")
    ap.add_argument("--K", type=int, default=4096)
    ap.add_argument("--top-m", type=int, default=256)
    ap.add_argument("--positions", type=int, default=256)
    args = ap.parse_args()

    import image_ptp.prepare_voronoi_tokens as pv

    dev = "cuda"
    d = torch.load(args.tokens, map_location="cpu")
    # Q for a fixed slice of positions, taken straight from the stored edges so
    # this probe does not depend on re-running the teacher.
    torch.manual_seed(0)
    P = args.positions
    V = int(d["config"]["num_codes"])
    Q = torch.rand(P, V, device=dev)
    Q = Q / Q.sum(1, keepdim=True)

    sphere = pv.unit_sphere(args.K, 256, seed=0, device=dev) \
        if hasattr(pv, "unit_sphere") else None
    if sphere is None:
        print("no unit_sphere helper; falling back to a stored sphere")
        sphere = torch.load("/home/mengy13/ptp-image-results/cifar_voro4k_v2_train.pt",
                            map_location=dev)["sphere"]
    head = torch.randn(V, sphere.shape[1], device=dev)
    head = head / head.norm(dim=1, keepdim=True)
    S = head @ sphere.T                                    # (V, K)

    ref = pv.load_referee()
    def run(lo, hi):
        return pv.assign_cells_referee(Q[lo:hi], S, args.K, args.top_m, ref)

    a = run(0, P)
    b = run(0, P)
    print(f"same call twice, identical: {torch.equal(a, b)}  "
          f"agreement {float((a == b).float().mean()):.4f}")

    chunks = torch.cat([run(i, min(i + 8, P)) for i in range(0, P, 8)])
    print(f"batched by 8 vs all-at-once, identical: {torch.equal(a, chunks)}  "
          f"agreement {float((a == chunks).float().mean()):.4f}")

    chunk32 = torch.cat([run(i, min(i + 32, P)) for i in range(0, P, 32)])
    print(f"batched by 32 vs by 8, identical: {torch.equal(chunk32, chunks)}  "
          f"agreement {float((chunk32 == chunks).float().mean()):.4f}")
    print("PROBE_DONE")


if __name__ == "__main__":
    main()
