"""Bin width -- the teacher's probability of the true token -- per token file.

This is the quantity the paper ties auxiliary quality to: "the quality of the
auxiliaries depends directly on the AR model's accuracy". A narrow bin means the
student must resolve u to that fraction of [0,1] to pick the right token, so it
sets the difficulty of everything downstream.

Both CIFAR teachers were first compared on train tokens, which both were trained
on, and the bigger one looked eight times better. A bigger model memorises more,
so that comparison cannot separate skill from memorisation; pass the held-out
file alongside the training one and the gap between the two columns says which
it was.

    python -m image_ptp.bin_width_report name=train.pt,test.pt ...
"""
import sys
from pathlib import Path

import torch


def stats(path):
    d = torch.load(path, map_location="cpu")
    w = (d["right_bin_edges"] - d["left_bin_edges"]).flatten().float()
    g = torch.Generator().manual_seed(0)
    s = w[torch.randperm(len(w), generator=g)[:200_000]]
    q = torch.quantile(s, torch.tensor([0.25, 0.5, 0.75]))
    return dict(median=float(q[1]), mean=float(w.mean()), p25=float(q[0]),
                p75=float(q[2]), tiny=float((w < 0.01).float().mean()), n=len(w))


def main(argv):
    print(f"{'teacher':<16} {'split':<6} {'median':>8} {'mean':>8} {'p25':>8} "
          f"{'p75':>8} {'<0.01':>7} {'n':>8}")
    for arg in argv:
        name, files = arg.split("=", 1)
        for split, f in zip(("train", "test"), files.split(",")):
            if not Path(f).exists():
                print(f"{name:<16} {split:<6}   <missing: {f}>")
                continue
            s = stats(f)
            print(f"{name:<16} {split:<6} {s['median']:8.4f} {s['mean']:8.4f} "
                  f"{s['p25']:8.4f} {s['p75']:8.4f} {s['tiny']:7.3f} "
                  f"{s['n']/1e6:7.1f}M")
    print("BINWIDTH_DONE")


if __name__ == "__main__":
    main(sys.argv[1:])
