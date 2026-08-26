"""Encode real ImageNet images into LlamaGen codes and cache the teacher's bin edges.

`pregenerate.py` samples its sequences from the teacher, which the PTP paper
prefers and which was the only option on lucy. It has a hard ceiling: however
many sequences you generate is how many you get, and generating 49152 took six
hours. At batch 4 that is 12288 steps per epoch, and the run before it -- 3840
sequences, 17 passes -- memorised its data outright (train/correct 0.886 against
a validation loss that climbed from step 4565 on).

ImageNet removes the ceiling. The teacher stays frozen and still defines
everything that matters: the bin edges come from its guided, truncated
distribution exactly as before, so `u` inverts the same CDF. Only the tokens
change, from teacher samples to encodings of real photographs -- which is what
the MNIST and CIFAR arms have always done.

The guidance scale, truncation and temperature are baked in, as in
`pregenerate.py`: changing any of them changes which distribution `u` inverts,
so the file has to be regenerated.

Shardable: pass --shard i --num-shards n to split the class list n ways and
merge the parts afterwards with --merge.

Usage:
    python prepare_imagenet_tokens.py --llamagen-root ~/LlamaGen \
        --gpt-ckpt ~/LlamaGen/pretrained_models/c2i_B_256.pt \
        --vq-ckpt ~/LlamaGen/pretrained_models/vq_ds16_c2i.pt \
        --imagenet /extra/ucibdl1/shared/data/imagenet/train \
        --per-class 64 --out ~/ptp-image-results/in_tokens_shard0.pt
"""
import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--llamagen-root", type=str, default="/home/mengy13/LlamaGen")
    p.add_argument("--gpt-model", type=str, default="GPT-B")
    p.add_argument("--gpt-ckpt", type=str, required=True)
    p.add_argument("--vq-ckpt", type=str, required=True)
    p.add_argument("--vq-model", type=str, default="VQ-16")
    p.add_argument("--imagenet", type=str, required=True)
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--per-class", type=int, default=64,
                   help="images per class; 1000 classes, so 64 gives 64000")
    p.add_argument("--enc-batch", type=int, default=64)
    p.add_argument("--edge-batch", type=int, default=8,
                   help="smaller: the logits are (B, S, 16384)")
    p.add_argument("--cfg-scale", type=float, default=4.0)
    p.add_argument("--top-k", type=int, default=100)
    p.add_argument("--top-p", type=float, default=0.999)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--merge", type=str, nargs="*", default=None,
                   help="merge these shard files into --out and exit")
    return p.parse_args()


def merge(paths, out):
    parts = [torch.load(Path(p).expanduser(), map_location="cpu") for p in paths]
    payload = {
        k: torch.cat([p[k] for p in parts])
        for k in ("tokens", "labels", "left_bin_edges", "right_bin_edges")
    }
    payload["config"] = dict(parts[0]["config"])
    payload["config"]["num_images"] = int(payload["tokens"].shape[0])
    # Shards are class-contiguous, so a run that reads the head of the file
    # would see only the first few classes. The datamodule holds out the first
    # val_split sequences, which makes that a real problem rather than a stylistic
    # one -- shuffle once, here, where it costs nothing.
    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(payload["tokens"].shape[0], generator=g)
    for k in ("tokens", "labels", "left_bin_edges", "right_bin_edges"):
        payload[k] = payload[k][perm]
    out = Path(out).expanduser()
    torch.save(payload, out)
    print(f"merged {len(paths)} shards -> {out} "
          f"({payload['tokens'].shape[0]} sequences, "
          f"{out.stat().st_size / 1e6:.1f} MB)")


def main():
    args = parse_args()
    if args.merge:
        return merge(args.merge, args.out)

    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    root = Path(args.llamagen_root).expanduser()

    from llamagen_ptp import LlamaGenPTP, load_llamagen
    gpt, seq_len = load_llamagen(root, args.gpt_model, args.gpt_ckpt,
                                 image_size=args.image_size, device=device)
    model = LlamaGenPTP(gpt, lora_rank=4).to(device).eval()

    sys.path.insert(0, str(root))
    from tokenizer.tokenizer_image.vq_model import VQ_models
    vq = VQ_models[args.vq_model](codebook_size=16384, codebook_embed_dim=8)
    vq_ckpt = torch.load(args.vq_ckpt, map_location="cpu")
    vq.load_state_dict(vq_ckpt["model"] if "model" in vq_ckpt else vq_ckpt)
    vq = vq.to(device).eval()

    from torchvision import transforms
    from PIL import Image
    # LlamaGen's own preprocessing: centre crop to a square, then resize.
    tfm = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
    ])

    # Sorted wnid order is the standard ImageNet class index, which is what the
    # c2i checkpoint was conditioned on. Do not reorder.
    data_root = Path(args.imagenet).expanduser()
    classes = sorted(d.name for d in data_root.iterdir() if d.is_dir())
    assert len(classes) == 1000, f"expected 1000 classes, found {len(classes)}"
    mine = classes[args.shard::args.num_shards]
    print(f"shard {args.shard}/{args.num_shards}: {len(mine)} classes, "
          f"{args.per_class} images each", flush=True)

    paths, labels = [], []
    for name in mine:
        idx = classes.index(name)
        files = sorted((data_root / name).iterdir())[:args.per_class]
        paths.extend(files)
        labels.extend([idx] * len(files))
    labels = torch.tensor(labels, dtype=torch.long)
    print(f"{len(paths)} images", flush=True)

    all_tokens = []
    started = time.time()
    for start in range(0, len(paths), args.enc_batch):
        chunk = paths[start:start + args.enc_batch]
        images = torch.stack([tfm(Image.open(p).convert("RGB")) for p in chunk])
        _, _, info = vq.encode(images.to(device))
        # info[2] is the flat index map; reshape to (B, seq_len) in raster order.
        all_tokens.append(info[2].reshape(len(chunk), -1).cpu())
        done = start + len(chunk)
        if (start // args.enc_batch) % 20 == 0 or done == len(paths):
            rate = done / (time.time() - started)
            print(f"encoded {done}/{len(paths)}  {rate:.1f} img/s  "
                  f"eta {(len(paths) - done) / max(rate, 1e-6) / 60:.1f} min", flush=True)
    tokens = torch.cat(all_tokens)
    assert tokens.shape[1] == seq_len, (tokens.shape, seq_len)

    lefts, rights = [], []
    started = time.time()
    for start in range(0, tokens.shape[0], args.edge_batch):
        stop = min(start + args.edge_batch, tokens.shape[0])
        left, right = model.bin_edges(
            labels[start:stop].to(device), tokens[start:stop].to(device),
            0, seq_len - 1, cfg_scale=args.cfg_scale, top_k=args.top_k,
            top_p=args.top_p, temperature=args.temperature)
        lefts.append(left.cpu())
        rights.append(right.cpu())
        if (stop // args.edge_batch) % 100 == 0 or stop == tokens.shape[0]:
            rate = stop / (time.time() - started)
            print(f"bin edges {stop}/{tokens.shape[0]}  {rate:.1f} img/s", flush=True)
    left, right = torch.cat(lefts), torch.cat(rights)
    width = right - left
    print(f"bin width: mean={width.mean():.4f} median={width.median():.4f} "
          f"min={width.min():.2e}")
    assert (width >= -1e-6).all(), "negative bin width means the edges are misaligned"

    # Truncation is the reason this can fail where the teacher-sampled data could
    # not: top-k 100 keeps only the teacher's own 100 favourites, and a real
    # photograph may well contain a code the teacher would never have proposed.
    # Such a token has zero probability, so no u recovers it, and training on it
    # would be asking the student to invert a CDF that does not contain the
    # answer. Measure it here rather than discovering it as a loss plateau.
    dead = (width <= 1e-8)
    print(f"tokens outside the teacher's truncated support: "
          f"{dead.float().mean():.4f} ({int(dead.sum())} of {dead.numel()})")

    payload = {
        "tokens": tokens.to(torch.int32),
        "labels": labels.to(torch.int32),
        "left_bin_edges": left,
        "right_bin_edges": right,
        "config": {
            "source": f"imagenet:{data_root.name}", "per_class": args.per_class,
            "cfg_scale": args.cfg_scale, "top_k": args.top_k, "top_p": args.top_p,
            "temperature": args.temperature, "gpt_model": args.gpt_model,
            "seq_len": seq_len, "num_images": int(tokens.shape[0]),
            "shard": args.shard, "num_shards": args.num_shards,
            "dead_fraction": float(dead.float().mean()),
        },
    }
    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
