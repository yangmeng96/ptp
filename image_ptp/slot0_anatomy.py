"""What bounds slot 0, and does the auxiliary representation touch it?

Slot 0 sits at 0.53 across every CIFAR variant tried -- vocabulary order, three
reorderings, a discrete table, Voronoi cells -- moving between 0.465 and 0.545
while slots 2-5 moved 26-30%. It is the one position that needs no inference
about its blockmates: the whole prefix is real tokens, so the teacher's
distribution for it is fully determined and only the auxiliary has to be read.

Three measurements separate the two things that could bound it.

  top-j        If the auxiliary narrows the candidates but does not pin them,
               top-1 stays flat while top-j rises. Voronoi's cells are built so
               that an auxiliary points at its token's output embedding, which
               is a hint, not an answer.

  error        Where a wrong prediction lands, measured as cosine distance in
               output-embedding space against the distance to a random code.
               A geometric hint that works puts errors on near codes.

  by mass      Accuracy against the mass the true token holds -- interval width
               for the scalar scheme, cell size for Voronoi. Rising means the
               student knows the distribution and misreads the auxiliary near
               boundaries; flat means it does not know the distribution, which
               no auxiliary representation can fix.
"""
import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ar-ckpt", type=str,
                   default="/home/mengy13/ptp-vqvae/checkpoints/ar_cifar10_raster.pt")
    p.add_argument("--arms", type=str, nargs="+", required=True,
                   help="name=ckpt=data=adapter=kwargs quintuples")
    p.add_argument("--block-len", type=int, default=8)
    p.add_argument("--num-layers", type=int, default=8)
    p.add_argument("--images", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--topj", type=int, nargs="*", default=[1, 2, 4, 16, 64])
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


@torch.no_grad()
def anatomy(model, tokens, left, right, aux_ids, bos, block_len, topj, positions,
            device, batch_size, seed):
    torch.manual_seed(seed)
    seq_len = tokens.shape[1]
    starts = list(range(1, seq_len - block_len + 1, block_len))
    V = positions.shape[0]
    hits = {j: 0 for j in topj}
    n = 0
    err_cos, rand_cos = [], []
    mass_all, ok_all = [], []
    voronoi = aux_ids is not None

    for s in range(0, tokens.shape[0], batch_size):
        t = tokens[s:s + batch_size].to(device)
        l = left[s:s + batch_size].to(device)
        r = right[s:s + batch_size].to(device)
        a = aux_ids[s:s + batch_size].to(device) if voronoi else None
        ids = torch.cat([torch.full((t.shape[0], 1), bos, device=device), t], 1)
        for start in starts:
            span = min(block_len, seq_len - start + 1)
            lo = l[:, start - 1:start - 1 + span]
            hi = r[:, start - 1:start - 1 + span]
            # Voronoi payloads store the drawn id in both edges; the scalar ones
            # store a genuine interval to draw inside.
            u = lo if voronoi else lo + (hi - lo) * torch.rand(lo.shape, device=device)
            _, comp = model(input_ids=ids[:, :start], auxiliaries=u)
            logits = comp.logits[:, 0].float()          # slot 0 only
            truth = ids[:, start]
            order = logits.argsort(dim=-1, descending=True)
            rank = (order == truth[:, None]).float().argmax(dim=-1)
            for j in topj:
                hits[j] += int((rank < j).sum())
            pred = order[:, 0]
            wrong = pred != truth
            if wrong.any():
                pv = positions[pred[wrong].clamp(max=V - 1)]
                tv = positions[truth[wrong].clamp(max=V - 1)]
                err_cos.append((1 - (pv * tv).sum(-1)).cpu())
                rnd = positions[torch.randint(0, V, (int(wrong.sum()),),
                                              device=device)]
                rand_cos.append((1 - (rnd * tv).sum(-1)).cpu())
            if voronoi:
                mass = a[:, start - 1].float()          # placeholder, filled below
            else:
                mass = (hi - lo)[:, 0]
            mass_all.append(mass.cpu())
            ok_all.append((~wrong).cpu())
            n += t.shape[0]
    out = {"topj": {j: hits[j] / n for j in topj}, "n": n}
    if err_cos:
        out["err_cos"] = torch.cat(err_cos)
        out["rand_cos"] = torch.cat(rand_cos)
    out["mass"] = torch.cat(mass_all)
    out["ok"] = torch.cat(ok_all)
    return out


def main():
    args = parse_args()
    torch.set_grad_enabled(False)
    device = "cuda"
    from image_ptp.vqvae_ar_hf import build, build_module
    from image_ptp.gated_full import GatedFullTransformerModel

    teacher, meta = build(args.ar_ckpt, device=device, dtype=torch.float32)
    teacher.eval()
    V, bos = meta["num_codes"], meta["num_codes"]
    W = None
    for nm, p in teacher.named_parameters():
        if p.dim() == 2 and p.shape[0] == V + 1:
            W = p.data
    positions = F.normalize(W[:V].float(), dim=-1)

    for spec in args.arms:
        name, ckpt, data, adapter, kwargs = spec.split("=", 4)
        inner = build_module(args.ar_ckpt, device=device, dtype=torch.float32,
                             num_layers=args.num_layers)
        model = GatedFullTransformerModel(
            model_id=inner, dtype=torch.float32, adapter_name=adapter,
            adapter_kwargs=json.loads(kwargs) if kwargs else None,
            attn_implementation="flex_attention").to(device).eval()
        state = torch.load(Path(ckpt).expanduser(), map_location="cpu",
                           weights_only=False)["state_dict"]
        missing, unexpected = model.load_state_dict(
            {k[len("model."):]: v for k, v in state.items()
             if k.startswith("model.")}, strict=False)
        stale = [k for k in list(missing) + list(unexpected) if "u_embed" in k]
        assert not stale, f"{name}: u embedding did not load: {stale[:3]}"

        payload = torch.load(Path(data).expanduser(), map_location="cpu")
        n = min(args.images, payload["tokens"].shape[0])
        aux = payload.get("aux_ids")
        res = anatomy(model, payload["tokens"][:n].long(),
                      payload["left_bin_edges"][:n], payload["right_bin_edges"][:n],
                      aux[:n].long() if aux is not None else None,
                      bos, args.block_len, args.topj, positions, device,
                      args.batch_size, args.seed)

        print(f"\n===== {name} ({res['n']} slot-0 positions) =====")
        print("  top-j   accuracy")
        for j in args.topj:
            print(f"  {j:>5}   {res['topj'][j]:.4f}")
        if "err_cos" in res:
            e, rc = res["err_cos"], res["rand_cos"]
            print(f"  wrong predictions land at cosine distance "
                  f"{float(e.median()):.4f} from the truth; a random code sits at "
                  f"{float(rc.median()):.4f}")
        mass, ok = res["mass"], res["ok"]
        if aux is None:
            qs = torch.tensor([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
            edges = torch.quantile(mass, qs)
            print("  interval width quintile -> accuracy")
            for i in range(5):
                sel = (mass >= edges[i]) & (mass <= edges[i + 1])
                if int(sel.sum()):
                    print(f"    [{float(edges[i]):.4f}, {float(edges[i+1]):.4f}]"
                          f"  {float(ok[sel].float().mean()):.4f}  "
                          f"(n={int(sel.sum())})")
        del model, inner
        torch.cuda.empty_cache()
    print("\nSLOT0_DONE", flush=True)


if __name__ == "__main__":
    main()
