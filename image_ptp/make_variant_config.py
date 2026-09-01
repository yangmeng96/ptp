"""Copy a train.yaml and change named keys, verifying each one landed.

Two configs in this project were generated with sed and silently kept the value
they were meant to override -- once a missing `adapter_name`, once a missing
`num_bins`. Both were caught late, by an assertion in a completely different
file. This reads the YAML, sets the keys, writes it back, and reads it again to
confirm; a typo in a key name is an error here rather than a run that quietly
measures the baseline twice.

    python -m image_ptp.make_variant_config SRC DST model.aux_sampling=beta ...

A leading `-` deletes a key instead of setting it, for options a class has since
dropped: `SphereInitAuxSampling` took `sphere_K` when the CIFAR configs were
written and now requires `sphere_from`, so a config copied forward carries a
keyword that reaches the parent constructor and raises.
"""
import sys
from pathlib import Path

import yaml


def set_path(cfg, dotted, value):
    keys = dotted.split(".")
    node = cfg
    for k in keys[:-1]:
        if k not in node:
            raise KeyError(f"{dotted}: no section {k!r} in {sorted(node)}")
        node = node[k]
    node[keys[-1]] = value


def get_path(cfg, dotted):
    node = cfg
    for k in dotted.split("."):
        node = node[k]
    return node


def del_path(cfg, dotted):
    keys = dotted.split(".")
    node = cfg
    for k in keys[:-1]:
        node = node[k]
    if keys[-1] not in node:
        raise KeyError(f"-{dotted}: nothing to delete; have {sorted(node)}")
    del node[keys[-1]]


def parse(v):
    for cast in (int, float):
        try:
            return cast(v)
        except ValueError:
            pass
    return {"true": True, "false": False}.get(v.lower(), v)


def main(argv):
    src, dst, *assigns = argv
    cfg = yaml.safe_load(Path(src).read_text())
    wanted, removed = {}, []
    for a in assigns:
        if a.startswith("-"):
            del_path(cfg, a[1:])
            removed.append(a[1:])
            continue
        k, v = a.split("=", 1)
        wanted[k] = parse(v)
        set_path(cfg, k, wanted[k])
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    Path(dst).write_text(
        f"# Generated from {src} by make_variant_config.\n"
        f"# Overrides: {', '.join(f'{k}={v}' for k, v in wanted.items())}\n"
        f"# Removed: {', '.join(removed) or '(none)'}\n"
        + yaml.safe_dump(cfg, sort_keys=False))
    check = yaml.safe_load(Path(dst).read_text())
    for k, v in wanted.items():
        got = get_path(check, k)
        if got != v:
            raise SystemExit(f"FAILED to set {k}: wanted {v!r}, file has {got!r}")
        print(f"  verified {k} = {got!r}")
    for k in removed:
        try:
            get_path(check, k)
        except KeyError:
            print(f"  verified {k} is gone")
        else:
            raise SystemExit(f"FAILED to delete {k}: still present")
    print(f"wrote {dst}")


if __name__ == "__main__":
    main(sys.argv[1:])
