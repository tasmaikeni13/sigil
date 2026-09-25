#!/usr/bin/env python3
"""Assign stable role names to the selected learned checkpoints.

Checkpoint filenames usually contain training-step metadata. This utility
turns the ranking output into the two role names used by the evaluation tools:
``fine.pt`` and ``coarse.pt``. Relative symbolic links keep the checkpoint tree
portable.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Optional


def best(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    if not data:
        return None
    candidates = [
        (name, rec)
        for name, rec in data.items()
        if rec.get("eligible") is True
        and rec.get("mock") is False
        and rec.get("key_mode") == "private"
    ]
    if not candidates:
        return None
    name, rec = max(candidates, key=lambda kv: (kv[1]["score"], kv[1]["psnr"]))
    return {"name": name, **rec}


def link(target: Path, name: Path) -> None:
    if name.is_symlink() or name.exists():
        raise FileExistsError(f"refusing to overwrite existing checkpoint role: {name}")
    # Relative, so the tree stays movable.
    name.symlink_to(os.path.relpath(target.resolve(), name.parent.resolve()))
    print(f"  {name}  ->  {name.readlink()}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fine", default="results/selection_fine.json")
    ap.add_argument("--coarse", default="results/selection_coarse.json")
    ap.add_argument("--out", default="checkpoints")
    args = ap.parse_args()

    out = Path(args.out)
    ready = []
    for role, sel in (("fine", args.fine), ("coarse", args.coarse)):
        b = best(Path(sel))
        if b is None:
            raise SystemExit(f"no eligible private-key ranking for {role} ({sel})")
        src = Path(b["checkpoint"])
        if not src.exists():
            raise SystemExit(f"ranked winner missing: {src}")
        if (out / f"{role}.pt").exists() or (out / f"{role}.pt").is_symlink():
            raise SystemExit(f"refusing to overwrite checkpoint role: {role}")
        ready.append((role, src, b))
    out.mkdir(parents=True, exist_ok=True)
    for role, src, b in ready:
        print(
            f"{role}: {b['name']}  grid {b['grid']}  bits {b['n_bits']}  "
            f"held {b['held']}  sumT {b['t_sum']:.1f}"
        )
        link(src, out / f"{role}.pt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
