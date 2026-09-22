#!/usr/bin/env python3
"""Train the SIGIL learned stratum.

Curriculum
----------
The encoder cannot learn anything if the very first batches are destroyed, so
the noise layer opens on the identity and mild processing and ramps toward the
generative round-trips.  Fidelity is likewise annealed: the residual budget
starts loose so a signal exists to be read, and tightens onto the perceptual
target once the decoder can read it.

Losses
------
``message``  binary cross-entropy on the transmitted codeword after attack.
``image``    an L2 term plus LPIPS, weighted onto a perceptual mask so the
             residual concentrates where contrast masking hides it.
``adaptive`` a jointly trained remover network and a short PGD attack on the
             decoder, so robustness is measured against an opponent that
             searches rather than a fixed attack menu.

Training:
    python scripts/train_latent.py --steps 40000
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sigil.latent import Decoder, Encoder, LatentConfig, Remover, perceptual_mask
from sigil.noise import NoiseLayer, VAEBank, pgd_attack

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


class PatchDataset(Dataset):
    """Random square crops resampled to the canonical grid.

    Cropping a large source and then resizing reproduces the deployment path —
    the detector always sees a canonical resample — while giving the encoder a
    wide range of effective scales and content statistics.
    """

    def __init__(self, roots, size: int = 256, length: int = 1 << 20, seed: int = 0):
        from sigil.common import list_images

        self.paths = []
        for r in roots:
            p = Path(r)
            if p.exists():
                self.paths.extend(list_images(p))
        if not self.paths:
            raise SystemExit(f"no images under {roots}")
        self.size = size
        self.length = length
        self.seed = seed

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        from PIL import Image

        rng = random.Random(self.seed * 1000003 + idx)
        for _ in range(8):
            try:
                p = self.paths[rng.randrange(len(self.paths))]
                img = Image.open(p).convert("RGB")
                w, h = img.size
                side = rng.randint(min(self.size, min(h, w)), min(h, w))
                x0 = rng.randint(0, w - side)
                y0 = rng.randint(0, h - side)
                crop = img.crop((x0, y0, x0 + side, y0 + side)).resize(
                    (self.size, self.size), Image.Resampling.LANCZOS
                )
                a = np.asarray(crop, dtype=np.float32) / 255.0
                return torch.from_numpy(a).permute(2, 0, 1)
            except Exception:
                continue
        return torch.zeros(3, self.size, self.size)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def is_main() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


def log(msg: str) -> None:
    if is_main():
        print(msg, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data",
        nargs="+",
        default=["data/div2k/DIV2K_train_HR", "data/corpus/synthetic"],
    )
    ap.add_argument("--out", default="checkpoints")
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--n-bits", type=int, default=256)
    ap.add_argument(
        "--grid",
        type=int,
        default=32,
        help="message cells per side; cell size is size/grid pixels, "
        "which is what a rotation has to resample across",
    )
    ap.add_argument("--strength", type=float, default=0.045)
    ap.add_argument("--target-psnr", type=float, default=36.0)
    ap.add_argument("--warmup", type=int, default=1500)
    ap.add_argument("--severity-steps", type=int, default=14000)
    ap.add_argument(
        "--turbo",
        action="store_true",
        help="put real SD-Turbo img2img in the noise layer (BPDA)",
    )
    ap.add_argument("--adversarial-from", type=int, default=6000)
    ap.add_argument("--vae-from", type=int, default=2500)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=20260727)
    ap.add_argument("--resume", default="")
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--save-every", type=int, default=2000)
    ap.add_argument(
        "--keep-snapshots",
        type=int,
        default=6,
        help="how many periodic checkpoints to retain",
    )
    args = ap.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    if world > 1:
        dist.init_process_group("gloo")
    device = "cpu"
    torch.manual_seed(args.seed + local_rank)
    np.random.seed(args.seed + local_rank)

    cfg = LatentConfig(
        canon=args.size, n_bits=args.n_bits, strength=args.strength, grid=args.grid
    )
    enc = Encoder(cfg).to(device)
    dec = Decoder(cfg).to(device)
    rem = Remover().to(device)

    start_step = 0
    if args.resume and Path(args.resume).exists():
        ck = torch.load(args.resume, map_location=device)
        enc.load_state_dict(ck["encoder"])
        dec.load_state_dict(ck["decoder"])
        rem.load_state_dict(ck.get("remover", rem.state_dict()))
        start_step = int(ck.get("step", 0))
        log(f"resumed from {args.resume} at step {start_step}")

    if world > 1:
        enc = nn.parallel.DistributedDataParallel(enc, device_ids=[local_rank])
        dec = nn.parallel.DistributedDataParallel(dec, device_ids=[local_rank])
        rem = nn.parallel.DistributedDataParallel(rem, device_ids=[local_rank])

    opt = torch.optim.AdamW(
        list(enc.parameters()) + list(dec.parameters()), lr=args.lr, weight_decay=1e-4
    )
    opt_rem = torch.optim.AdamW(rem.parameters(), lr=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt,
        max_lr=args.lr,
        total_steps=max(args.steps - start_step, 1),
        pct_start=0.02,
        div_factor=4,
        final_div_factor=25,
    )

    try:
        import lpips

        percep = lpips.LPIPS(net="alex").to(device)
        percep.requires_grad_(False)
    except Exception as exc:  # pragma: no cover
        log(f"LPIPS unavailable ({exc}); falling back to L1 only")
        percep = None

    vaes = None
    try:
        vaes = VAEBank(
            ["stabilityai/sd-vae-ft-mse", "madebyollin/sdxl-vae-fp16-fix"],
            device=device,
        )
        log(f"VAE bank: {vaes.ids}")
    except Exception as exc:
        log(f"VAE bank unavailable ({exc}); training without generative round-trips")

    turbo = None
    if args.turbo:
        try:
            from sigil.noise import TurboRegen

            turbo = TurboRegen(device=device)
            log("SD-Turbo regeneration active in the noise layer")
        except Exception as exc:
            log(f"SD-Turbo unavailable ({exc}); training without real regeneration")

    noise_mild = NoiseLayer(vaes=None, seed=args.seed + local_rank)
    noise_full = NoiseLayer(vaes=vaes, seed=args.seed + 7 + local_rank, turbo=turbo)

    ds = PatchDataset(args.data, size=args.size, seed=args.seed + local_rank)
    dl = DataLoader(
        ds,
        batch_size=args.batch,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=args.workers > 0,
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    hist = []
    t0 = time.time()
    step = start_step
    ema_acc, ema_psnr = 0.5, 0.0

    for batch in dl:
        if step >= args.steps:
            break
        x = batch.to(device, non_blocking=True)
        b = x.shape[0]
        bits = torch.randint(0, 2, (b, cfg.n_bits), device=device).float()

        # --- embed ---
        residual = enc(x, bits)
        mask = perceptual_mask(x)
        ramp = min(1.0, (step + 1) / max(args.warmup, 1))
        amp = cfg.strength * (2.2 - 1.2 * ramp)
        x_w = (x + amp * residual * mask).clamp(0, 1)

        # --- attack ---
        severity = min(1.0, step / max(args.severity_steps, 1))
        noise_mild.severity = severity
        noise_full.severity = severity
        layer = (
            noise_full if (vaes is not None and step >= args.vae_from) else noise_mild
        )
        x_a, attack_name = layer(x_w)
        if step >= args.adversarial_from:
            u = float(np.random.rand())
            if u < 0.25:
                x_a = (rem.module if world > 1 else rem)(x_a)
            elif u < 0.40:
                base = dec.module if world > 1 else dec
                x_a = pgd_attack(
                    base, x_a, bits, eps=float(np.random.uniform(2, 8)) / 255, steps=3
                )

        logits = dec(x_a)
        msg_loss = F.binary_cross_entropy_with_logits(logits, bits)

        # --- fidelity ---
        delta = x_w - x
        l2 = (delta**2).mean()
        lp = (
            percep((x_w * 2 - 1), (x * 2 - 1)).mean()
            if percep is not None
            else delta.abs().mean()
        )
        psnr = 10 * torch.log10(1.0 / l2.clamp_min(1e-12))
        # Push fidelity only while it is worse than target; below target the
        # gradient budget belongs entirely to robustness.
        fid_w = torch.clamp((args.target_psnr - psnr) / 6.0, min=0.0, max=3.0).detach()
        loss = msg_loss + fid_w * (24.0 * l2 + 1.6 * lp)

        if not torch.isfinite(loss):
            opt.zero_grad(set_to_none=True)
            step += 1
            continue
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list((enc.module if world > 1 else enc).parameters())
            + list((dec.module if world > 1 else dec).parameters()),
            2.0,
        )
        opt.step()
        sched.step()

        # --- opponent update ---
        if step >= args.adversarial_from and step % 2 == 0:
            with torch.no_grad():
                x_w_d = x_w.detach()
            cleaned = rem(x_w_d)
            adv_logits = (dec.module if world > 1 else dec)(cleaned)
            # Maximise decoder confusion, keep the image close to its input.
            rem_loss = (
                -F.binary_cross_entropy_with_logits(adv_logits, bits)
                + 90.0 * ((cleaned - x_w_d) ** 2).mean()
            )
            opt_rem.zero_grad(set_to_none=True)
            rem_loss.backward()
            opt_rem.step()

        with torch.no_grad():
            acc = ((logits > 0).float() == bits).float().mean().item()
        ema_acc = 0.98 * ema_acc + 0.02 * acc
        ema_psnr = 0.98 * ema_psnr + 0.02 * float(psnr.item())
        step += 1

        if step % args.log_every == 0:
            rate = (step - start_step) / max(time.time() - t0, 1e-9)
            log(
                f"step {step:6d}  acc={ema_acc:.4f}  psnr={ema_psnr:5.2f}  "
                f"msg={msg_loss.item():.4f}  lpips={float(lp):.4f}  "
                f"last={attack_name:9s}  {rate:.2f} it/s"
            )
            hist.append(
                {
                    "step": step,
                    "acc": ema_acc,
                    "psnr": ema_psnr,
                    "msg_loss": float(msg_loss.item()),
                    "attack": attack_name,
                }
            )

        if step % args.save_every == 0 and is_main():
            # Keep periodic snapshots, not just the latest.  Robustness does not
            # increase monotonically — switching on adversarial training visibly
            # cost clean accuracy in one run — and overwriting a single file
            # makes it impossible to go back and check, let alone recover.
            snap = out_dir / f"latent_step{step:06d}.pt"
            torch.save(
                {
                    "encoder": (enc.module if world > 1 else enc).state_dict(),
                    "decoder": (dec.module if world > 1 else dec).state_dict(),
                    "remover": (rem.module if world > 1 else rem).state_dict(),
                    "config": cfg.__dict__,
                    "step": step,
                    "args": vars(args),
                },
                snap,
            )
            import shutil as _sh

            _sh.copyfile(snap, out_dir / "latent.pt")
            keep = sorted(out_dir.glob("latent_step*.pt"))
            for old in keep[: -args.keep_snapshots]:
                old.unlink()
            (out_dir / "train_log.json").write_text(json.dumps(hist, indent=2))
            log(f"  checkpoint at step {step}")

    if is_main():
        torch.save(
            {
                "encoder": (enc.module if world > 1 else enc).state_dict(),
                "decoder": (dec.module if world > 1 else dec).state_dict(),
                "remover": (rem.module if world > 1 else rem).state_dict(),
                "config": cfg.__dict__,
                "step": step,
                "args": vars(args),
            },
            out_dir / "latent.pt",
        )
        (out_dir / "train_log.json").write_text(json.dumps(hist, indent=2))
        log("done")
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
