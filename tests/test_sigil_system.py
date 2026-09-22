"""End-to-end smoke test for the Sigil watermarking system on Google Cloud TPU v4."""

import numpy as np
from scipy.ndimage import gaussian_filter

from sigil.invariant import InvariantConfig
from sigil.system import Sigil, SigilConfig


def test_sigil_end_to_end_tpu():
    """Verify end-to-end watermarking embedding and detection on Google Cloud TPU."""
    # Textured image representing realistic spatial frequency content
    rng = np.random.default_rng(42)
    raw = rng.uniform(0.2, 0.8, size=(512, 512, 3)).astype(np.float32)
    img = gaussian_filter(raw, sigma=[2.0, 2.0, 0.0])

    # Configure Sigil with TPU acceleration
    cfg = SigilConfig(
        invariant=InvariantConfig(),
        latent_checkpoint=None,  # Pure analytic stratum test
        device="tpu",
        alpha=1e-6,
    )
    sig = Sigil(cfg)

    # Embed mark
    res = sig.embed(img, nonce=12345)
    assert res.image.shape == img.shape
    assert res.psnr > 30.0, f"Expected high embedding PSNR, got {res.psnr:.1f} dB"
    assert res.ssim > 0.85, f"Expected high embedding SSIM, got {res.ssim:.3f}"

    # Detect mark with expected nonce
    det = sig.detect(res.image, expected_nonce=res.nonce)
    assert det.detected is True, (
        f"Expected detection to succeed, got pvalue={det.pvalue:.2e}"
    )
    assert det.pvalue <= 1e-6, f"Expected pvalue <= 1e-6, got {det.pvalue:.2e}"
    assert det.analytic is not None
    assert det.analytic.best_anchor == "nonce"
    assert det.analytic.statistic > 10.0

    # Verify that an unmarked image does not falsely detect (H0 null check)
    det_unmarked = sig.detect(img, expected_nonce=res.nonce)
    assert det_unmarked.detected is False, "Unmarked image must not trigger detection"
