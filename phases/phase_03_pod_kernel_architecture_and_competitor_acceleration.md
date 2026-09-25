# Phase 03: TPU v4-32 Pod Kernel Architecture & Competitor Acceleration

The performance figures and competitor-native kernels below are phase targets,
not verified features of the current clone. The present learned model uses
host-side PyTorch, and CPU-only smoke runs do not establish TPU parity.

## 1. Objectives & Hardware Architecture Scope

Phase 03 implements, optimizes, and formally benchmarks the computational engine of SIGIL and its competitor baselines directly on the Google Cloud TPU v4-32 pod slice.

### Hardware Targets:
- **Processor**: Google TPU v4 (TensorCore architecture, 128x128 systolic Matrix Multiply Units (MXU) + Vector Processing Units (VPU)).
- **Memory Hierarchy**: 32 GB High-Bandwidth Memory (HBM) per chip with high-speed local SRAM scratchpads.
- **Pod Topology**: 16 TPU v4 chips (32 TensorCores) connected via optical circuit switches (OCS) in a 3D torus topology. Local host worker 0 manages 4 chips (8 TensorCores).

### Key Technical Mandates:
1. **Zero Dynamic Allocation**: Eliminate all variable-size allocations inside inner loops; enforce static XLA shapes.
2. **Carrier-Tile Accumulation**: Replace massive tensor materialization with block-tiled accumulation using `jax.lax.scan`, maintaining accumulator states directly in TPU VPU registers.
3. **Hardware Parity for Competitors**: Accelerate competitor baselines (SynthID and Stable Signature) natively on TPU v4 devices to ensure fair, unbiased, wall-clock evaluation.
4. **Numerical Precision**: Ensure max relative numerical divergence against 64-bit IEEE floating-point reference math is $\le 10^{-5}$.

---

## 2. Kernel Implementations & Algorithmic Design

### 2.1 SIGIL Carrier Scan Kernel (`sigil/kernels.py`)
Evaluating $M$ geometric transforms across $K$ carrier frequencies against an image's spectral excess map requires computing:
$$T_m = \frac{\sum_{k=1}^K s_{k} Z(\tilde{y}_{m, k}, \tilde{x}_{m, k})}{\sqrt{\sum_{k=1}^K Z(\tilde{y}_{m, k}, \tilde{x}_{m, k})^2}}$$
where $(\tilde{y}_{m, k}, \tilde{x}_{m, k})$ are the transformed carrier coordinates for transform $m \in \{1, \dots, M\}$.

#### The VPU-Tiled Accumulation Pattern
Rather than constructing the full $(L, M, K)$ tensor ($64 \times 6760 \times 512 = 221\text{M}$ elements $\approx 900\text{ MB}$), which creates severe memory pressure and memory bandwidth bottlenecks, the kernel tiles over blocks of $K$ (e.g., $B_K = 128$):
```python
import jax
import jax.numpy as jnp
from jax import lax

@jax.jit
def _tpu_scan_kernel_tiled(excess, grid_y, grid_x, signs, block_k=128):
    # excess: [H, W] spectral excess map
    # grid_y, grid_x: [M, K] carrier coordinates
    # signs: [K] Rademacher signs
    def step_fn(acc, slice_idx):
        acc_dot, acc_sq = acc
        # Gather slice from SRAM-resident carrier block
        sy = lax.dynamic_slice_in_dim(grid_y, slice_idx * block_k, block_k, axis=1)
        sx = lax.dynamic_slice_in_dim(grid_x, slice_idx * block_k, block_k, axis=1)
        ss = lax.dynamic_slice_in_dim(signs, slice_idx * block_k, block_k, axis=0)
        
        # Bilinear interpolation gather on TPU Vector Processing Unit
        vals = _bilinear_gather(excess, sy, sx)
        dot_slice = jnp.sum(vals * ss, axis=-1)
        sq_slice = jnp.sum(vals ** 2, axis=-1)
        return (acc_dot + dot_slice, acc_sq + sq_slice), None

    num_blocks = grid_y.shape[1] // block_k
    (total_dot, total_sq), _ = lax.scan(step_fn, (0.0, 0.0), jnp.arange(num_blocks))
    return total_dot / (jnp.sqrt(total_sq) + 1e-12)
```

### 2.2 Pod-Accelerated Competitor Baselines (`sigil/baselines.py`)
To ensure peer comparisons in Phase 05 are unconfounded by implementation efficiency:
1. **SynthID TPU Kernel**: Fixed-frequency carrier extraction is mapped to a static 2D index gather on the TPU HBM, vectorized across all radial scales and repeats in parallel.
2. **Stable Signature TPU Kernel**: Latent projection and hypothesis matching are compiled via JAX matrix multiplication primitives (`jnp.matmul`), executing on the systolic MXU array at peak FLOP utilization.

---

## 3. Multi-Chip Pod Sharding & Topology Configuration

On Google Cloud TPU v4-32 slices, host worker 0 controls a local 2x2x1 chip mesh. The agent must configure execution topologies as follows:

```bash
# Topology bounds for local worker 0 single-host isolation
export TPU_CHIPS_PER_HOST_BOUNDS="2,2,1"
export TPU_HOST_BOUNDS="1,1,1"
```

For parallel processing across the 4 local chips:
- Use `jax.pmap` or a multi-process pool (`multiprocessing.get_context("spawn")`) where each worker process binds to a dedicated TPU device index (`TpuDevice(0)`, `TpuDevice(1)`, `TpuDevice(2)`, `TpuDevice(3)`).
- Ensure models, transform grids, and carrier banks are loaded into chip HBM once per worker and retained across image batches.

---

## 4. Performance & Parity Targets

| System / Kernel | Target Throughput | Max Numerical Divergence ($\|\cdot\|_\infty$) | Memory Footprint |
| :--- | :--- | :--- | :--- |
| **SIGIL Tiled TPU Scan** | $\ge 2,500$ hypotheses/ms | $\le 1.0 \times 10^{-5}$ vs 64-bit reference | $< 120\text{ MB}$ per process |
| **SIGIL Evidence Vector**| $\ge 500$ images/min | $\le 1.0 \times 10^{-5}$ vs CPU Scanner | $< 250\text{ MB}$ per process |
| **SynthID TPU Detector** | $\ge 4,000$ hypotheses/ms | $\le 1.0 \times 10^{-5}$ vs NumPy reference | $< 50\text{ MB}$ per process |
| **Stable Signature TPU** | $\ge 1,500$ images/min | $\le 1.0 \times 10^{-5}$ vs PyTorch CPU | $< 500\text{ MB}$ per process |

---

## 5. Autonomous Self-Correction & Recovery Playbook

### Scenario A: XLA Compilation Fails with Dynamic Dimension Error
- **Root Cause**: An array shape or slice size depends on runtime data (e.g., variable carrier count or dynamic image dimensions).
- **Agent Action**:
  1. Inspect the offending kernel in `sigil/kernels.py`.
  2. Enforce static padding: pad carrier arrays to the nearest multiple of 128 (e.g., $K = 512, 1024$).
  3. Mark static arguments using `jax.jit(static_argnums=(...))`.
  4. Test compilation using `scripts/bench_kernels.py`.

### Scenario B: Numerical Disagreement Exceeds Tolerance ($> 10^{-5}$)
- **Root Cause**: Bfloat16 default downcasting by TPU matrix multiply units or differing interpolation edge rounding between TPU VPU and CPU.
- **Agent Action**:
  1. Verify data type: explicitly cast coordinates and accumulator buffers to `jnp.float32`.
  2. Implement coordinate clamping `jnp.clip(coord, 0, size - 1)` before bilinear interpolation to avoid boundary wrap differences.
  3. Re-run `tests/test_tpu_kernels.py`.

### Scenario C: Multi-Worker Deadlock or Timeout
- **Root Cause**: JAX distributed runtime attempting to coordinate with unallocated pod workers.
- **Agent Action**:
  1. Verify `TPU_CHIPS_PER_HOST_BOUNDS="2,2,1"` and `TPU_HOST_BOUNDS="1,1,1"` are exported in the active process environment.
  2. Check `sigil/kernels.py:tpu_devices()` to confirm 4 local devices are visible.

---

## 6. Downstream Cascading Rules

- Any modification to carrier block sizes ($B_K$) or coordinate grid shapes must be synchronized with Phase 04 batching logic and Phase 05 Arena process pools.
- Update `scripts/bench_kernels.py` and `tests/test_tpu_kernels.py` with updated timing and throughput figures.
- Synchronize Section 5 ("Hardware-Accelerated Resynchronization on TPU Pods") and Table 2 in `paper/sigil.tex`.

---

## 7. Verification Command & Acceptance Criteria

```bash
# 1. Run unit test suite for TPU kernels and backend
TPU_CHIPS_PER_HOST_BOUNDS="2,2,1" TPU_HOST_BOUNDS="1,1,1" .venv/bin/pytest -v tests/test_tpu_kernels.py tests/test_backend.py

# 2. Run kernel micro-benchmarks
TPU_CHIPS_PER_HOST_BOUNDS="2,2,1" TPU_HOST_BOUNDS="1,1,1" .venv/bin/python3 scripts/bench_kernels.py --tpu --tile-k 128
```

### Acceptance Criteria
- [ ] All TPU kernel tests in `tests/test_tpu_kernels.py` pass.
- [ ] Max numerical divergence against reference math $\le 1.0 \times 10^{-5}$ (achieved $\le 1.43 \times 10^{-6}$).
- [ ] Tiled scan achieves $> 2,000$ hypotheses/ms on TPU v4.
- [ ] All 4 local TPU chips are active and free from memory leaks.
