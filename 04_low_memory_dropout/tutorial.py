# %% [markdown]
# # 04 — Low-Memory Dropout
# Materialized mask → fused per-element rand → Triton `tl.rand` (no mask tensor).

"""
Low-memory dropout as described in Triton Docs
https://triton-lang.org/main/getting-started/tutorials/04-low-memory-dropout.html

Standard dropout:
    y = x / (1 - p)  if keep, else 0
with keep ~ Bernoulli(1 - p).

Storing a full `bool` mask adds one byte per element (about 75% extra traffic for fp32). Generating the random
keep/drop decision on the fly (and never writing the mask) is the "low memory"
trick — both in the Python fuse and in the Triton kernel via tl.rand.
"""

# %%
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
import torch
import triton
import triton.language as tl


def _active_device():
    """Pick Triton active device, else CUDA, else CPU (study helper; no common/ package)."""
    try:
        try:
            return triton.runtime.driver.active.get_active_torch_device()
        except Exception:
            pass
    except Exception:
        pass
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


DEVICE = _active_device()


def require_gpu() -> None:
    if not torch.cuda.is_available() and getattr(DEVICE, "type", None) != "cuda":
        pytest.skip(f"GPU not available (active device: {DEVICE})")


####################################################
### NAIVE: MATERIALIZE FULL BERNOULLI MASK       ###
####################################################


# %%
def naive_dropout_pseudo(x, p, seed: int = 0):
    """
    Classic dropout: allocate a full keep-mask, then apply it.

    Memory picture for an n-element tensor:
        - read  x          (n floats)
        - write mask       (n bools)   <-- extra global buffer
        - read  x + mask
        - write y

    The mask is O(N), but it still costs an extra allocation and a round trip. That is the cost we want to avoid.
    """
    assert 0.0 <= p < 1.0
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    # Materialize the full Bernoulli mask in memory
    x_cpu = x.detach().cpu()
    mask = torch.rand(x_cpu.shape, generator=g, dtype=x_cpu.dtype) >= p
    y = torch.where(mask, x_cpu / (1.0 - p), torch.zeros_like(x_cpu))
    return y.to(device=x.device, dtype=x.dtype)


####################################################
### INTERMEDIATE: FUSED RAND + APPLY (NO MASK)   ###
####################################################


# %%
def fused_dropout_pseudo(x, p, seed: int = 0):
    """
    Same math, but generate the keep decision per element without storing mask.

    We use a Python loop (for teaching): random() and the scale/zero happen in
    one pass — no n-element mask tensor is ever allocated. The Triton kernel
    does the same with tl.rand(seed, offsets).
    """
    assert 0.0 <= p < 1.0
    n = x.numel()
    y = torch.empty_like(x, device="cpu").cpu()
    x_flat = x.detach().cpu().reshape(-1)
    y_flat = y.reshape(-1)

    g = torch.Generator(device="cpu")
    g.manual_seed(seed)

    for i in range(n):
        # Uniform[0, 1) keep decision — never written to a mask buffer
        r = torch.rand((), generator=g).item()
        if r >= p:
            y_flat[i] = x_flat[i] / (1.0 - p)
        else:
            y_flat[i] = 0.0

    return y.to(device=x.device, dtype=x.dtype)


######################################################
### IDENTICAL TRITON CODE THAT DOES THE SAME THING ###
######################################################


# %%
@triton.jit
def dropout_kernel(
    x_ptr,
    output_ptr,
    n_elements,
    p,
    seed,
    BLOCK_SIZE: tl.constexpr,
):
    # Blocked 1-D grid (same pattern as vector-add)
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    # On-the-fly RNG — no mask tensor in HBM
    random = tl.rand(seed, offsets)
    x_keep = random >= p
    output = tl.where(x_keep, x / (1 - p), 0.0)
    tl.store(output_ptr + offsets, output, mask=mask)


def dropout(x, p, seed=0):
    output = torch.empty_like(x)
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    dropout_kernel[grid](x, output, n_elements, p, seed, BLOCK_SIZE=1024)
    return output


# %%
### TESTS ###
def test_pseudo_statistical() -> None:
    """CPU-runnable: keep rate ≈ (1-p); mean ≈ 1 on a ones vector; seeded fuse is reproducible."""
    torch.manual_seed(0)
    n = 50_000
    x = torch.ones(n)  # ones → expected mean is 1 under inverted dropout
    p = 0.25

    y_naive = naive_dropout_pseudo(x, p, seed=0)
    y_fused = fused_dropout_pseudo(x, p, seed=7)

    keep_naive = (y_naive != 0).float().mean().item()
    keep_fused = (y_fused != 0).float().mean().item()
    assert abs(keep_naive - (1.0 - p)) < 0.02
    assert abs(keep_fused - (1.0 - p)) < 0.02

    # Inverted dropout on ones → kept values are 1/(1-p); overall mean ≈ 1
    assert abs(y_naive.mean().item() - 1.0) < 0.05
    assert abs(y_fused.mean().item() - 1.0) < 0.05

    # p=0 → identity
    assert torch.allclose(naive_dropout_pseudo(x, p=0.0, seed=0), x)
    assert torch.allclose(fused_dropout_pseudo(x, p=0.0, seed=0), x)

    # Seeded reproducibility for fused Python path
    y_fused_2 = fused_dropout_pseudo(x, p, seed=7)
    assert torch.equal(y_fused, y_fused_2)


def test_correctness() -> None:
    require_gpu()
    torch.manual_seed(0)
    x = torch.randn(4096, device=DEVICE)
    p = 0.25
    seed = 123
    y = dropout(x, p, seed=seed)
    y2 = dropout(x, p, seed=seed)
    # Same seed → bitwise-identical output (tl.rand is deterministic given seed+offset)
    assert torch.equal(y, y2)

    # Kept values must be x / (1-p)
    scale = 1.0 / (1.0 - p)
    nonzero = y != 0
    assert torch.allclose(y[nonzero], x[nonzero] * scale, atol=1e-5)

    # Statistical sanity on a larger buffer of ones
    ones = torch.ones(100_000, device=DEVICE)
    y_stat = dropout(ones, p, seed=seed)
    keep_rate = (y_stat != 0).float().mean().item()
    assert abs(keep_rate - (1.0 - p)) < 0.02
    assert abs(y_stat.mean().item() - 1.0) < 0.05
    print("deterministic dropout OK", y.count_nonzero().item())


# %%
if __name__ == "__main__":
    print("Running Tests...")
    test_pseudo_statistical()
    try:
        test_correctness()
    except pytest.skip.Exception as e:
        print(f"skipped: {e}")
