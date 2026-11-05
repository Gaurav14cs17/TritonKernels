# %% [markdown]
# # 01 — Vector Addition
# Naive (one element per pid) → blocked + masked → Triton `tl` kernel.

"""
Vector addition as described in Triton Docs
https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html

z = x + y  (elementwise)

We start with a Python "pseudo-kernel" that maps one program id (pid) to one
element, then block the work so each pid owns a contiguous chunk with masking
for the remainder — the same idea the Triton kernel implements with
tl.load / tl.store.
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
        import triton

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
### PSEUDOCODE: ONE ELEMENT PER PROGRAM ID       ###
####################################################


# %%
def naive_add_pseudo(x, y):
    """
    Elementwise add with one "pid" per output element.

    Think of each pid as a tiny GPU program. For vector add the mapping is
    trivial: pid i reads x[i] and y[i], writes z[i].

        pid 0 -> z[0] = x[0] + y[0]
        pid 1 -> z[1] = x[1] + y[1]
        ...

    On a real GPU this would launch n_elements programs — far too many tiny
    kernels. The next step packs many elements into each program.
    """
    assert x.shape == y.shape
    n = x.numel()
    z = torch.empty_like(x)

    # Flatten so indexing mirrors the 1-D Triton grid
    x_flat = x.reshape(-1)
    y_flat = y.reshape(-1)
    z_flat = z.reshape(-1)

    for pid in range(n):
        z_flat[pid] = x_flat[pid] + y_flat[pid]

    return z


####################################################
### PSEUDOCODE: BLOCKED + MASKED (LIKE TRITON)   ###
####################################################


# %%
def blocked_add_pseudo(x, y, BLOCK_SIZE: int = 1024):
    """
    Each pid owns a contiguous block of BLOCK_SIZE elements.

    If n = 5 and BLOCK_SIZE = 4 we launch ceil(5/4) = 2 pids:

        pid 0: offsets [0, 1, 2, 3]  — all valid
        pid 1: offsets [4, 5, 6, 7]  — only offset 4 is in-bounds

    The mask (offsets < n) skips the remainder so we never read/write OOB.
    This is exactly what the Triton kernel does with tl.load(..., mask=...).
    """
    assert x.shape == y.shape
    n = x.numel()
    z = torch.empty_like(x)

    x_flat = x.reshape(-1)
    y_flat = y.reshape(-1)
    z_flat = z.reshape(-1)

    num_pids = (n + BLOCK_SIZE - 1) // BLOCK_SIZE  # ceil division
    for pid in range(num_pids):
        block_start = pid * BLOCK_SIZE
        offsets = [block_start + i for i in range(BLOCK_SIZE)]

        for off in offsets:
            if off < n:  # mask for the last (partial) block
                z_flat[off] = x_flat[off] + y_flat[off]

    return z


######################################################
### IDENTICAL TRITON CODE THAT DOES THE SAME THING ###
######################################################


# %%
@triton.jit
def add_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Same blocked idea as blocked_add_pseudo:
    #   pid -> block_start -> offsets -> mask -> load / add / store
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, x + y, mask=mask)


def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    output = torch.empty_like(x)
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    return output


# %%
### TESTS ###
def test_pseudo_matches_torch() -> None:
    """CPU-runnable: naive and blocked Python stages match torch.add."""
    torch.manual_seed(0)
    x = torch.rand(1000)
    y = torch.rand(1000)
    ref = x + y
    assert torch.allclose(naive_add_pseudo(x, y), ref)
    assert torch.allclose(blocked_add_pseudo(x, y, BLOCK_SIZE=128), ref)
    # Remainder case: n not divisible by BLOCK_SIZE
    assert torch.allclose(blocked_add_pseudo(x, y, BLOCK_SIZE=256), ref)


def test_correctness() -> None:
    require_gpu()
    torch.manual_seed(0)
    size = 98432
    x = torch.rand(size, device=DEVICE)
    y = torch.rand(size, device=DEVICE)
    out_torch = x + y
    out_triton = add(x, y)
    assert torch.allclose(out_torch, out_triton), torch.max(torch.abs(out_torch - out_triton)).item()
    print("correctness OK", torch.max(torch.abs(out_torch - out_triton)).item())


# %%
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["size"],
        x_vals=[2**i for i in range(12, 28)],
        x_log=True,
        line_arg="provider",
        line_vals=["triton", "torch"],
        line_names=["Triton", "Torch"],
        styles=[("blue", "-"), ("green", "-")],
        ylabel="GB/s",
        plot_name="vector-add-performance",
        args={},
    )
)
def benchmark(size, provider):
    x = torch.rand(size, device=DEVICE, dtype=torch.float32)
    y = torch.rand(size, device=DEVICE, dtype=torch.float32)
    quantiles = [0.5, 0.2, 0.8]
    if provider == "torch":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: x + y, quantiles=quantiles)
    if provider == "triton":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: add(x, y), quantiles=quantiles)
    gbps = lambda ms: 3 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms), gbps(max_ms), gbps(min_ms)


# %%
if __name__ == "__main__":
    print("Running Tests...")
    test_pseudo_matches_torch()
    try:
        test_correctness()
    except pytest.skip.Exception as e:
        print(f"skipped: {e}")

    print("Running Benchmark...")
    if torch.cuda.is_available() or DEVICE.type == "cuda":
        benchmark.run(print_data=True, show_plots=False)
