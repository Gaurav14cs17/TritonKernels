# %% [markdown]
# # 02 — Fused Softmax
# Multi-pass naive → per-row Python fuse → single Triton kernel (less HBM traffic).

"""
Fused Softmax as described in Triton Docs
https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html

Softmax(x)_i = exp(x_i - max(x)) / sum_j exp(x_j - max(x))

A naive implementation walks the row many times (max, subtract, exp, sum, div),
each pass reading/writing global memory. Fusion keeps the row in registers /
SRAM and does all reductions in one kernel launch.
"""

# %%
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
import torch
import triton
import triton.language as tl
from triton.runtime import driver

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

def is_hip() -> bool:
    try:
        return triton.runtime.driver.active.get_current_target().backend == "hip"
    except Exception:
        return False


def is_cdna() -> bool:
    try:
        return is_hip() and triton.runtime.driver.active.get_current_target().arch in (
            "gfx940",
            "gfx941",
            "gfx942",
            "gfx90a",
        )
    except Exception:
        return False


####################################################
### NAIVE: MULTI-PASS (LOTS OF MEMORY TRAFFIC)   ###
####################################################


# %%
def naive_softmax(x):
    """
    Numerically stable softmax as a sequence of full-tensor passes.

    Each line below is roughly one kernel / one full read (or write) of the
    activation buffer in a non-fused eager framework:

        1. x.max(dim=1)          — read X, write row-max
        2. x - x_max             — read X + max, write Z
        3. torch.exp(z)          — read Z, write numerator
        4. numerator.sum(dim=1)  — read numerator, write denom
        5. numerator / denom     — read numerator + denom, write Y

    That is ~5 reads of the big [M, N] buffer (plus a few tiny row vectors).
    Bandwidth, not FLOPs, dominates. Fusing collapses this into one read of X
    and one write of Y.
    """
    # Pass 1: row-wise max (for numerical stability)
    x_max = x.max(dim=1)[0]
    # Pass 2: subtract max
    z = x - x_max[:, None]
    # Pass 3: exp
    numerator = torch.exp(z)
    # Pass 4: row-wise sum
    denominator = numerator.sum(dim=1)
    # Pass 5: normalize
    return numerator / denominator[:, None]


####################################################
### INTERMEDIATE: ONE ROW AT A TIME (STILL PY)   ###
####################################################


# %%
def row_softmax_pseudo(x):
    """
    Same math as naive_softmax, but structured like the Triton kernel:
    one Python function body processes an entire row.

    Conceptually we still "touch" the row multiple times inside the function
    (max → sub → exp → sum → div), but we no longer materialize five full
    [M, N] temporaries in global memory. The Triton kernel goes further: those
    intermediates live in registers / SRAM for a single program.
    """
    M, N = x.shape
    y = torch.empty_like(x)

    for row_idx in range(M):
        row = x[row_idx]  # one logical load of the row
        row_max = row.max()
        numer = torch.exp(row - row_max)
        denom = numer.sum()
        y[row_idx] = numer / denom  # one logical store of the row

    return y


######################################################
### IDENTICAL TRITON CODE THAT DOES THE SAME THING ###
######################################################


# %%
@triton.jit
def softmax_kernel(
    output_ptr,
    input_ptr,
    input_row_stride,
    output_row_stride,
    n_rows,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
    num_stages: tl.constexpr,
):
    # Persistent programs: each pid walks a strided set of rows
    # (same "one row at a time" idea as row_softmax_pseudo).
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)
    for row_idx in tl.range(row_start, n_rows, row_step, num_stages=num_stages):
        row_start_ptr = input_ptr + row_idx * input_row_stride
        col_offsets = tl.arange(0, BLOCK_SIZE)
        input_ptrs = row_start_ptr + col_offsets
        mask = col_offsets < n_cols
        # Pad OOB with -inf so max/exp ignore them
        row = tl.load(input_ptrs, mask=mask, other=-float("inf"))
        row_minus_max = row - tl.max(row, axis=0)
        numerator = tl.exp(row_minus_max)
        denominator = tl.sum(numerator, axis=0)
        softmax_output = numerator / denominator
        output_row_start_ptr = output_ptr + row_idx * output_row_stride
        output_ptrs = output_row_start_ptr + col_offsets
        tl.store(output_ptrs, softmax_output, mask=mask)


def _device_props():
    """Lazy so importing this module on CPU (no Triton driver) still works for pseudo tests."""
    properties = driver.active.utils.get_device_properties(DEVICE.index)
    return properties


def softmax(x):
    properties = _device_props()
    NUM_SM = properties["multiprocessor_count"]
    NUM_REGS = properties["max_num_regs"]
    SIZE_SMEM = properties["max_shared_mem"]
    WARP_SIZE = properties["warpSize"]

    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    num_warps = 8
    num_stages = 4 if SIZE_SMEM > 200000 else 2
    y = torch.empty_like(x)
    # Warmup to read register / smem usage, then pick occupancy-aware grid
    kernel = softmax_kernel.warmup(
        y,
        x,
        x.stride(0),
        y.stride(0),
        n_rows,
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_stages=num_stages,
        num_warps=num_warps,
        grid=(1,),
    )
    kernel._init_handles()
    n_regs = kernel.n_regs
    size_smem = kernel.metadata.shared
    if is_hip():
        num_gprs = NUM_REGS * 2 if is_cdna() else NUM_REGS
        max_num_threads = properties["max_threads_per_sm"]
        max_num_waves = max_num_threads // WARP_SIZE
        occupancy = min(num_gprs // WARP_SIZE // n_regs, max_num_waves) // num_warps
    else:
        occupancy = NUM_REGS // (n_regs * WARP_SIZE * num_warps)
    occupancy = min(occupancy, SIZE_SMEM // size_smem)
    num_programs = min(NUM_SM * occupancy, n_rows)
    kernel[(num_programs, 1, 1)](
        y, x, x.stride(0), y.stride(0), n_rows, n_cols, BLOCK_SIZE, num_stages
    )
    return y


# %%
### TESTS ###
def test_pseudo_matches_torch() -> None:
    """CPU-runnable: naive and per-row pseudo match torch.softmax."""
    torch.manual_seed(0)
    x = torch.randn(32, 64)
    ref = torch.softmax(x, dim=1)
    assert torch.allclose(naive_softmax(x), ref, atol=1e-5, rtol=1e-5)
    assert torch.allclose(row_softmax_pseudo(x), ref, atol=1e-5, rtol=1e-5)


def test_correctness() -> None:
    require_gpu()
    torch.manual_seed(0)
    x = torch.randn(1823, 781, device=DEVICE)
    y_triton = softmax(x)
    y_torch = torch.softmax(x, axis=1)
    assert torch.allclose(y_triton, y_torch)
    print("correctness OK")


# %%
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["N"],
        x_vals=[128 * i for i in range(2, 40)],
        line_arg="provider",
        line_vals=["triton", "torch", "naive_softmax"],
        line_names=["Triton", "Torch", "Naive Softmax"],
        styles=[("blue", "-"), ("green", "-"), ("red", "-")],
        ylabel="GB/s",
        plot_name="softmax-performance",
        args={"M": 4096},
    )
)
def benchmark(M, N, provider):
    x = torch.randn(M, N, device=DEVICE, dtype=torch.float32)
    if provider == "torch":
        ms = triton.testing.do_bench(lambda: torch.softmax(x, axis=-1))
    if provider == "triton":
        ms = triton.testing.do_bench(lambda: softmax(x))
    if provider == "naive_softmax":
        ms = triton.testing.do_bench(lambda: naive_softmax(x))
    gbps = lambda ms: 2 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms)


# %%
if __name__ == "__main__":
    print("Running Tests...")
    test_pseudo_matches_torch()
    test_correctness()

    print("Running Benchmark...")
    if torch.cuda.is_available() or DEVICE.type == "cuda":
        benchmark.run(print_data=True, show_plots=False)
