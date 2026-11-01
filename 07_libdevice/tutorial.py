# %% [markdown]
# # 07 — Libdevice
# Naive GELU → torch GELU → Triton `libdevice.erf` elementwise kernel.
#
# Official: https://triton-lang.org/main/getting-started/tutorials/07-extern-functions.html

# %%
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import math

import pytest
import torch
import triton
import triton.language as tl
import triton.language.extra.libdevice as libdevice

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
### PSEUDOCODE / PYTHON REFERENCE GELU           ###
####################################################


# %%
def naive_gelu_pseudo(x):
    """
    Exact GELU (Gaussian Error Linear Unit):

        gelu(x) = 0.5 * x * (1 + erf(x / sqrt(2)))

    Implemented with a Python loop over elements using math.erf so the
    formula is obvious. Prefer gelu_torch / the Triton kernel for speed.
    """
    flat = x.detach().cpu().reshape(-1)
    out = torch.empty_like(flat)
    inv_sqrt2 = 1.0 / math.sqrt(2.0)
    for i in range(flat.numel()):
        v = float(flat[i])
        out[i] = 0.5 * v * (1.0 + math.erf(v * inv_sqrt2))
    return out.reshape(x.shape).to(device=x.device, dtype=x.dtype)


# %%
def gelu_torch(x):
    """Same formula with vectorized torch ops (still not a custom kernel)."""
    return 0.5 * x * (1.0 + torch.erf(x / math.sqrt(2.0)))


######################################################
### TRITON + LIBDEVICE                               ###
######################################################


# %%
@triton.jit
def gelu_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """Elementwise GELU using CUDA libdevice erf via Triton's extra bindings."""
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask).to(tl.float32)
    # 0.7071... = 1/sqrt(2); libdevice.erf is the device math library
    y = libdevice.erf(x * 0.7071067811865476)
    y = 0.5 * x * (1.0 + y)
    tl.store(y_ptr + offs, y, mask=mask)


def gelu_libdevice(x):
    y = torch.empty_like(x)
    n = x.numel()
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    gelu_kernel[grid](x, y, n, BLOCK_SIZE=1024)
    return y


# %%
### TESTS ###
def test_pseudo_matches_torch() -> None:
    torch.manual_seed(0)
    x = torch.randn(128)
    ref = torch.nn.functional.gelu(x)
    assert torch.allclose(naive_gelu_pseudo(x), ref, atol=1e-5, rtol=1e-5)
    assert torch.allclose(gelu_torch(x), ref, atol=1e-5, rtol=1e-5)


def test_correctness() -> None:
    require_gpu()
    x = torch.randn(8192, device=DEVICE, dtype=torch.float32)
    y = gelu_libdevice(x)
    y_ref = torch.nn.functional.gelu(x)
    assert torch.allclose(y, y_ref, atol=1e-2, rtol=1e-2)
    print("gelu via libdevice OK")


# %%
if __name__ == "__main__":
    print("Running Tests...")
    test_pseudo_matches_torch()
    try:
        test_correctness()
    except pytest.skip.Exception as e:
        print(f"skipped: {e}")
