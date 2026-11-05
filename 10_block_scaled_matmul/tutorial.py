# %% [markdown]
# # 10 — Block-Scaled Matrix Multiplication
# Naive scale-then-matmul → reuse tutorial-03 matmul → Triton scale + matmul path.
#
# Official (FP8 block scales): https://triton-lang.org/main/getting-started/tutorials/10-block-scaled-matmul.html

"""
Block-scaled matmul (simplified teaching version):

  C = (A * scale_a) @ (B * scale_b)

In the official FP8 tutorial, scales are *per-block* tensors that dequantize
low-precision tiles on the fly inside the GEMM. Here we start with scalar
(or broadcastable) scales so the Naive → next → tl ladder is clear, then
hand off to tutorial 03's grouped matmul for the heavy lifting.
"""

# %%
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import importlib.util

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

_HERE = Path(__file__).resolve().parent
_REPO = _HERE
for _candidate in (_HERE, *_HERE.parents):
    if (_candidate / '03_matrix_multiplication' / 'tutorial.py').exists():
        _REPO = _candidate
        break
else:
    raise FileNotFoundError(
        'cannot locate 03_matrix_multiplication from ' + str(_HERE)
    )
_matmul_path = _REPO / '03_matrix_multiplication' / 'tutorial.py'
_spec = importlib.util.spec_from_file_location("matmul_mod", _matmul_path)
_matmul_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_matmul_mod)
matmul = _matmul_mod.matmul


# %%
def naive_block_scaled_matmul_pseudo(a, b, scale_a, scale_b):
    """
    Materialize scaled A/B in memory, then matmul.

    Memory: two full scaled tensors + the GEMM.
    Fine for scalars; wasteful when scales are per-block and you could
    dequantize inside the GEMM tile loop (official FP8 path).
    """
    return torch.matmul(a * scale_a, b * scale_b)


# %%
def block_scaled_matmul_ref(a, b, scale_a, scale_b):
    """
    Intermediate: still scale in PyTorch, but GEMM via Triton tutorial 03.
    """
    return matmul(a * scale_a, b * scale_b)


# %%
@triton.jit
def scale_kernel(x_ptr, y_ptr, scale, n_elements, BLOCK_SIZE: tl.constexpr):
    """Elementwise y = x * scale — stand-in for on-the-fly dequant loads."""
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask)
    tl.store(y_ptr + offs, x * scale, mask=mask)


def triton_scale(x, scale):
    y = torch.empty_like(x)
    n = x.numel()
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    scale_kernel[grid](x, y, float(scale), n, BLOCK_SIZE=1024)
    return y


def block_scaled_matmul_triton(a, b, scale_a, scale_b):
    """
    Triton path: scale with a tiny kernel, then grouped matmul from tutorial 03.

    Real block-scaled FP8 kernels fuse the scale lookup into the K-loop loads;
    this keeps the teaching steps separable.
    """
    return matmul(triton_scale(a, scale_a), triton_scale(b, scale_b))


# %%
def test_pseudo_matches_torch() -> None:
    torch.manual_seed(0)
    a = torch.randn(32, 16)
    b = torch.randn(16, 24)
    sa, sb = 0.5, 3.0          # product 1.5, not 1: a kernel that ignores scales fails
    ref = torch.einsum("ik,kj->ij", a * sa, b * sb)
    assert torch.allclose(
        naive_block_scaled_matmul_pseudo(a, b, sa, sb), ref, atol=1e-5
    )


def test_correctness() -> None:
    require_gpu()
    m, k, n = 256, 128, 512
    a = torch.randn(m, k, device=DEVICE, dtype=torch.float16)
    b = torch.randn(k, n, device=DEVICE, dtype=torch.float16)
    sa = torch.tensor(0.5, device=DEVICE, dtype=torch.float16)
    sb = torch.tensor(3.0, device=DEVICE, dtype=torch.float16)
    ref = torch.matmul(a.float() * sa, b.float() * sb)
    c_ref = block_scaled_matmul_ref(a, b, sa, sb)
    c_tl = block_scaled_matmul_triton(a, b, float(sa), float(sb))
    assert torch.allclose(c_ref.float(), ref.float(), atol=1e-2, rtol=1e-2)
    assert torch.allclose(c_tl.float(), ref.float(), atol=1e-2, rtol=1e-2)
    print("block-scaled OK")


# %%
if __name__ == "__main__":
    print("Running Tests...")
    test_pseudo_matches_torch()
    try:
        test_correctness()
    except pytest.skip.Exception as e:
        print(f"skipped: {e}")
