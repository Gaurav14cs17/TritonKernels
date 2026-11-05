# %% [markdown]
# # 05 — Layer Normalization
# Naive multi-pass → fused per-row pseudo → Triton forward kernel.

"""
LayerNorm as described in Triton Docs
https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html

For each row x of length N:
  mean = sum(x) / N
  var  = sum((x - mean)^2) / N
  y    = (x - mean) / sqrt(var + eps) * w + b

A naive PyTorch implementation streams the row multiple times (mean, var,
normalize). A fused kernel keeps the row in SRAM and does the math in one go.
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



# %%
def naive_layer_norm_pseudo(x, weight, bias, eps=1e-5):
    """
    Separate passes over each row — easy to read, poor memory traffic.

    Pass 1: mean
    Pass 2: variance
    Pass 3: normalize + affine
    """
    M, N = x.shape
    y = torch.empty_like(x)
    for m in range(M):
        row = x[m]
        mean = row.sum() / N
        var = ((row - mean) ** 2).sum() / N
        y[m] = (row - mean) / torch.sqrt(var + eps) * weight + bias
    return y


# %%
def fused_layer_norm_pseudo(x, weight, bias, eps=1e-5):
    """
    Still Python, but one function per row — mirrors the fused Triton program
    that owns a single row (pid = row index).
    """
    M, N = x.shape
    y = torch.empty_like(x)
    for m in range(M):
        row = x[m].to(torch.float32)
        mean = row.sum() / N
        centered = row - mean
        var = (centered * centered).sum() / N
        rstd = 1.0 / torch.sqrt(var + eps)
        y[m] = (centered * rstd * weight + bias).to(x.dtype)
    return y


# %%
@triton.jit
def _layer_norm_fwd_kernel(
    Y,
    X,
    W,
    B,
    Mean,
    Rstd,
    stride,
    N,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    # One program per row — same ownership as fused_layer_norm_pseudo
    row = tl.program_id(0)
    Y += row * stride
    X += row * stride
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N
    x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / N
    x_centered = tl.where(mask, x - mean, 0.0)
    var = tl.sum(x_centered * x_centered, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    y = (x - mean) * rstd * w + b
    tl.store(Mean + row, mean)
    tl.store(Rstd + row, rstd)
    tl.store(Y + cols, y, mask=mask)


def layer_norm(x, weight, bias, eps=1e-5):
    M, N = x.shape
    x = x.contiguous()          # the kernel assumes a dense trailing dimension
    y = torch.empty_like(x)
    mean = torch.empty((M,), device=x.device, dtype=torch.float32)
    rstd = torch.empty((M,), device=x.device, dtype=torch.float32)
    BLOCK_SIZE = triton.next_power_of_2(N)
    _layer_norm_fwd_kernel[(M,)](
        y,
        x,
        weight,
        bias,
        mean,
        rstd,
        x.stride(0),
        N,
        eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return y


# %%
def test_pseudo_matches_torch() -> None:
    torch.manual_seed(0)
    x = torch.randn(4, 32)
    w = torch.randn(32)
    b = torch.randn(32)
    ref = torch.nn.functional.layer_norm(x, (32,), w, b)
    assert torch.allclose(naive_layer_norm_pseudo(x, w, b), ref, atol=1e-4, rtol=1e-4)
    assert torch.allclose(fused_layer_norm_pseudo(x, w, b), ref, atol=1e-4, rtol=1e-4)


def test_correctness() -> None:
    require_gpu()
    torch.manual_seed(0)
    x = torch.randn(128, 256, device=DEVICE)
    w = torch.randn(256, device=DEVICE)
    b = torch.randn(256, device=DEVICE)
    y = layer_norm(x, w, b)
    ref = torch.nn.functional.layer_norm(x, (256,), w, b)
    assert torch.allclose(y, ref, atol=1e-2, rtol=1e-2)
    print("layer_norm OK")


# %%
def test_mask_tail() -> None:
    """GPU: N=255 is not a power of two, so the mask path is live."""
    require_gpu()
    torch.manual_seed(0)
    x = torch.randn(8, 255, device=DEVICE)
    weight = torch.randn(255, device=DEVICE)
    bias = torch.randn(255, device=DEVICE)
    ref = torch.nn.functional.layer_norm(x, (255,), weight, bias, 1e-5)
    out = layer_norm(x, weight, bias)
    assert torch.allclose(out, ref, atol=1e-4, rtol=1e-4)
    print("layer-norm N=255 mask tail OK")


# %%
if __name__ == "__main__":
    print("Running Tests...")
    test_pseudo_matches_torch()
    for fn in (test_correctness, test_mask_tail):
        try:
            fn()
        except pytest.skip.Exception as e:
            print(f"skipped {fn.__name__}: {e}")
