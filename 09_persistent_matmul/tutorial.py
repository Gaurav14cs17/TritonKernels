# %% [markdown]
# # 09 — Persistent Matmul
# Persistent Python pseudo → classic blocked baseline (tutorial 03) → small persistent Triton kernel.
#
# Official: https://triton-lang.org/main/getting-started/tutorials/09-persistent-matmul.html

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

####################################################
### NAIVE: ONE PROGRAM PER OUTPUT TILE             ###
####################################################


# %%
def naive_blocked_matmul(A, B):
    """
    Classic blocked launch: grid ≈ (#tiles_M) * (#tiles_N).
    Each program owns exactly one output tile, then exits.

    We reuse tutorial 03's `matmul` (fused grouped GEMM) as that baseline.
    """
    return matmul(A, B)


# %%
def persistent_matmul_pseudo(A, B, BLOCK_M, BLOCK_N, num_programs):
    """
    Persistent idea in Python:

      - Launch only `num_programs` workers (often ≈ #SMs), not one per tile.
      - Total tiles = grid_m * grid_n.
      - Program `pid` walks tiles pid, pid+num_programs, pid+2*num_programs, ...

    Same math as blocked matmul; different work distribution. Real kernels also
    wave-specialize / pipeline loads; here we just show the tile iteration.
    """
    M, K = A.shape
    K2, N = B.shape
    assert K == K2
    assert M % BLOCK_M == 0 and N % BLOCK_N == 0

    C = torch.zeros(M, N, device=A.device, dtype=A.dtype)
    grid_m = M // BLOCK_M
    grid_n = N // BLOCK_N
    num_tiles = grid_m * grid_n

    for pid in range(num_programs):
        for tile_id in range(pid, num_tiles, num_programs):
            pid_m = tile_id // grid_n
            pid_n = tile_id % grid_n
            m0 = pid_m * BLOCK_M
            n0 = pid_n * BLOCK_N
            C[m0 : m0 + BLOCK_M, n0 : n0 + BLOCK_N] = (
                A[m0 : m0 + BLOCK_M, :] @ B[:, n0 : n0 + BLOCK_N]
            )
    return C


######################################################
### TRITON: SMALL PERSISTENT MATMUL KERNEL         ###
######################################################


# %%
@triton.jit
def persistent_matmul_kernel(
    A_ptr,
    B_ptr,
    C_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Few programs; each strides across output tiles with tl.num_programs."""
    pid = tl.program_id(axis=0)
    num_pid = tl.num_programs(axis=0)

    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_tiles = num_pid_m * num_pid_n

    for tile_id in range(pid, num_tiles, num_pid):
        pid_m = tile_id // num_pid_n
        pid_n = tile_id % num_pid_n

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        a_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        b_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k in range(0, tl.cdiv(K, BLOCK_K)):
            k_remaining = K - k * BLOCK_K
            a = tl.load(
                a_ptrs,
                mask=(offs_m[:, None] < M) & (offs_k[None, :] < k_remaining),
                other=0.0,
            )
            b = tl.load(
                b_ptrs,
                mask=(offs_k[:, None] < k_remaining) & (offs_n[None, :] < N),
                other=0.0,
            )
            acc += tl.dot(a, b)
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk

        c = acc.to(C_ptr.dtype.element_ty)
        c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
        tl.store(c_ptrs, c, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def persistent_matmul(a, b, num_programs=None, BLOCK_M=64, BLOCK_N=64, BLOCK_K=32):
    """Launch a small persistent grid; each program loops over many tiles."""
    M, K = a.shape
    K2, N = b.shape
    assert K == K2
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    num_tiles = triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)
    if num_programs is None:
        try:
            sms = torch.cuda.get_device_properties(a.device).multi_processor_count
        except Exception:
            sms = 8  # CPU / interpreter fallback
        num_programs = min(num_tiles, sms)
    persistent_matmul_kernel[(num_programs,)](
        a,
        b,
        c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )
    return c


# %%
### TESTS ###
def test_pseudo_matches_torch() -> None:
    torch.manual_seed(0)
    a = torch.randn(64, 32)
    b = torch.randn(32, 48)
    ref = a @ b
    out = persistent_matmul_pseudo(a, b, BLOCK_M=32, BLOCK_N=16, num_programs=3)
    assert torch.allclose(out, ref, atol=1e-5)


def test_correctness() -> None:
    require_gpu()
    torch.manual_seed(0)
    a = torch.randn(256, 128, device=DEVICE, dtype=torch.float16)
    b = torch.randn(128, 256, device=DEVICE, dtype=torch.float16)
    ref = torch.matmul(a, b)

    baseline = naive_blocked_matmul(a, b)
    assert torch.allclose(baseline.float(), ref.float(), atol=1e-2, rtol=1e-2)

    out = persistent_matmul(a, b, num_programs=4, BLOCK_M=64, BLOCK_N=64, BLOCK_K=32)
    assert torch.allclose(out.float(), ref.float(), atol=1e-2, rtol=1e-2)
    print("persistent matmul OK")


# %%
if __name__ == "__main__":
    print("Running Tests...")
    test_pseudo_matches_torch()
    try:
        test_correctness()
    except pytest.skip.Exception as e:
        print(f"skipped: {e}")
