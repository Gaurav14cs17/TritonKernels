# %% [markdown]
# # 08 — Group GEMM
# Naive sequential torch → sequential Triton matmul → (next) single-launch Group GEMM.
#
# Official: https://triton-lang.org/main/getting-started/tutorials/08-group-gemm.html

"""
Group GEMM = many independent matmuls with different (M, N, K) shapes.

Naive: for each problem, launch torch.matmul (or our Triton matmul).
Cost: one kernel launch + autotune bookkeeping per problem.

Next step in the official tutorial: pack problem metadata and run a *single*
Triton launch that walks the group. Here we teach the sequential path that
reuses tutorial 03's grouped (L2-friendly) matmul.
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
def naive_group_gemm_pseudo(problems):
    """
    Sequential torch.matmul for each (A, B) pair.

    problems: list of (A, B) with A:[Mi, Ki], B:[Ki, Ni]

    Each call is a separate cuBLAS/torch launch. Fine for a few large GEMMs;
    painful when you have hundreds of tiny ones (launch overhead dominates).
    """
    return [torch.matmul(a, b) for a, b in problems]


# %%
def group_gemm_sequential(problems):
    """
    Same loop, but each GEMM uses tutorial 03's Triton grouped matmul.

    Still one launch per problem — the *grouping* inside each matmul is the
    L2-friendly program ordering from tutorial 03, not "group of GEMMs".

    True Group GEMM (official tutorial) packs many problems into one grid so
    the GPU stays busy without repeated host→device launch overhead.
    """
    return [matmul(a, b) for a, b in problems]


@triton.jit
def group_gemm_packed_kernel(
    A_all,
    B_all,
    C_all,
    M_ptr,
    N_ptr,
    K_ptr,
    A_off_ptr,
    B_off_ptr,
    C_off_ptr,
    tile_off_ptr,
    group_ids,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Method 03: one launch over all groups' tiles.

    Every group is packed into contiguous flat buffers.  The host supplies the
    per-group metadata, base offsets, tile prefix sums, and a per-tile group id.
    """
    pid = tl.program_id(0)
    g = tl.load(group_ids + pid)
    local = pid - tl.load(tile_off_ptr + g)

    M = tl.load(M_ptr + g)
    N = tl.load(N_ptr + g)
    K = tl.load(K_ptr + g)

    num_n = tl.cdiv(N, BLOCK_N)
    pid_m = local // num_n
    pid_n = local % num_n

    a_base = A_all + tl.load(A_off_ptr + g)
    b_base = B_all + tl.load(B_off_ptr + g)
    c_base = C_all + tl.load(C_off_ptr + g)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_base + offs_m[:, None] * K + offs_k[None, :]
    b_ptrs = b_base + offs_k[:, None] * N + offs_n[None, :]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        remaining = K - k * BLOCK_K
        a = tl.load(
            a_ptrs,
            mask=(offs_m[:, None] < M) & (offs_k[None, :] < remaining),
            other=0.0,
        )
        b = tl.load(
            b_ptrs,
            mask=(offs_k[:, None] < remaining) & (offs_n[None, :] < N),
            other=0.0,
        )
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K
        b_ptrs += BLOCK_K * N

    c = acc.to(C_all.dtype.element_ty)
    c_ptrs = c_base + offs_m[:, None] * N + offs_n[None, :]
    tl.store(
        c_ptrs,
        c,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


def group_gemm_one_launch(problems, BLOCK_M=64, BLOCK_N=64, BLOCK_K=32):
    """Method 03 launcher: pack groups, build metadata, launch one grid."""
    if not problems:
        return []

    device = problems[0][0].device
    dtype = problems[0][0].dtype
    As = [a.contiguous() for a, _ in problems]
    Bs = [b.contiguous() for _, b in problems]
    Ms = [a.shape[0] for a in As]
    Ns = [b.shape[1] for b in Bs]
    Ks = [a.shape[1] for a in As]

    C_all = torch.empty(
        sum(m * n for m, n in zip(Ms, Ns)), device=device, dtype=dtype
    )
    Cs, c_off, off = [], [], 0
    for m, n in zip(Ms, Ns):
        Cs.append(C_all[off : off + m * n].view(m, n))
        c_off.append(off)
        off += m * n

    A_all = torch.cat([a.reshape(-1) for a in As])
    B_all = torch.cat([b.reshape(-1) for b in Bs])

    a_off, b_off, off = [], [], 0
    for a in As:
        a_off.append(off)
        off += a.numel()
    off = 0
    for b in Bs:
        b_off.append(off)
        off += b.numel()

    counts = [
        triton.cdiv(m, BLOCK_M) * triton.cdiv(n, BLOCK_N)
        for m, n in zip(Ms, Ns)
    ]
    tile_off = [0]
    for c in counts:
        tile_off.append(tile_off[-1] + c)
    total_tiles = tile_off[-1]

    def i32(values):
        return torch.tensor(values, device=device, dtype=torch.int32)

    group_ids = torch.repeat_interleave(
        torch.arange(len(problems), device=device),
        torch.tensor(counts, device=device, dtype=torch.int64),
    ).to(torch.int32)

    group_gemm_packed_kernel[(total_tiles,)](
        A_all,
        B_all,
        C_all,
        i32(Ms),
        i32(Ns),
        i32(Ks),
        i32(a_off),
        i32(b_off),
        i32(c_off),
        i32(tile_off),
        group_ids,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )
    return Cs


# %%
def test_pseudo_matches_torch() -> None:
    torch.manual_seed(0)
    problems = [
        (torch.randn(32, 16), torch.randn(16, 24)),
        (torch.randn(8, 32), torch.randn(32, 8)),
    ]
    outs = naive_group_gemm_pseudo(problems)
    for (a, b), c in zip(problems, outs):
        # Independent formulation, so this is not just the function under test.
        ref = torch.einsum("ik,kj->ij", a, b)
        assert torch.allclose(c, ref, atol=1e-5)


def test_correctness() -> None:
    require_gpu()
    problems = [
        (
            torch.randn(128, 64, device=DEVICE, dtype=torch.float16),
            torch.randn(64, 256, device=DEVICE, dtype=torch.float16),
        ),
        (
            torch.randn(64, 128, device=DEVICE, dtype=torch.float16),
            torch.randn(128, 32, device=DEVICE, dtype=torch.float16),
        ),
    ]
    outs = group_gemm_sequential(problems)
    for (a, b), c in zip(problems, outs):
        ref = torch.matmul(a, b)
        assert torch.allclose(c.float(), ref.float(), atol=1e-2, rtol=1e-2)
    print("group gemm (sequential) OK", len(outs))


def test_group_gemm_one_launch() -> None:
    """Method 03 (one launch) vs the per-group references."""
    require_gpu()
    torch.manual_seed(0)
    problems = [
        (
            torch.randn(128, 64, device=DEVICE, dtype=torch.float16),
            torch.randn(64, 256, device=DEVICE, dtype=torch.float16),
        ),
        (
            torch.randn(64, 128, device=DEVICE, dtype=torch.float16),
            torch.randn(128, 32, device=DEVICE, dtype=torch.float16),
        ),
    ]
    outs = group_gemm_one_launch(problems)
    for (a, b), c in zip(problems, outs):
        ref = torch.matmul(a, b)
        assert torch.allclose(c.float(), ref.float(), atol=1e-2, rtol=1e-2)
    print("group GEMM one-launch OK")


# %%
if __name__ == "__main__":
    print("Running Tests...")
    test_pseudo_matches_torch()
    for fn in (test_correctness, test_group_gemm_one_launch):
        try:
            fn()
        except pytest.skip.Exception as e:
            print(f"skipped {fn.__name__}: {e}")
