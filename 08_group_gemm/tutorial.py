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

_matmul_path = Path(__file__).resolve().parents[1] / "03_matrix_multiplication" / "tutorial.py"
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


# %%
def test_pseudo_matches_torch() -> None:
    torch.manual_seed(0)
    problems = [
        (torch.randn(32, 16), torch.randn(16, 24)),
        (torch.randn(8, 32), torch.randn(32, 8)),
    ]
    outs = naive_group_gemm_pseudo(problems)
    for (a, b), c in zip(problems, outs):
        assert torch.allclose(c, a @ b, atol=1e-5)


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


# %%
if __name__ == "__main__":
    print("Running Tests...")
    test_pseudo_matches_torch()
    try:
        test_correctness()
    except pytest.skip.Exception as e:
        print(f"skipped: {e}")
