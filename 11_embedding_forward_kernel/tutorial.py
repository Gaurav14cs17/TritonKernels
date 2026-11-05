"""
Embedding forward and backward in Triton.

Ladder:
  Method 01 - one-hot matrix product / Python scatter-add
  Method 02 - tiled Triton gather + atomic scatter-add
  Method 03 - fused lookup + position add

Run:  python tutorial.py
Notes: ./doc/Embedding_kernal.pdf
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd().parent))

import pytest
import torch
import triton
import triton.language as tl


def _active_device():
    """Pick Triton's active device, else CUDA, else CPU (study helper)."""
    try:
        return triton.runtime.driver.active.get_active_torch_device()
    except Exception:
        pass
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


DEVICE = _active_device()


def require_gpu() -> None:
    """Skip a GPU-only test when no accelerator is present."""
    if not torch.cuda.is_available() and getattr(DEVICE, "type", None) != "cuda":
        pytest.skip(f"GPU not available (active device: {DEVICE})")


def get_dtype_flag(dtype: torch.dtype) -> int:
    """Encode the element type for the kernel: 0=fp32, 1=fp16, 2=bf16."""
    if dtype == torch.float32:
        return 0
    if dtype == torch.float16:
        return 1
    if dtype == torch.bfloat16:
        return 2
    raise TypeError(f"unsupported dtype: {dtype}")


print("device:", DEVICE, "| dtype flag fp32:", get_dtype_flag(torch.float32))

# %%
# Tiny case from the notes: V=5, D=3, M=4.
table = torch.arange(1, 16, dtype=torch.float32).view(5, 3)
ids = torch.tensor([2, 0, 1, 2])
grad = torch.arange(1, 13, dtype=torch.float32).view(4, 3)

one_hot = torch.zeros(4, 5)
one_hot[torch.arange(4), ids] = 1.0

lookup = table[ids]
matmul_form = one_hot @ table
print("P =\n", one_hot)
print("X = W[I] =\n", lookup)
print("P @ W   =\n", matmul_form)
print("same:", torch.equal(lookup, matmul_form))

# Backward: scatter-add the gradient rows into dW.
dW = torch.zeros_like(table)
dW.index_add_(0, ids, grad)
print("dW =\n", dW)
print("expected dW[2] = G[0] + G[3] =", grad[0] + grad[3])

# %%
def naive_embedding_pseudo(table: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    """Method 01 forward: one Python iteration per token (the gather definition)."""
    table = torch.as_tensor(table)
    ids = torch.as_tensor(ids)
    out = torch.empty(*ids.shape, table.shape[1], dtype=table.dtype, device=table.device)
    flat_ids = ids.reshape(-1)
    flat_out = out.reshape(-1, table.shape[1])
    for m in range(flat_ids.numel()):
        flat_out[m] = table[int(flat_ids[m])]
    return out


def one_hot_embedding_pseudo(table: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    """Method 01 as a matrix product: X = P W with P[m, ids[m]] = 1."""
    table = torch.as_tensor(table)
    ids = torch.as_tensor(ids)
    flat_ids = ids.reshape(-1)
    P = torch.zeros(flat_ids.numel(), table.shape[0], dtype=table.dtype, device=table.device)
    P[torch.arange(flat_ids.numel()), flat_ids] = 1.0
    return (P @ table).reshape(*ids.shape, table.shape[1])


def naive_embedding_backward_pseudo(
    grad_out: torch.Tensor, vocab_size: int, ids: torch.Tensor
) -> torch.Tensor:
    """Method 01 backward: dW[j] += G[m] for every token m with ids[m] == j."""
    grad_out = torch.as_tensor(grad_out)
    ids = torch.as_tensor(ids)
    dW = torch.zeros(
        vocab_size, grad_out.shape[-1], dtype=grad_out.dtype, device=grad_out.device
    )
    grad_rows = grad_out.reshape(-1, grad_out.shape[-1])
    flat_ids = ids.reshape(-1)
    for m in range(flat_ids.numel()):
        dW[int(flat_ids[m])] += grad_rows[m]
    return dW

# %%
def test_pseudo_matches_torch() -> None:
    """CPU-runnable: one-hot, Python gather, and Python scatter match PyTorch."""
    torch.manual_seed(0)
    table = torch.randn(11, 5)
    ids = torch.tensor([[3, 1, 4], [1, 5, 9]])
    ref = table[ids]

    assert torch.allclose(naive_embedding_pseudo(table, ids), ref)
    assert torch.allclose(one_hot_embedding_pseudo(table, ids), ref)

    grad = torch.randn_like(ref)
    ref_dw = torch.zeros_like(table)
    ref_dw.index_add_(0, ids.reshape(-1), grad.reshape(-1, 5))
    assert torch.allclose(
        naive_embedding_backward_pseudo(grad, table.shape[0], ids), ref_dw
    )
    print("method 01: forward and backward OK")


def test_worked_example() -> None:
    """Check the exact numbers printed in the notes."""
    table = torch.arange(1, 16, dtype=torch.float32).view(5, 3)
    ids = torch.tensor([2, 0, 1, 2])
    grad = torch.arange(1, 13, dtype=torch.float32).view(4, 3)

    expected_x = torch.tensor([[7, 8, 9], [1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float32)
    assert torch.equal(table[ids], expected_x)

    expected_dw = torch.zeros(5, 3)
    expected_dw[0] = torch.tensor([4.0, 5.0, 6.0])
    expected_dw[1] = torch.tensor([7.0, 8.0, 9.0])
    expected_dw[2] = torch.tensor([11.0, 13.0, 15.0])
    assert torch.equal(naive_embedding_backward_pseudo(grad, 5, ids), expected_dw)
    print("worked example: X and dW match the notes")


test_pseudo_matches_torch()
test_worked_example()

# %%
@triton.jit
def embedding_forward_kernel(
    table_ptr,          # W: [V, D]
    ids_ptr,            # I: [M] int32
    out_ptr,            # X: [M, D]
    n_tokens,           # M (runtime)
    embed_dim,          # D (runtime)
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    DTYPE_FLAG: tl.constexpr,   # 0=fp32, 1=fp16, 2=bf16
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # The notes dispatch on DTYPE_FLAG so one kernel serves all three dtypes.
    if DTYPE_FLAG == 0:
        table_ptr = tl.cast(table_ptr, tl.pointer_type(tl.float32))
        out_ptr = tl.cast(out_ptr, tl.pointer_type(tl.float32))
    elif DTYPE_FLAG == 1:
        table_ptr = tl.cast(table_ptr, tl.pointer_type(tl.float16))
        out_ptr = tl.cast(out_ptr, tl.pointer_type(tl.float16))
    else:
        table_ptr = tl.cast(table_ptr, tl.pointer_type(tl.bfloat16))
        out_ptr = tl.cast(out_ptr, tl.pointer_type(tl.bfloat16))
    ids_ptr = tl.cast(ids_ptr, tl.pointer_type(tl.int32))

    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    row_mask = rows < n_tokens
    col_mask = cols < embed_dim
    mask = row_mask[:, None] & col_mask[None, :]

    # Load the token ids for this row tile; masked lanes use id 0.
    ids = tl.load(ids_ptr + rows, mask=row_mask, other=0)

    # Gather W[ids[m], cols] for every (m, col) in the tile.
    table_offsets = ids[:, None].to(tl.int64) * embed_dim + cols[None, :]
    values = tl.load(table_ptr + table_offsets, mask=mask, other=0.0)

    # X[m, col] = the gathered value.
    out_offsets = rows[:, None].to(tl.int64) * embed_dim + cols[None, :]
    tl.store(out_ptr + out_offsets, values, mask=mask)

# %%
def embedding_forward(
    table: torch.Tensor,
    ids: torch.Tensor,
    BLOCK_M: int = 128,
    BLOCK_N: int = 128,
    check_indices: bool = False,
) -> torch.Tensor:
    """Method 02 forward: X = W[I] with a 2-D grid over (tokens, embedding dim)."""
    assert table.ndim == 2, "table must be [V, D]"
    table = table.contiguous()
    ids = ids.contiguous()

    if check_indices and ids.numel() > 0:
        # A bounds check forces a GPU sync, so it is opt-in (as in the notes).
        assert ids.min() >= 0, "negative token id"
        assert ids.max() < table.shape[0], "token id out of range"

    ids_i32 = ids.to(torch.int32)
    out = torch.empty(*ids.shape, table.shape[1], device=table.device, dtype=table.dtype)
    n_tokens = ids.numel()
    if n_tokens == 0:
        return out

    grid = (
        triton.cdiv(n_tokens, BLOCK_M),
        triton.cdiv(table.shape[1], BLOCK_N),
    )
    embedding_forward_kernel[grid](
        table,
        ids_i32,
        out,
        n_tokens,
        table.shape[1],
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        DTYPE_FLAG=get_dtype_flag(table.dtype),
    )
    return out

# %%
@triton.jit
def embedding_backward_kernel(
    grad_out_ptr,       # G: [M, D]
    grad_weight_ptr,    # dW: [V, D]
    ids_ptr,            # I: [M] int32
    n_tokens,           # M
    embed_dim,          # D
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    DTYPE_FLAG: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    if DTYPE_FLAG == 0:
        grad_out_ptr = tl.cast(grad_out_ptr, tl.pointer_type(tl.float32))
        grad_weight_ptr = tl.cast(grad_weight_ptr, tl.pointer_type(tl.float32))
    elif DTYPE_FLAG == 1:
        grad_out_ptr = tl.cast(grad_out_ptr, tl.pointer_type(tl.float16))
        grad_weight_ptr = tl.cast(grad_weight_ptr, tl.pointer_type(tl.float16))
    else:
        grad_out_ptr = tl.cast(grad_out_ptr, tl.pointer_type(tl.bfloat16))
        grad_weight_ptr = tl.cast(grad_weight_ptr, tl.pointer_type(tl.bfloat16))
    ids_ptr = tl.cast(ids_ptr, tl.pointer_type(tl.int32))

    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    row_mask = rows < n_tokens
    col_mask = cols < embed_dim
    mask = row_mask[:, None] & col_mask[None, :]

    ids = tl.load(ids_ptr + rows, mask=row_mask, other=0)

    # Read the gradient rows owned by this tile.
    grad_offsets = rows[:, None].to(tl.int64) * embed_dim + cols[None, :]
    grad = tl.load(grad_out_ptr + grad_offsets, mask=mask, other=0.0)

    # dW[ids[m], cols] += G[m, cols].  Atomic because ids can repeat.
    weight_offsets = ids[:, None].to(tl.int64) * embed_dim + cols[None, :]
    tl.atomic_add(grad_weight_ptr + weight_offsets, grad, mask=mask)

# %%
def embedding_backward(
    grad_out: torch.Tensor,
    vocab_size: int,
    ids: torch.Tensor,
    BLOCK_M: int = 128,
    BLOCK_N: int = 128,
    check_indices: bool = False,
) -> torch.Tensor:
    """Method 02 backward: dW[I[m]] += G[m], accumulated with atomics."""
    grad_out = grad_out.contiguous()
    ids = ids.contiguous()

    if check_indices and ids.numel() > 0:
        assert ids.min() >= 0, "negative token id"
        assert ids.max() < vocab_size, "token id out of range"

    out_dtype = grad_out.dtype
    if out_dtype == torch.bfloat16:
        # bf16 atomics are slow/unsupported; accumulate in fp32, cast back after.
        grad_out = grad_out.float()

    ids_i32 = ids.to(torch.int32)
    grad_weight = torch.zeros(
        vocab_size, grad_out.shape[-1], device=grad_out.device, dtype=grad_out.dtype
    )
    n_tokens = ids.numel()
    if n_tokens == 0:
        return grad_weight.to(out_dtype)

    grid = (
        triton.cdiv(n_tokens, BLOCK_M),
        triton.cdiv(grad_out.shape[-1], BLOCK_N),
    )
    embedding_backward_kernel[grid](
        grad_out,
        grad_weight,
        ids_i32,
        n_tokens,
        grad_out.shape[-1],
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        DTYPE_FLAG=get_dtype_flag(grad_out.dtype),
    )
    return grad_weight.to(out_dtype)

# %%
def _assert_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    """Tolerance depends on the dtype (fp16/bf16 are looser)."""
    if actual.dtype in (torch.float16, torch.bfloat16):
        torch.testing.assert_close(actual.float(), expected.float(), atol=2e-2, rtol=2e-2)
    else:
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


def test_correctness() -> None:
    """GPU: forward matches table[ids] for fp32/fp16/bf16 and odd sizes."""
    require_gpu()
    torch.manual_seed(0)
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        table = torch.randn(37, 19, device=DEVICE, dtype=dtype)
        ids = torch.randint(0, 37, (5, 7), device=DEVICE)
        out = embedding_forward(table, ids, BLOCK_M=4, BLOCK_N=8)
        _assert_close(out, table[ids])
        print(f"forward {dtype}: OK")

    # Non-contiguous table (a transposed view) must still work.
    big = torch.randn(19, 41, device=DEVICE)
    table_nc = big[:, :37].t()            # shape (37, 19), stride is not (19, 1)
    ids = torch.randint(0, 37, (3, 3), device=DEVICE)
    _assert_close(embedding_forward(table_nc, ids), table_nc[ids])
    print("forward non-contiguous table: OK")


def test_backward() -> None:
    """GPU: dW equals the scatter-add reference, including duplicate ids."""
    require_gpu()
    torch.manual_seed(0)
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        table = torch.randn(23, 11, device=DEVICE, dtype=dtype)
        ids = torch.randint(0, 23, (4, 6), device=DEVICE)
        grad = torch.randn(4, 6, 11, device=DEVICE, dtype=dtype)

        dW = embedding_backward(grad, table.shape[0], ids)
        ref = torch.zeros_like(table)
        ref.index_add_(0, ids.reshape(-1), grad.reshape(-1, 11))
        _assert_close(dW, ref)
        print(f"backward {dtype}: OK")

    # Many tokens share ids 0, 1 -> the atomic add must sum all of them.
    ids = torch.tensor([[0, 0, 0, 1, 1]], device=DEVICE)
    grad = torch.ones(1, 5, 7, device=DEVICE)
    dW = embedding_backward(grad, 3, ids)
    assert torch.allclose(dW[0], torch.full((7,), 3.0, device=DEVICE))
    assert torch.allclose(dW[1], torch.full((7,), 2.0, device=DEVICE))
    assert torch.allclose(dW[2], torch.zeros(7, device=DEVICE))
    print("backward duplicate-id collision: OK")

# %%
@triton.jit
def fused_embedding_position_kernel(
    table_ptr,          # W: [V, D]
    ids_ptr,            # I: [M] int32
    pos_ptr,            # P: [M, D]
    out_ptr,            # H = W[I] + P: [M, D]
    n_tokens,
    embed_dim,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    DTYPE_FLAG: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    if DTYPE_FLAG == 0:
        table_ptr = tl.cast(table_ptr, tl.pointer_type(tl.float32))
        pos_ptr = tl.cast(pos_ptr, tl.pointer_type(tl.float32))
        out_ptr = tl.cast(out_ptr, tl.pointer_type(tl.float32))
    elif DTYPE_FLAG == 1:
        table_ptr = tl.cast(table_ptr, tl.pointer_type(tl.float16))
        pos_ptr = tl.cast(pos_ptr, tl.pointer_type(tl.float16))
        out_ptr = tl.cast(out_ptr, tl.pointer_type(tl.float16))
    else:
        table_ptr = tl.cast(table_ptr, tl.pointer_type(tl.bfloat16))
        pos_ptr = tl.cast(pos_ptr, tl.pointer_type(tl.bfloat16))
        out_ptr = tl.cast(out_ptr, tl.pointer_type(tl.bfloat16))
    ids_ptr = tl.cast(ids_ptr, tl.pointer_type(tl.int32))

    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    row_mask = rows < n_tokens
    col_mask = cols < embed_dim
    mask = row_mask[:, None] & col_mask[None, :]

    ids = tl.load(ids_ptr + rows, mask=row_mask, other=0)
    table_offsets = ids[:, None].to(tl.int64) * embed_dim + cols[None, :]
    values = tl.load(table_ptr + table_offsets, mask=mask, other=0.0)

    flat_offsets = rows[:, None].to(tl.int64) * embed_dim + cols[None, :]
    position = tl.load(pos_ptr + flat_offsets, mask=mask, other=0.0)

    # Add in fp32, then cast once: one store, and no intermediate X in HBM.
    total = values.to(tl.float32) + position.to(tl.float32)
    tl.store(out_ptr + flat_offsets, total.to(out_ptr.dtype.element_ty), mask=mask)

# %%
def fused_embedding_forward(
    table: torch.Tensor,
    ids: torch.Tensor,
    position: torch.Tensor,
    BLOCK_M: int = 128,
    BLOCK_N: int = 128,
) -> torch.Tensor:
    """Method 03: H = W[I] + position in one kernel (same-shape position)."""
    assert table.ndim == 2, "table must be [V, D]"
    table = table.contiguous()
    ids = ids.contiguous()
    position = position.contiguous()
    assert position.shape == (*ids.shape, table.shape[1]), (
        f"position must have shape {(*ids.shape, table.shape[1])}, got {tuple(position.shape)}"
    )
    assert position.dtype == table.dtype, "position and table dtypes must match"

    out = torch.empty_like(position)
    n_tokens = ids.numel()
    if n_tokens == 0:
        return out

    grid = (
        triton.cdiv(n_tokens, BLOCK_M),
        triton.cdiv(table.shape[1], BLOCK_N),
    )
    fused_embedding_position_kernel[grid](
        table,
        ids.to(torch.int32),
        position,
        out,
        n_tokens,
        table.shape[1],
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        DTYPE_FLAG=get_dtype_flag(table.dtype),
    )
    return out


def fused_embedding_backward(
    grad_out: torch.Tensor,
    vocab_size: int,
    ids: torch.Tensor,
    BLOCK_M: int = 128,
    BLOCK_N: int = 128,
):
    """Method 03 backward: dW is a scatter-add; dP = dH for a same-shape position."""
    d_weight = embedding_backward(grad_out, vocab_size, ids, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N)
    d_position = grad_out.clone()   # dH/dP = 1
    return d_weight, d_position

# %%
def test_fused_position() -> None:
    """GPU: fused H = W[I] + P matches PyTorch and its gradient is correct."""
    require_gpu()
    torch.manual_seed(0)
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        table = torch.randn(29, 13, device=DEVICE, dtype=dtype)
        ids = torch.randint(0, 29, (3, 5), device=DEVICE)
        position = torch.randn(3, 5, 13, device=DEVICE, dtype=dtype)

        out = fused_embedding_forward(table, ids, position, BLOCK_M=4, BLOCK_N=8)
        _assert_close(out, table[ids] + position)

        grad = torch.randn_like(out)
        d_weight, d_position = fused_embedding_backward(grad, table.shape[0], ids)
        ref_dw = torch.zeros_like(table)
        ref_dw.index_add_(0, ids.reshape(-1), grad.reshape(-1, 13))
        _assert_close(d_weight, ref_dw)
        _assert_close(d_position, grad)
        print(f"fused forward/backward {dtype}: OK")

# %%
def embedding_bytes(vocab: int, dim: int, tokens: int, itemsize: int):
    """Analytic traffic for one forward + one backward (bytes)."""
    forward = 2 * tokens * dim * itemsize          # read rows + write output
    backward = (vocab * dim + tokens * dim) * itemsize  # touch table + read grad
    return forward, backward


V, D, M = 50_000, 1_024, 2_048
fwd, bwd = embedding_bytes(V, D, M, itemsize=2)   # fp16
print(f"forward traffic : {fwd / 1e6:8.2f} MB")
print(f"backward traffic: {bwd / 1e6:8.2f} MB")
print(f"backward / forward ratio: {bwd / fwd:,.1f}x")
print(f"backward table alone    : {V * D * 2 / 1e6:8.2f} MB")

# %%
print("Running CPU tests ...")
test_pseudo_matches_torch()
test_worked_example()

print("\nRunning GPU tests (skipped without CUDA/ROCm) ...")
for fn in (test_correctness, test_backward, test_fused_position):
    try:
        fn()
    except pytest.skip.Exception as exc:
        print(f"SKIP {fn.__name__}: {exc}")


# %%
if __name__ == "__main__":
    print("Running Tests...")
    test_pseudo_matches_torch()
    test_worked_example()
    for fn in (test_correctness, test_backward, test_fused_position):
        try:
            fn()
        except pytest.skip.Exception as exc:
            print(f"skipped: {exc}")
