# %% [markdown]
# # 06 — Fused Attention
# Naive → blocked tiles → Triton online-softmax forward (FlashAttention-style sketch).
#
# Official: https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html

# %%
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import math

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
### PSEUDOCODE: FULL ATTENTION (MATERIALIZES S) ###
####################################################


# %%
def naive_attention_pseudo(Q, K, V):
    """
    Standard scaled-dot-product attention for one (or batched) head:

        S = Q @ K^T / sqrt(D)     # [..., N, N]  — O(N^2) scores in memory
        P = softmax(S, dim=-1)    # still O(N^2)
        O = P @ V                 # [..., N, D]

    Q, K, V: [..., N, D]  (last two dims are sequence × head dim)

    Teaching point: the N×N score matrix is the memory bottleneck that
    FlashAttention-style kernels avoid by never fully materializing S/P.
    """
    D = Q.shape[-1]
    scale = 1.0 / math.sqrt(D)

    # Full QK^T — allocates an N×N tensor (bad for long sequences)
    S = torch.matmul(Q, K.transpose(-2, -1)) * scale
    P = torch.softmax(S, dim=-1)
    O = torch.matmul(P, V)
    return O


# %%
def blocked_attention_pseudo(Q, K, V, BLOCK_M=32, BLOCK_N=32):
    """
    Same math as naive, but process query rows in BLOCK_M tiles and key/value
    columns in BLOCK_N tiles. Still uses torch matmul on each tile — this is
    conceptual scaffolding for the Triton grid (program_id → query tile).

    For each query block we still form a BLOCK_M × N score strip (or iterate
    K/V blocks with online softmax). Here we keep a simple blocked QK^T for
    clarity; the Triton kernel below upgrades this to online softmax.
    """
    *batch, N, D = Q.shape
    assert N % BLOCK_M == 0 and N % BLOCK_N == 0, "assume divisible for teaching"

    scale = 1.0 / math.sqrt(D)
    O = torch.zeros_like(Q)

    # Flatten leading batch dims so we can loop over heads/batches simply
    Qf = Q.reshape(-1, N, D)
    Kf = K.reshape(-1, N, D)
    Vf = V.reshape(-1, N, D)
    Of = O.reshape(-1, N, D)

    for b in range(Qf.shape[0]):
        for start_m in range(0, N, BLOCK_M):
            q = Qf[b, start_m : start_m + BLOCK_M, :]  # [BLOCK_M, D]
            # Accumulate attention over key blocks with online softmax
            m_i = torch.full((BLOCK_M,), -float("inf"), dtype=torch.float32, device=Q.device)
            l_i = torch.zeros(BLOCK_M, dtype=torch.float32, device=Q.device)
            acc = torch.zeros(BLOCK_M, D, dtype=torch.float32, device=Q.device)

            for start_n in range(0, N, BLOCK_N):
                k = Kf[b, start_n : start_n + BLOCK_N, :]  # [BLOCK_N, D]
                v = Vf[b, start_n : start_n + BLOCK_N, :]  # [BLOCK_N, D]
                qk = (q.float() @ k.float().T) * scale  # [BLOCK_M, BLOCK_N]

                m_ij = qk.max(dim=-1).values
                m_new = torch.maximum(m_i, m_ij)
                alpha = torch.exp(m_i - m_new)
                p = torch.exp(qk - m_new[:, None])
                l_i = l_i * alpha + p.sum(dim=-1)
                m_i = m_new
                acc = acc * alpha[:, None] + p @ v.float()

            Of[b, start_m : start_m + BLOCK_M, :] = (acc / l_i[:, None]).to(Q.dtype)

    return O


######################################################
### TRITON: ONLINE-SOFTMAX FUSED FORWARD ATTENTION ###
######################################################


# %%
@triton.jit
def _attn_fwd_kernel(
    Q,
    K,
    V,
    Out,
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qk,
    stride_kz,
    stride_kh,
    stride_kn,
    stride_kk,
    stride_vz,
    stride_vh,
    stride_vn,
    stride_vk,
    stride_oz,
    stride_oh,
    stride_om,
    stride_ok,
    Z,
    H,
    N_CTX,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):
    """One program = one query tile of BLOCK_M rows for one (batch, head)."""
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H

    q_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh
    k_offset = off_z.to(tl.int64) * stride_kz + off_h.to(tl.int64) * stride_kh
    v_offset = off_z.to(tl.int64) * stride_vz + off_h.to(tl.int64) * stride_vh
    o_offset = off_z.to(tl.int64) * stride_oz + off_h.to(tl.int64) * stride_oh

    # Load Q tile once; stream K/V tiles along the sequence
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    q_ptrs = Q + q_offset + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    q_mask = offs_m[:, None] < N_CTX
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)

    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    scale = 1.0 / tl.sqrt(tl.cast(BLOCK_DMODEL, tl.float32))

    for start_n in range(0, N_CTX, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        offs_n = start_n + tl.arange(0, BLOCK_N)

        k_ptrs = K + k_offset + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kk
        k_mask = offs_n[:, None] < N_CTX
        k = tl.load(k_ptrs, mask=k_mask, other=0.0)

        qk = tl.dot(q, tl.trans(k)) * scale
        # Mask padded key positions so they never win the max / get weight
        qk = tl.where(offs_n[None, :] < N_CTX, qk, float("-inf"))

        m_ij = tl.max(qk, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        m_i = m_new

        v_ptrs = V + v_offset + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk
        v = tl.load(v_ptrs, mask=k_mask, other=0.0)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)

    acc = acc / l_i[:, None]
    o_ptrs = Out + o_offset + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok
    tl.store(o_ptrs, acc.to(Out.dtype.element_ty), mask=q_mask)


def attention(q, k, v, BLOCK_M=64, BLOCK_N=64):
    """Launch fused forward attention. q/k/v: [Z, H, N_CTX, D]."""
    Z, H, N_CTX, D = q.shape
    assert k.shape == v.shape == (Z, H, N_CTX, D)
    out = torch.empty_like(q)
    grid = (triton.cdiv(N_CTX, BLOCK_M), Z * H)
    _attn_fwd_kernel[grid](
        q,
        k,
        v,
        out,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        Z,
        H,
        N_CTX,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_DMODEL=D,
    )
    return out


# Alias used in docs / older snippets
attention_ref = naive_attention_pseudo


# %%
### TESTS ###
def test_pseudo_matches_torch() -> None:
    torch.manual_seed(0)
    Z, H, N, D = 1, 2, 64, 32
    q = torch.randn(Z, H, N, D)
    k = torch.randn(Z, H, N, D)
    v = torch.randn(Z, H, N, D)
    ref = naive_attention_pseudo(q, k, v)
    blocked = blocked_attention_pseudo(q, k, v, BLOCK_M=32, BLOCK_N=32)
    assert torch.allclose(blocked, ref, atol=1e-4, rtol=1e-4)


def test_correctness() -> None:
    require_gpu()
    torch.manual_seed(0)
    Z, H, N_CTX, D = 1, 2, 128, 64
    q = torch.randn(Z, H, N_CTX, D, device=DEVICE, dtype=torch.float16)
    k = torch.randn(Z, H, N_CTX, D, device=DEVICE, dtype=torch.float16)
    v = torch.randn(Z, H, N_CTX, D, device=DEVICE, dtype=torch.float16)
    ref = naive_attention_pseudo(q.float(), k.float(), v.float()).half()
    out = attention(q, k, v, BLOCK_M=64, BLOCK_N=64)
    assert torch.allclose(out, ref, atol=2e-2, rtol=2e-2)
    print(f"fused attention OK  shape={tuple(out.shape)}")


# %%
if __name__ == "__main__":
    print("Running Tests...")
    test_pseudo_matches_torch()
    try:
        test_correctness()
    except pytest.skip.Exception as e:
        print(f"skipped: {e}")
