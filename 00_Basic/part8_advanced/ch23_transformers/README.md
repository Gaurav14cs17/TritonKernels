# Chapter 23 — Transformer Kernels

Part 8: Advanced Triton operations and production.

> **Goal.** Read a transformer block as shape-preserving stages around a
> residual stream, build each stage in PyTorch and in Triton, and recognize the
> hand-computed attention row inside every kernel.

Notebook: [`23_complete.ipynb`](23_complete.ipynb) · Helpers:
[`../../ttools.py`](../../ttools.py) · Apparatus:
[Setup](../../docs/SETUP.md) · [Notation](../../docs/NOTATION.md) ·
[Glossary](../../docs/GLOSSARY.md) · [API index](../../docs/API_INDEX.md) ·
[Formula index](../../docs/FORMULA_INDEX.md) · [Diagrams](../../docs/DIAGRAMS.md) ·
[Troubleshooting](../../docs/TROUBLESHOOTING.md) ·
[Exercise hints](../../docs/SOLUTIONS.md)

[Index](../../00_index.ipynb) · [← Previous](../ch22_dot_gemm/22_complete.ipynb) ·
[Next →](../ch24_production/24_complete.ipynb)

**Estimated time:** 150–180 min · **Difficulty:** Advanced

## Learning objectives

- State the **attention shape rule**: $Q,K,V\in\mathbb R^{T\times d}$,
  $S=QK^\top/\sqrt d\in\mathbb R^{T\times T}$, $P=\operatorname{softmax}(S)$,
  and $O=PV\in\mathbb R^{T\times d}$.
- Implement an embedding gather, position encoding, RoPE, and a QKV projection.
- Implement stable row softmax, LayerNorm, RMSNorm, and a GELU MLP in Triton.
- Implement a two-tile FlashAttention kernel with state $(m,\ell,\mathrm{acc})$.
- Explain the KV-cache shapes and why token decoding is memory-bound.
- Fuse stages and verify every value against PyTorch with silent assertions.

**Connections**
- **From Chapter 11.** The embedding is the token-id gather; this chapter adds the rest of the block.
- **Within the chapter.** Embedding → position → RoPE → QKV → scores → softmax → output → FlashAttention → LayerNorm/RMSNorm → MLP → fused block → KV cache.
- **To Chapter 24.** These kernels become `autograd.Function` modules, gradient checks, and profiling targets.
- **Recurring idea.** Every stage preserves the residual shape; the only square object is the $T\times T$ score matrix.

## Study ladder

1. **Method 01 — stage-by-stage PyTorch.** One function per formula; CPU
   correctness oracle; intentionally too slow for production.
2. **Method 02 — one Triton kernel per stage.** Embedding gather, LayerNorm and
   RMSNorm, row softmax, attention scores, attention output, KV append. Every
   kernel is masked and every test is a silent `assert`.
3. **Method 03 — online softmax and fusion.** A two-tile FlashAttention kernel,
   a fused GELU MLP, and a full pre-norm block checked against the reference.

**Study tip.**
- Take the two-token, $d=2$ worked example: $q_0=(1,1)$, $k_0=(1,1)$, $k_1=(1,0)$.
- Compute $s_0=(1.4142,\,0.7071)$, $p_0=(0.6698,\,0.3302)$, $o_0=(1,\,1.3395)$.
- Write those four numbers on paper; the rest is bookkeeping.

## Mathematical definition

- **Notation.** $T$ tokens, width $d$, batch $B$, $H$ heads of width $d_h=d/H$.
- **Shape rule.** $Q,K,V\in\mathbb R^{T\times d}$, then
  $S=QK^\top/\sqrt d\in\mathbb R^{T\times T}$,
  $P=\operatorname{softmax}(S)\in\mathbb R^{T\times T}$, and
  $O=PV\in\mathbb R^{T\times d}$.
- **Batch/heads.** Prepend $[B,H]$ to every shape.
- **Square object.** The score matrix is square in the sequence length, not the
  model width; hence $O(T^2d)$ time and the need for FlashAttention.
- **Cost.** $4T^2d$ FLOPs per head for the two products; the direct path moves
  about $4T^2b$ bytes, while FlashAttention moves about $4Tdb$.

## Worked example (the chapter spine)

- **Spine.** A two-token, $d=2$ sequence; every value is checked by the notebook.

### Embedding and QKV

$$
E=\begin{bmatrix}1&0\\0&1\\1&1\\2&-1\end{bmatrix},\quad
\pi=(2,0),\quad
X=E[\pi]=\begin{bmatrix}1&1\\1&0\end{bmatrix},
$$

$$
W_q=W_k=I,\quad
W_v=\begin{bmatrix}1&0\\0&2\end{bmatrix},\quad
Q=\begin{bmatrix}1&1\\1&0\end{bmatrix},\quad
K=\begin{bmatrix}1&1\\1&0\end{bmatrix},\quad
V=\begin{bmatrix}1&2\\1&0\end{bmatrix}.
$$

### Scores, softmax, output

$$
\begin{aligned}
S&=\frac{QK^\top}{\sqrt2}
=\begin{bmatrix}1.4142&0.7071\\0.7071&0.7071\end{bmatrix},\\
P&=\operatorname{softmax}(S)
=\begin{bmatrix}0.6698&0.3302\\0.5&0.5\end{bmatrix},\\
O&=PV
=\begin{bmatrix}1.0000&1.3395\\1.0000&1.0000\end{bmatrix}.
\end{aligned}
$$

**Hand checks.** $S_{00}=\sqrt2=1.4142$;
$e^{1.4142}/(e^{1.4142}+e^{0.7071})=0.6698$; each row of $P$ sums to one;
$O_{01}=0.6698\cdot2=1.3395$.

### RoPE and KV cache

- **One rotation.** $x=(1,0)$, $p=1$, $\theta_0=1$:

$$
\begin{bmatrix}x'_0\\x'_1\end{bmatrix}
=\begin{bmatrix}\cos1&-\sin1\\ \sin1&\cos1\end{bmatrix}
\begin{bmatrix}1\\0\end{bmatrix}
=\begin{bmatrix}0.5403\\0.8415\end{bmatrix}.
$$

- **KV shapes.** For $B=2$, $H=4$, $T=3$, $d=2$:

| object | shape | elements | FP16 bytes |
|---|---|---:|---:|
| $K$ cache before | $(2,4,3,2)$ | $48$ | $96$ |
| new $k$ at $t=3$ | $(2,4,1,2)$ | $16$ | $32$ |
| $K$ cache after | $(2,4,4,2)$ | $64$ | $128$ |
| scores against cache | $(2,4,1,4)$ | $32$ | $64$ |

- **Arithmetic.** $B\cdot H\cdot T\cdot d=48$ elements and $48\cdot2=96$ bytes
  in FP16; after the append $T=4$ and the cache is $64$ elements, $128$ bytes.

## Rules

1. **Shape rule.** $Q$ and $K$ agree on $d$; $P$ and $V$ agree on $T_k$. The
   score matrix is $[T_q,T_k]$, the output is $[T_q,d_v]$.
2. **Scale rule.** Apply $1/\sqrt d$ to the scores **before** the softmax; it is
   a temperature, not post-processing.
3. **Mask rule.** Put $-\infty$ on masked logits before the row maximum. A mask
   after the softmax leaves the remaining probabilities unnormalized.
4. **Softmax axis.** Attention normalizes over the **key** axis, so every query
   row sums to one.
5. **Residual rule.** Every sublayer maps $[B,T,d]\to[B,T,d]$, so the residual
   add is always legal; a shape change means a bad reshape.
6. **Cache rule.** A contiguous $[B,H,T,d]$ cache uses
   `((b*H+h)*T+t)*d+c`. A paged or transposed cache must pass its stride or page
   table; never assume contiguity.
7. **Precision rule.** Upcast logits to FP32 before `exp`; FP16 `exp`
   saturates, and a saturated softmax looks like a correct one-hot row.

## Chapter map

| § | Stage | Formula | Shape rule | Kernel/demo |
|---|---|---|---|---|
| **23.1** | Embedding | $Y_i=E_{\pi(i),:}$ | $(\cdots)\to(\cdots,d)$ | gather kernel |
| **23.2** | Position encoding | $PE_{t,2i}=\sin(t/10000^{2i/d})$ | $[T,d]\to[T,d]$ | `position_ref` + formula |
| **23.3** | RoPE | $x'=R(p)x$ | $[T,d]\to[T,d]$ | `rope_ref` + formula |
| **23.4** | QKV projection | $(Q,K,V)=X(W_q,W_k,W_v)$ | $[T,d]\to[T,3d]$ | `qkv_ref` projection |
| **23.5** | Attention scores | $S=QK^\top/\sqrt d$ | $[T_q,d]\times[T_k,d]\to[T_q,T_k]$ | score kernel |
| **23.6** | Softmax | $P_{ij}=e^{S_{ij}-m_i}/\sum_j e^{S_{ij}-m_i}$ | $[T_q,T_k]\to[T_q,T_k]$ | row-softmax kernel |
| **23.7** | Attention output | $O=PV$ | $[T_q,T_k]\times[T_k,d]\to[T_q,d]$ | output kernel |
| **23.8** | FlashAttention | online $(m,\ell,\mathrm{acc})$ | state $[T_q]$ and $[T_q,d]$ | two-tile flash kernel |
| **23.9** | LayerNorm | $y=\gamma(x-\mu)/\sqrt{\sigma^2+\epsilon}+\beta$ | $[T,d]\to[T,d]$ | norm kernel |
| **23.10** | RMSNorm | $y=\gamma x/\sqrt{\operatorname{mean}(x^2)+\epsilon}$ | $[T,d]\to[T,d]$ | norm kernel (`RMS=True`) |
| **23.11** | MLP | $Y=\operatorname{GELU}(XW_1+b_1)W_2+b_2$ | $[T,d]\to[T,4d]\to[T,d]$ | fused MLP kernel |
| **23.12** | Fused blocks | $Y=X+\operatorname{MLP}(\operatorname{LN}(X+\operatorname{Attn}(\operatorname{LN}(X))))$ | $[T,d]\to[T,d]$ | `block_ref` composition |
| **23.13** | KV cache | $K^{\text{cache}}_{b,h,t,:}\leftarrow k$ | $[B,H,T,d]\to[B,H,T+1,d]$ | append kernel |

- **Square object.** The score matrix is $T_q\times T_k$, not $d\times d$.
- **Everything else.** A row-wise map or a projection back to width $d$.
- **Silent checks.** Code cells verify with `assert`; prints are one-line
  summaries plus the final dispatch.
- **GPU gating.** Triton tests use `run_if_gpu`; CPU readers see `SKIP ...`.
- **Dot sizes.** Dot-based tests use $T=32$, $D=16$ because `tl.dot` wants tiles
  of at least $16$.

## How to run

```bash
cd part8_advanced/ch23_transformers
jupyter notebook 23_complete.ipynb
```

The notebook imports `ttools.py` by walking up to `00_Basic`. To run the Triton
tests on CPU, set `TRITON_INTERPRET=1` and the study override
`ttools.HAS_GPU = True` before the cells execute; otherwise they print `SKIP`.

## Study checklist

- [ ] State the attention shape rule and write the six shapes for $T=4$, $d=3$.
- [ ] Reproduce the worked example by hand and check the first code cell.
- [ ] Run those kernels on a GPU, or on CPU with the interpreter override.
- [ ] Trace the two-tile online-softmax update and explain both rescaling
      factors.
- [ ] Break one mask, one scale, and one stride; explain the assertion failure.
- [ ] Compute the KV-cache size and read/write ratio for a model you know.

## Exercises

1. **Embedding.** Extend the gather kernel to a 3-D id tensor $(B,T)$.
2. **Position.** Implement both interleaved and half-split sinusoidal layouts;
   show where the channels move.
3. **RoPE.** Add a `PAIR_LAYOUT` flag for adjacent vs half-split pairs.
4. **QKV.** Write a fused QKV GEMM with $N=3d$ and three output slices.
5. **Softmax.** Replace the row softmax with a two-pass online version and test
   a row with a large logit.
6. **FlashAttention.** Test a length that is not a multiple of $B_N$; identify
   the fully masked tile and its effect on $m$ and $\ell$.
7. **LayerNorm.** Implement Welford's one-pass variance and compare it with the
   two-pass version on a row with a large mean.
8. **MLP.** Fuse the bias adds and GELU into the GEMM epilogues; count the HBM
   round trips saved.

The kernels are teaching kernels: one program per row for normalizations and
softmax, one tile per program for the dot-based stages. They are masked and
correct but not autotuned. Production FlashAttention adds head batching, GQA,
paged caches, and a tuned tile schedule. Numerics use `float32`; KV byte counts
use FP16.
