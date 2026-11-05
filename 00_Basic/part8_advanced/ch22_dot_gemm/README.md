# Chapter 22 — Advanced Matrix Operations

> **Goal.** Read a GEMM as one contraction along $K$, cut into output tiles and
> K panels. `tl.dot` does the small products, FP32 keeps the accumulator honest,
> tensor cores supply throughput, and the epilogue casts, biases, and activates.

Notebook: [`22_complete.ipynb`](22_complete.ipynb) · Helpers:
[`../../ttools.py`](../../ttools.py) · Apparatus: [Setup](../../docs/SETUP.md) ·
[Notation](../../docs/NOTATION.md) · [Glossary](../../docs/GLOSSARY.md) ·
[API index](../../docs/API_INDEX.md) · [Formula index](../../docs/FORMULA_INDEX.md) ·
[Diagrams](../../docs/DIAGRAMS.md) · [Troubleshooting](../../docs/TROUBLESHOOTING.md) ·
[Exercise hints](../../docs/SOLUTIONS.md)

[Index](../../00_index.ipynb) · [← Previous](../ch21_atomics/21_complete.ipynb) · [Next →](../ch23_transformers/23_complete.ipynb)

**Estimated time:** 90–120 min · **Difficulty:** Advanced

## Thread

- **Recurring idea.** A GEMM is one contraction along $K$; every kernel is a scheduling choice around that sum.
- **Chain.** one tile (`tl.dot`) → the global $(M,K)\cdot(K,N)$ rule → a tiled grid → an FP32 accumulator over K panels → tensor cores → mixed precision → the epilogue → optimization.
- **Invariant.** inner dimensions match, the accumulator stays FP32, and the store is the only cast.

## Connections

- **Builds on.** Chapter 18 (2-D grids), Chapter 19 (tiling), Chapter 20 (fusion), and the scalar dot product.
- **Internal chain.** 22.1 one tile → 22.2 global rule → 22.3 grid → 22.4 K-panel accumulator → 22.5 tensor cores → 22.6-22.8 dtypes → 22.9 epilogue → 22.10 optimization.
- **Leads to.** Chapter 23 builds attention and transformer blocks from these GEMMs; Chapter 24 adds autotuning, benchmarking, and roofline analysis.

## Learning objectives

- State $C_{ij}=\sum_k A_{ik}B_{kj}$ and check inner dimensions before launching.
- Read a tiled GEMM as a grid of $(B_M,B_N)$ tiles over $\lceil K/B_K\rceil$ panels.
- Use `tl.dot` with an FP32 accumulator and masks for every non-multiple size.
- Separate FP16/BF16/TF32 inputs from the FP32 accumulator; cast at the store.
- Fuse bias and activation, and estimate FLOPs, bytes, and arithmetic intensity.

## Study ladder

1. **Method 01 — naive triple loop.** One iteration per output element and per
   term; the correctness oracle and intentionally too slow.
2. **Method 02 — tiled Triton GEMM.** One program per output tile, a K loop of
   `tl.dot` panels, FP32 accumulation, masks on rows, columns, and `rk`.
3. **Method 03 — epilogue and mixed precision.** Fused bias/activation, FP16 or
   BF16 inputs with an FP32 accumulator, and a TF32 variant for FP32.

**Study tip.** With $A=\begin{bmatrix}1&2&3\\4&5&6\end{bmatrix}$ and
$B=\begin{bmatrix}1&2\\3&4\\5&6\end{bmatrix}$, compute one entry by hand:
$C_{0,1}=1\cdot2+2\cdot4+3\cdot6=28$. The full product is
$C=\begin{bmatrix}22&28\\49&64\end{bmatrix}$; the notebook verifies it, the four
dot products, and the 24 FLOPs.

## Mathematical definition

- **Contraction.** $C_{ij}=\sum_{k=0}^{K-1}A_{ik}B_{kj}$ for $A\in\mathbb{R}^{M\times K}$ and $B\in\mathbb{R}^{K\times N}$, so $C\in\mathbb{R}^{M\times N}$; $K$ is the only shared axis.
- **Tensor-core tile.** $D_{\text{MMA}}=A_{\text{MMA}}B_{\text{MMA}}+C_{\text{MMA}}$ on small tiles such as $16\times16\times16$.
- **Tiling.** $G_M=\lceil M/B_M\rceil$, $G_N=\lceil N/B_N\rceil$, grid $=G_MG_N$, with $p_m=\lfloor\text{pid}/G_N\rfloor$ and $p_n=\text{pid}\bmod G_N$.
- **Accumulation.** $\mathrm{acc}_{t+1}=\mathrm{acc}_t+A_tB_t$ with $\mathrm{acc}_0=0$; after $T=\lceil K/B_K\rceil$ panels, $\mathrm{acc}_T=C$.

## Rules

### Contraction
- $(M,K)\cdot(K,N)\to(M,N)$; the inner sizes must agree exactly.
- Assert `a.shape[1] == b.shape[0]` on the host, then encode row-major offsets explicitly.

### Tiling
- One program per output tile and one store.
- `BLOCK_M`/`BLOCK_N` control reuse and register pressure; `BLOCK_K` controls panel depth and shared memory.
- Use `tl.cdiv` and mask every axis that can be partial.

### Dtype and accumulation
- Inputs may be FP16, BF16, or TF32; the accumulator is FP32 and the cast happens at the store.
- FP16 has 10 mantissa bits and max $65504$; BF16 has 7 mantissa bits with the FP32 exponent range; TF32 has 10 mantissa bits with the FP32 range.

### Epilogue
- $Y_{ij}=\operatorname{act}(C_{ij}+b_j)$ with $b\in\mathbb{R}^{B_N}$ broadcast as `bv[None, :]`.
- A bias of length $B_M$ added as `bv[:, None]` is shape-correct but wrong.

## Worked example

### The 2×3 product
- **Rule.** $K=3$, so each entry is a sum of three products: $C_{ij}=A_{i0}B_{0j}+A_{i1}B_{1j}+A_{i2}B_{2j}$.
- **Example.** $A=\begin{bmatrix}1&2&3\\4&5&6\end{bmatrix}$ and $B=\begin{bmatrix}1&2\\3&4\\5&6\end{bmatrix}$.
- **Entries.** $C_{0,0}=1+6+15=22$, $C_{0,1}=2+8+18=28$, $C_{1,0}=4+15+30=49$, $C_{1,1}=8+20+36=64$.

$$
C=\begin{bmatrix}22&28\\49&64\end{bmatrix},
\qquad
\mathrm{FLOPs}=24,
\qquad
\text{FP32 bytes}=64.
$$

### The 4×4 tiled product, $B_M=B_N=B_K=2$
- **Rule.** Four output tiles and two K panels per tile.
- **Example.** $A=\begin{bmatrix}1&2&3&4\\5&6&7&8\\9&10&11&12\\13&14&15&16\end{bmatrix}$ and $B=\begin{bmatrix}1&0&1&0\\0&1&0&1\\1&0&0&1\\0&1&1&0\end{bmatrix}$.
- **Tile map.** Each `pid` owns one tile:

| `pid` | $(p_m,p_n)$ | rows | cols | output |
|---:|---|---|---|---|
| 0 | (0,0) | 0:2 | 0:2 | $C_{00}$ |
| 1 | (0,1) | 0:2 | 2:4 | $C_{01}$ |
| 2 | (1,0) | 2:4 | 0:2 | $C_{10}$ |
| 3 | (1,1) | 2:4 | 2:4 | $C_{11}$ |

- **Panel check.** For tile $(0,0)$, both panels multiply by $I$, so $\begin{bmatrix}1&2\\5&6\end{bmatrix}I+\begin{bmatrix}3&4\\7&8\end{bmatrix}I=\begin{bmatrix}4&6\\12&14\end{bmatrix}$.
- **Tiles.** $C_{00}=\begin{bmatrix}4&6\\12&14\end{bmatrix}$, $C_{01}=\begin{bmatrix}5&5\\13&13\end{bmatrix}$, $C_{10}=\begin{bmatrix}20&22\\28&30\end{bmatrix}$, $C_{11}=\begin{bmatrix}21&21\\29&29\end{bmatrix}$.

$$
C=\begin{bmatrix}
4&6&5&5\\12&14&13&13\\20&22&21&21\\28&30&29&29
\end{bmatrix}.
$$

## Chapter map

| § | Topic | Formula | Shape/dtype rule | Kernel/demo |
|---|---|---|---|---|
| **22.1** | `tl.dot()` | $D=\operatorname{dot}(a,b)+c$ | $(B_M,B_K)\cdot(B_K,B_N)\to(B_M,B_N)$ | 2×2 contraction |
| **22.2** | Matrix multiplication | $C_{ij}=\sum_k A_{ik}B_{kj}$ | $(M,K)\cdot(K,N)\to(M,N)$ | worked 2×3 |
| **22.3** | Tiled GEMM | $\text{pid}\mapsto(p_m,p_n)$ | grid $G_MG_N$ | tile map |
| **22.4** | Accumulation | $\mathrm{acc}_{t+1}=\mathrm{acc}_t+A_tB_t$ | FP32 $(B_M,B_N)$ | K panels |
| **22.5** | Tensor cores | $D=AB+C$ on MMA tiles | multiple of 16 per axis | MMA count |
| **22.6** | Mixed precision | $C=\mathrm{FP32}(\sum_k A_{ik}B_{kj})$ | low precision in, FP32 acc | mixed kernel |
| **22.7** | FP16 and BF16 | 10 vs 7 mantissa bits | narrower storage and inputs | rounding cell |
| **22.8** | TF32 | 10 mantissa bits, FP32 exponent | FP32 in, `"tf32"` | tf32 kernel |
| **22.9** | GEMM epilogues | $Y=\operatorname{act}(AB+b^\top)$ | bias $(B_N,)$ over rows | bias + ReLU |
| **22.10** | GEMM optimization | $\mathrm{FLOPs}=2MNK$ | tile/panel choices change reuse | cost model |

All correctness checks are assertions; the only prints are one-line summaries.
The final cell dispatches the CPU and GPU test lists.

## Performance model

- **Rule.** Arithmetic is fixed by the contraction; traffic and reuse depend on the schedule.

$$
\mathrm{FLOPs}=2MNK,\qquad
Q_{\min}=b\,(MK+KN+MN),\qquad
\mathrm{AI}_{\min}=\frac{2MNK}{Q_{\min}},\qquad
Q_{\text{tile}}=b\,MNK\left(\frac{1}{B_M}+\frac{1}{B_N}\right).
$$

- **Ideal.** For $M=N=K=1024$ and FP32: $2.147\times10^9$ FLOPs, $Q_{\min}=12.58$ MB, and $\mathrm{AI}_{\min}=170.7$ FLOP/byte.
- **Tiled.** At $B_M=B_N=64$: $Q_{\text{tile}}=134.2$ MB and $\mathrm{AI}=16.0$; at $128$: $67.1$ MB and $\mathrm{AI}=32.0$.
- **Shared memory.** One FP32 stage for $B_M=B_N=64$, $B_K=32$ is $4(B_M+B_N)B_K=16$ KiB.
- **Where.** The ideal assumes perfect cache reuse; L2 closes part of the gap between the ideal and the tile model.

## Pitfalls

| Pitfall | What happens | Fix |
|---|---|---|
| $K$ mismatch | wrong contraction or a runtime error | assert `a.shape[1] == b.shape[0]` |
| Tile shape not supported | compile error or slow fallback | use multiples of 16 per axis |
| FP16 accumulator | large error or `inf` on long sums | `tl.zeros(..., tl.float32)` |
| Missing masks | out-of-bounds read/write | mask rows, cols, and `rk` |
| Transposed `B` | shape-correct but wrong values | pass `(K,N)` or explicit strides |
| Epilogue broadcast | bias added down columns | add `bv[None, :]` with `cols` |
| TF32 precision | third-digit mismatch | use `"ieee"` or tolerance $10^{-2}$ |
| Cast before epilogue | bias/activation in FP16 | keep `acc` in FP32 until the store |

## How to run

```bash
cd part8_advanced/ch22_dot_gemm
jupyter notebook 22_complete.ipynb
# interpreter smoke test, from 00_Basic:
cd 00_Basic
TRITON_INTERPRET=1 PYTHONPATH=$PWD python tools/interpreter_smoke.py
```

On CPU the worked examples, the naive reference, the dtype checks, and the cost
model run; GPU tests print `SKIP <name>: ...`. The interpreter exercises the
FP16 path and the tiny FP32 cases; BF16 `tl.dot` and TF32 are guarded to CUDA.

## Study checklist

- [ ] State the contraction rule and check the 2×3 inner dimensions by hand.
- [ ] Write the 4×4 tile map, including the two K panels per tile.
- [ ] Remove one mask from the tiled kernel and explain the failure.
- [ ] Change the accumulator to FP16 and describe the error growth.
- [ ] Add a bias on the wrong axis and build a value-wrong case.
- [ ] Compute AI for $1024^3$ under the ideal and $64\times64$ models.
- [ ] Explain why the epilogue cast happens at the store.

## Exercises

1. **Contraction by hand.** For $A=\begin{bmatrix}2&1&0\\1&3&2\end{bmatrix}$ and
   $B=\begin{bmatrix}1&4\\2&0\\1&1\end{bmatrix}$, compute $AB$ and its FLOP count.
2. **Tile map.** For $M=70$, $N=45$, $B_M=32$, $B_N=16$, list the grid size and
   the two partial tiles.
3. **K panels.** Rewrite the 2×3 example with $B_K=1$ and show the three updates.
4. **Mask audit.** Remove `rk < K` and explain the out-of-bounds read for
   $K=20$, $B_K=16$.
5. **Accumulator dtype.** Use FP16 for `acc` on a $128^3$ product; measure the
   error and find where it becomes unusable.
6. **Bias axis.** Add `bv[:, None]` with a bias of length $M$ and build a small
   shape-correct, value-wrong case.
7. **TF32 tolerance.** Compare `"ieee"` and `"tf32"` on a $64^3$ FP32 product;
   report maximum absolute and relative errors.
8. **Traffic.** For $4096^3$ FP32, compute $Q_{\min}$ and tile traffic at
   $B_M=B_N=64$ and $128$.
9. **Epilogue fusion.** Add a tanh-approximation GELU and compare it with
   `torch.nn.functional.gelu`; choose a tolerance.
10. **Autotune.** Sweep tile sizes, `num_warps`, and `num_stages` on a GPU while
    keeping the correctness assertion in the loop.

Hints are in [`../../docs/SOLUTIONS.md`](../../docs/SOLUTIONS.md).
