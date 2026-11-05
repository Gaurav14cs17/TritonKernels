# Chapter 14 — Basic Reduction Operations
Part 6: Reductions, scans, and combined recipes.

> **Goal.** A reduction replaces an axis with one number per slice. The chapter
> asks one question twelve ways: *which axis disappears, what is the neutral
> element, and what shape comes out?*

**Thread — one idea.** A reduction is one combine operator over one axis: the axis disappears, every other axis keeps its size, and masked lanes must carry the operator's neutral element.

**Thread — the chain.** Scalar reductions (`sum`, `max`, `min`, `argmax`, `argmin`) → statistics (`mean`, `variance`, `std`) → axis and masked reductions → Method 01 reference → Method 02 kernels → Method 03 fusion.

**Connections**

- **From Chapter 13.** Coalesced access and tiling decide whether the combine walk is cheap; axis $0$ is strided and axis $1$ is contiguous.
- **Inside the chapter.** Every section reuses 14.1's shape rule and neutral-element pattern; 14.6–14.8 add division and a square root; 14.9–14.12 move the same walk onto matrix axes and masks.
- **To Chapter 15.** A scan keeps one output per prefix instead of collapsing the axis; the combine tree becomes a prefix tree.

Notebook: [`14_complete.ipynb`](14_complete.ipynb) · Helpers:
[`../../ttools.py`](../../ttools.py) · Apparatus:
[Setup](../../docs/SETUP.md) · [Notation](../../docs/NOTATION.md) ·
[Glossary](../../docs/GLOSSARY.md) · [API index](../../docs/API_INDEX.md) ·
[Formula index](../../docs/FORMULA_INDEX.md) · [Diagrams](../../docs/DIAGRAMS.md) ·
[Troubleshooting](../../docs/TROUBLESHOOTING.md) ·
[Exercise hints](../../docs/SOLUTIONS.md)

[Index](../../00_index.ipynb) · [← Previous](../../part5_layouts/ch13_memory_optimization/13_complete.ipynb) · [Next →](../ch15_scans/15_complete.ipynb)

**Estimated time:** 65–80 min · **Difficulty:** Intermediate

## Learning objectives
- Read a reduction as a combine tree and state its output shape.
- Use `tl.sum`, `tl.max`, `tl.min`, `tl.argmax`, and `tl.argmin` correctly.
- Compute mean, variance, and standard deviation in one fused pass.
- Reduce along axis 0, axis 1, several axes, and a padded axis.
- Build a masked reduction that divides by the valid count.
- Choose tie-breaking and NaN policies explicitly.

## Prerequisites
- Chapter 13: memory access patterns, coalescing, and tiling.
- Chapter 7: elementwise arithmetic and the meaning of `other=`.
- Basic statistics: mean, population variance, and standard deviation.

## Study ladder
1. **Method 01 — naive reference.** Two nested Python loops implement every reduction and are the correctness oracle.
2. **Method 02 — Triton kernels.** A 1-D operator-selector kernel covers sum, max, and min; a second covers argmax and argmin; a 2-D row kernel reduces axis 1 with a tail mask and neutral padding.
3. **Method 03 — fused and masked.** One load produces sum, sum of squares, mean, and variance per row; axis-0 and axis-1 variants are both written; the masked mean divides by the mask count.

**Study tip.** Fix $X=\begin{bmatrix}1&2&3\\4&5&6\end{bmatrix}$ and fill the two worked-example tables by hand before reading a kernel. If the shapes and numbers agree, the Triton code is only bookkeeping.

## Mathematical definition
**Connection.** Builds on Chapter 7's elementwise tiles: a reduction is the first operation whose output shape differs from its input shape.

- **Combine operator.** For $X\in\mathbb R^{M\times N}$, axis $1$ gives $Y_i=\bigoplus_{j=0}^{N-1}X_{ij}$ with shape $(M,)$, and axis $0$ gives $Z_j=\bigoplus_{i=0}^{M-1}X_{ij}$ with shape $(N,)$.
- **Neutral elements.** $x+0=x$, $x\cdot1=x$, $\max(x,-\infty)=x$, and $\min(x,+\infty)=x$; the operator, not the dtype, fixes the value.
- **Statistics.** $\mu=\frac1N\sum_j x_j$, $\sigma^2=\frac1N\sum_j (x_j-\mu)^2$, and $\sigma=\sqrt{\sigma^2}$.
- **One-pass warning.** $\sigma^2=\left(\frac1N\sum_j x_j^2\right)-\mu^2$ is algebraic but cancels when $\mu$ is large compared with $\sigma$; prefer two-pass or Welford.
- **Masked form.** $y=\sum_j m_jx_j$ and $\bar x=\frac{\sum_j m_jx_j}{\sum_j m_j}$ with $m_j\in\{0,1\}$.

## The rules this chapter teaches
### Axis rule
**Connection.** Builds on the concept diagrams and applies to every later section.

**Rule.** Axis $a$ is the axis the reduction walks over and the axis that disappears from the shape. Axis $1$ walks across each row; axis $0$ walks down each column. Write the output shape first; a wrong axis usually shows up as a shape mismatch or a plausible but wrong number on an asymmetric matrix.

### Shape rule
**Connection.** Builds on the axis rule; it is the formal statement of "which axis disappears."

**Rule.** If $x$ has shape $(d_0,\dots,d_{n-1})$ and the reduction is over axis $a$, then $\operatorname{shape}(y)_k=1$ for $k=a$ with keepdim, the axis is removed without keepdim, and $\operatorname{shape}(y)_k=d_k$ for $k\ne a$. A tuple of axes removes all of them in order; a full reduction returns $()$, or $(1,\dots,1)$ with keepdim.

### Keepdim rule
**Connection.** Builds on the shape rule and explains why `(M,)` and `(M,1)` are not interchangeable.

**Rule.** `X.sum(1)` has shape `(M,)`; `X.sum(1, keepdim=True)` has shape `(M,1)`. PyTorch names the flag `keepdim`, Triton `keep_dims`; the rule is the same. A row statistic used to normalize `(M,N)` must be `(M,1)`, not `(M,)`, or it broadcasts across the wrong dimension.

### Masked reduction rule
**Connection.** Builds on 14.6 and 14.10 and adds the mask count as the denominator.

**Rule.** The mask has the same shape as the guarded tile. The load uses the neutral element of the combine operator, and the denominator is the mask count:

```python
x = tl.load(x_ptr + offs, mask=m, other=0.0)
total = tl.sum(tl.where(m, x, 0.0), axis=1)
count = tl.sum(m.to(tl.float32), axis=1)
mean = tl.where(count > 0, total / tl.maximum(count, 1.0), 0.0)
```

**Failure.** An all-masked row returns `0.0`, not `0/0 = nan`; dividing by `BLOCK_N` scales every mean by `valid/BLOCK_N`.

### Tie and NaN rule
**Connection.** Builds on 14.4–14.5 and 14.12: indices and masks make tie and NaN policies part of the contract.

**Rule.** PyTorch and Triton break argmax/argmin ties at the first (leftmost) index by default. PyTorch's `max`/`min` propagate NaN; Triton's reduction default ignores NaN in current releases. Filter with `x == x` if NaN means "missing", and assert finiteness if NaN means "error".

## Chapter map
| § | Reduction | Formula | Output shape | Kernel/demo |
|---|---|---|---|---|
| **14.1** | `tl.sum()` | $y=\sum_j x_j$ | axis removed, or $1$ | 1-D sum; row sum |
| **14.2** | `tl.max()` | $y=\max_j x_j$ | same as sum | 1-D max; row max |
| **14.3** | `tl.min()` | $y=\min_j x_j$ | same as sum | 1-D min; row min |
| **14.4** | `tl.argmax()` | $y=\arg\max_j x_j$ | same as max, `int32` | 1-D argmax |
| **14.5** | `tl.argmin()` | $y=\arg\min_j x_j$ | same as min, `int32` | 1-D argmin |
| **14.6** | Mean | $\mu=\frac1N\sum_j x_j$ | same as sum | Fused moments |
| **14.7** | Variance | $\sigma^2=\frac1N\sum_j(x_j-\mu)^2$ | same as mean | Fused moments |
| **14.8** | Standard deviation | $\sigma=\sqrt{\sigma^2}$ | same as variance | Fused moments |
| **14.9** | Axis 0 | $Y_j=\sum_i X_{ij}$ | $(M,N)\to(N,)$ or $(1,N)$ | Column moments |
| **14.10** | Axis 1 | $Y_i=\sum_j X_{ij}$ | $(M,N)\to(M,)$ or $(M,1)$ | Row reduction |
| **14.11** | Multi-dimensional | $\bigoplus_{i_k}X_{\dots}$ | remove or keep as $1$ | Two-axis composition |
| **14.12** | Masked reduction | $y=\sum_j m_jx_j$ | reduced-axis shape | Masked mean |

## Worked numerical example
**Connection.** Every reduction from 14.1–14.12 appears once here; these tables are the hand-checkable version of the chapter map.

**Setup.** $X=\begin{bmatrix}1&2&3\\4&5&6\end{bmatrix}$.

### Axis 0 — collapse the rows
**Rule.** Each column is reduced over the two rows; output shape $(3,)$, keepdim $(1,3)$.

| Reduction | Result | Shape | Keepdim |
|---|---|---|---|
| sum | $(5,7,9)$ | $(3,)$ | $(1,3)$ |
| max | $(4,5,6)$ | $(3,)$ | $(1,3)$ |
| min | $(1,2,3)$ | $(3,)$ | $(1,3)$ |
| argmax | $(1,1,1)$ | $(3,)$ | $(1,3)$ |
| argmin | $(0,0,0)$ | $(3,)$ | $(1,3)$ |
| mean | $\left(\frac52,\frac72,\frac92\right)$ | $(3,)$ | $(1,3)$ |
| variance | $\left(\frac94,\frac94,\frac94\right)$ | $(3,)$ | $(1,3)$ |
| std | $\left(\frac32,\frac32,\frac32\right)$ | $(3,)$ | $(1,3)$ |

**Hand check.** Column 1: $2+5=7$, mean $3.5$, variance $\frac{(2-3.5)^2+(5-3.5)^2}{2}=2.25$, std $1.5$.

### Axis 1 — collapse the columns
**Rule.** Each row is reduced over the three columns; output shape $(2,)$, keepdim $(2,1)$.

| Reduction | Result | Shape | Keepdim |
|---|---|---|---|
| sum | $(6,15)$ | $(2,)$ | $(2,1)$ |
| max | $(3,6)$ | $(2,)$ | $(2,1)$ |
| min | $(1,4)$ | $(2,)$ | $(2,1)$ |
| argmax | $(2,2)$ | $(2,)$ | $(2,1)$ |
| argmin | $(0,0)$ | $(2,)$ | $(2,1)$ |
| mean | $(2,5)$ | $(2,)$ | $(2,1)$ |
| variance | $\left(\frac23,\frac23\right)$ | $(2,)$ | $(2,1)$ |
| std | $\left(\sqrt{\frac23},\sqrt{\frac23}\right)$ | $(2,)$ | $(2,1)$ |

**Hand check.** Row 0: $1+2+3=6$, mean $2$, centered $(-1,0,1)$, variance $\frac{1+0+1}{3}=\frac23$, std $\sqrt{2/3}\approx0.816497$.

### A $2\times2\times3$ reduction to $2\times3$
**Setup.** $A_0=\begin{bmatrix}0&1&2\\3&4&5\end{bmatrix}$ and $A_1=\begin{bmatrix}6&7&8\\9&10&11\end{bmatrix}$ form $A\in\mathbb R^{2\times2\times3}$.

**Rule.** Reducing axis $1$ leaves shape $(2,3)$:

$$ \sum_{i_1=0}^{1}A_{i_0,i_1,i_2} = \begin{bmatrix}3&5&7\\15&17&19\end{bmatrix}, \qquad \max_{i_1}A_{i_0,i_1,i_2} = \begin{bmatrix}3&4&5\\9&10&11\end{bmatrix}. $$

**Rule.** Reducing the last axis gives shape $(2,2)$: $\sum_{i_2=0}^{2}A_{i_0,i_1,i_2}=\begin{bmatrix}3&12\\21&30\end{bmatrix}$.

**Verification.** The notebook asserts every number and shape in these tables. In `float32`, it also asserts the cancellation warning: for $(100001,100002,100003,100004)$ the two-pass variance is $1.25$ while the one-pass form is over $1000$.

## Pitfalls
| Pitfall | What happens | Fix |
|---|---|---|
| Axis confusion | Row and column statistics are swapped | Write the output shape first |
| Forgetting keepdim | `(M,)` broadcasts across the wrong dimension | Use `(M,1)` |
| Wrong masked `other` | Padded lanes win a max or inflate a sum | `0`, `1`, `-inf`, `+inf` |
| NaN policy | PyTorch propagates, Triton may ignore | Filter with `x == x` |
| Variance cancellation | `E[x^2]-mu^2` loses digits for large `mu` | Two-pass or Welford |
| Integer overflow | `int32` sum wraps past $2^{31}-1$ | Use `int64` or `float32` |
| Wrong reduction axis | A plausible but wrong number | Test an asymmetric matrix |
| Dividing by `BLOCK_N` | Masked mean is scaled by `valid/BLOCK_N` | Divide by the count |

## How to run
**Connection.** Runs the same notebook that produced the tables above; the CPU path checks the reference and policy cells, and the interpreter path checks the kernels.

```bash
cd part6_reductions/ch14_reductions
jupyter notebook 14_complete.ipynb
```

- **CPU.** Every numerical cell, the Method 01 reference, the worked-example assertions, the cancellation check, and the policy checks run; Triton tests print `SKIP <name>: ...`.
- **Interpreter.** From `00_Basic`: `TRITON_INTERPRET=1 PYTHONPATH=$PWD python tools/interpreter_smoke.py`.
- **Paths.** The notebook imports `ttools.py` by walking up to `00_Basic`, so it works from the chapter folder, from `00_Basic`, or from the repository root.

## Study checklist
- [ ] Draw the sum tree and the max tree for $(1,2,3,4)$.
- [ ] Fill both $2\times3$ tables in the worked example by hand.
- [ ] State the output shape of `(2,3)` along axis 0 and axis 1, with and without keepdim.
- [ ] Explain why a masked max needs `other=-inf`, not `other=0`.
- [ ] Derive the two-pass variance and show where the one-pass form cancels.
- [ ] Explain why the masked mean divides by the count, not by `BLOCK_N`.
- [ ] Break the tail mask in the row kernel and identify which store fails.
- [ ] State the tie rule and the NaN policy that the notebook tests.

**Where this leads.** The exercises stress the boundaries of these rules; Chapter 15 replaces the collapsing reduction with a prefix scan, keeping one output per element.

## Exercises
1. **Reduction tree.** Draw the sum tree for $N=8$ and count its additions.
2. **Shape rule.** For a $(4,5,6)$ tensor, list every output shape for axis $0$, $1$, $2$, axes $(0,1)$, and all axes, with and without keepdim.
3. **Axis swap.** Reduce $X$ along axis 0 and axis 1, transpose it, and repeat.
4. **Masked max.** Convert the masked mean into a masked max. Which `other` value is correct, and what should an all-masked row return?
5. **Count denominator.** Divide by `BLOCK_N` instead of the count for a row of length $3$ in a block of width $8$. What is the scale factor?
6. **Variance cancellation.** Compare the one-pass and two-pass errors for $x=10^5+(1,2,3,4)$ in `float32`.
7. **Sample versus population.** Recompute variance with $N-1$ and state which convention LayerNorm uses.
8. **Argmax tie and NaN.** Build a row with two maxima; check the first-index rule. Then feed `[1.0, nan, 3.0]` to `torch.max` and a NaN-filtered `tl.max`.
9. **Fusion traffic.** Compare three separate statistic launches with the fused moments kernel in bytes read.

Hints are in [`../../docs/SOLUTIONS.md`](../../docs/SOLUTIONS.md).
