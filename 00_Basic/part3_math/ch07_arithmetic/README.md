# Chapter 7 — Elementwise Arithmetic

Part 3: Small mathematical operations.

> **Thesis.** Read an elementwise kernel as one scalar function lifted to a tile:
> `tl.load` → `f` → `tl.store`. The twenty operations in this chapter are five
> families of that one idea. The arithmetic is cheap; the traffic is not.

Notebook: [`07_complete.ipynb`](07_complete.ipynb) · Shared helpers:
[`../../ttools.py`](../../ttools.py) · Book apparatus:
[Setup](../../docs/SETUP.md) · [Notation](../../docs/NOTATION.md) ·
[Glossary](../../docs/GLOSSARY.md) · [API index](../../docs/API_INDEX.md) ·
[Formula index](../../docs/FORMULA_INDEX.md) · [Diagrams](../../docs/DIAGRAMS.md) ·
[Troubleshooting](../../docs/TROUBLESHOOTING.md) ·
[Exercise hints](../../docs/SOLUTIONS.md)

[Index](../../00_index.ipynb) · [← Previous](../../part2_memory/ch06_indexing/06_complete.ipynb) · [Next →](../ch08_comparisons/08_complete.ipynb)

**Estimated time:** 75–90 min · **Difficulty:** Intermediate

## Learning objectives

- Read an elementwise kernel as a scalar function lifted to a tile.
- Write the tile map, mask, and pointer offsets for a partial last block.
- Choose the right operator for a family of problems, and state its domain.
- Explain why elementwise arithmetic is memory-bound.
- Select `other` from the operation that follows, not by habit.
- Derive the backward pass of the five families and check it with autograd.

## Prerequisites

- Chapter 6: pointers, offsets, `tl.load`, `tl.store`, and masking.
- Basic calculus: the chain rule and the power rule.
- No prior GPU arithmetic background is assumed.

## Connections

**Connections** — how this chapter sits in the book:

- **From Chapter 6.** Pointers, offsets, loads, stores, and masks are the machinery; this
  chapter supplies the scalar function.
- **Within the chapter.** §1–§5 fix the contract and the spine; §7.1–§7.20 walk the five
  families; §7.21 collects the canonical kernel; Part IV composes them; Part V differentiates
  them.
- **To Chapter 8.** The mask and `other=` conventions become predicates and `tl.where`.
- **To Chapter 9.** `exp`, `log`, `maximum`, and `clamp` become activation functions.
- **To Chapter 10.** Broadcasting size-1 axes becomes a fused epilogue; its gradient is a
  reduction.
- **To Chapter 14.** A reduction is the first operation that lets one lane read another.
- **To the rest of Part 3.** The same tile skeleton reappears in comparisons, activations,
  and broadcasting.

## Thread of the chapter

**Thread.** Every section is one link in the same chain:

- **Elementwise lift.** A scalar $f$ acts on lanes independently: $F(x)_i=f(x_i)$.
- **Tile skeleton.** `pid`, `offs`, `mask`, `load`, `store` are identical for all twenty
  operations.
- **Memory-bound.** Traffic, not FLOPs, sets the lower bound; fusion and narrow dtypes are
  the levers.
- **Five families.** Field, quotient/remainder, powers/roots, transcendentals,
  order/bounds each share a reason and a failure mode.
- **Backward.** Every local derivative is another elementwise kernel; a broadcast axis
  turns some gradients into reductions.

## Study ladder

1. **Method 01 — naive reference.** A Python loop applies one scalar function at a time.
   It is the correctness oracle for every kernel, and it is intentionally too slow for a
   GPU workload.
2. **Method 02 — one Triton kernel per operation.** Every kernel is the same seven-line
   skeleton with one expression changed. The tests use the chapter spine, so the partial
   last tile is exercised by every operation.
3. **Method 03 — fused and production-shaped.** A fused arithmetic chain loads once and
   stores once, then four applied kernels (affine/residual, RMSNorm, stable log-sum-exp,
   bounds) build real compositions. The capstone is a stable softmax.

**Study tip.** Fix one tiny shape for the whole chapter before reading any kernel: $N=6$,
`BLOCK=4`, so $P=\lceil6/4\rceil=2$.

- **Spine.** $x=(1,2,3,4,5,6)$ and $y=(6,5,4,3,2,1)$.
- **Tile map.** Program 1 owns indices $4,5,6,7$, but only $4,5$ are live.

| pid | offsets | mask | live lanes | $x$ tile (`other=0`) | $y$ tile |
|---|---|---|---|---|---|
| 0 | 0,1,2,3 | T,T,T,T | 0,1,2,3 | 1,2,3,4 | 6,5,4,3 |
| 1 | 4,5,6,7 | T,T,F,F | 4,5 | 5,6,0,0 | 2,1,0,0 |

- **Stride walk.** $\text{start}=1\cdot4=4$, so the lanes are $4,5,6,7$ and the mask keeps
  $4,5$.
- **Bridge.** Every kernel in the notebook reuses these literals.

## Mathematical definition

- **Rule.** Let $[N]=\{0,\dots,N-1\}$ and let $f:\mathbb R\to\mathbb R$. The elementwise
  lift is $F(x)_i=f(x_i)$ for $i\in[N]$; for a binary scalar function,
  $F(x,y)_i=f(x_i,y_i)$.
- **Independence.** Output lane $i$ depends only on input lane $i$. Any partition of $[N]$
  can be assigned to a different program, and the pieces can be concatenated.
- **Shape invariance.** An elementwise map cannot change the logical shape of its input,
  apart from broadcasting size-1 axes.
- **Mask.** A mask is a restriction to the indices that exist:
  $I_p=\{pB,\dots,pB+B-1\}\cap[N]$. Both the load and the store must respect it.
- **Failure.** An operation that reads another lane is not elementwise; it is a reduction
  (Chapter 14).

## Cost model

- **Rule.** For $N$ elements of $b$ bytes, a unary map moves $Q_{\text{unary}}=2Nb$ and a
  binary map moves $Q_{\text{binary}}=3Nb$.
- **Formula.** The arithmetic intensity is $\mathrm{AI}_{\text{unary}}=1/(2b)$ or
  $\mathrm{AI}_{\text{binary}}=1/(3b)$; the roofline is
  $t\ge\max(N/P_{\text{peak}},Q/\mathrm{BW})\approx Q/\mathrm{BW}$.
- **Example.** For `float32`, a binary map has $\mathrm{AI}\approx0.083$ FLOP/byte, while
  a modern accelerator has a machine balance of roughly $10$–$100$ FLOP/byte.
- **Where.** The lever is fewer bytes, not fewer instructions: fusion, narrower dtypes,
  and coalesced access.
- **Failure.** Counting FLOPs instead of bytes optimizes the wrong side of the roofline.

## The five families

| Family | Members | Shared reason |
|---|---|---|
| The field | $+,-,\times,/$ | affine maps and normalization |
| Quotient/remainder | $\lfloor x/y\rfloor,\ x\bmod y$ | integer index arithmetic |
| Powers and roots | $x^p,\lvert x\rvert,1/x,\sqrt{x},1/\sqrt{x}$ | norms, distances, variance |
| Transcendentals | $e^x,\ln x,2^x,\log_2 x$ | probability and information |
| Order and bounds | $\max,\min,\operatorname{clamp}$ | keeping a value inside its domain |

## Worked example (chapter spine)

On $x=(1,2,3,4,5,6)$ and $y=(6,5,4,3,2,1)$:

| operation | result |
|---|---|
| $x+y$ | $(7,7,7,7,7,7)$ |
| $x-y$ | $(-5,-3,-1,1,3,5)$ |
| $x\,y$ | $(6,10,12,12,10,6)$ |
| $x/y$ | $(0.1667,0.4,0.75,1.3333,2.5,6)$ |
| $x^y$ | $(1,32,81,64,25,6)$ |
| $\max(x,y)$ | $(6,5,4,4,5,6)$ |
| $\min(x,y)$ | $(1,2,3,3,2,1)$ |
| $\operatorname{clamp}(x,2,5)$ | $(2,2,3,4,5,5)$ |

- **Hand checks.** $z_1=2+5=7$; $2^5=32$; $\max(2,5)=5$; $\operatorname{clamp}(1,2,5)=2$.
- **Where.** `test_worked_example()` verifies the table exactly.
- **Compiler check.** `BLOCK=4` gives $P=2$ and a partial last tile. `BLOCK=3` also gives
  $P=2$, but the last mask changes. `BLOCK=2` gives $P=3$. The vector never moves; only
  the tile boundaries do.

## Chapter map

| § | Topic | Key formula | Applied kernel |
|---|---|---|---|
| **7.1** | Addition | $z_i=x_i+y_i$ | Affine / residual |
| **7.2** | Subtraction | $z_i=x_i-y_i$ | Affine / residual |
| **7.3** | Multiplication | $z_i=x_i\,y_i$ | Affine / residual |
| **7.4** | Division | $z_i=x_i/y_i$ | Affine / residual |
| **7.5** | Floor division | $z_i=\lfloor x_i/y_i\rfloor$ | — |
| **7.6** | Modulo | $z_i=x_i-y_i\lfloor x_i/y_i\rfloor$ | — |
| **7.7** | Power | $z_i=x_i^{\,y_i}$ | — |
| **7.8** | Negation | $z_i=-x_i$ | — |
| **7.9** | Absolute value | $z_i=\lvert x_i\rvert$ | — |
| **7.10** | Reciprocal | $z_i=1/x_i$ | — |
| **7.11** | Square | $z_i=x_i^2$ | RMS normalization |
| **7.12** | Square root | $z_i=\sqrt{x_i}$ | RMS normalization |
| **7.13** | Reciprocal square root | $z_i=1/\sqrt{x_i}$ | RMS normalization |
| **7.14** | Exponential | $z_i=e^{x_i}$ | Stable log-sum-exp |
| **7.15** | Natural logarithm | $z_i=\ln x_i$ | Stable log-sum-exp |
| **7.16** | Base-2 exponential | $z_i=2^{x_i}$ | — |
| **7.17** | Base-2 logarithm | $z_i=\log_2 x_i$ | — |
| **7.18** | Maximum | $z_i=\max(x_i,y_i)$ | Log-sum-exp · bounds |
| **7.19** | Minimum | $z_i=\min(x_i,y_i)$ | Bounds |
| **7.20** | Clamping | $z_i=\min(\max(x_i,l),u)$ | Bounds |
| **7.21** | The canonical elementwise kernel | one skeleton, `OP: tl.constexpr` | bridge to Part IV |
| **★** | Capstone | stable softmax | all five families |

Every operation row (7.1–7.20) has a hand-note, a worked example with the spine, a silent
numerical code cell with assertions, and a minimal Triton micro-kernel. Section 7.21
collects the shared skeleton in one canonical kernel.

## 7.21 The canonical elementwise kernel

- **Connection.** §7.1–§7.20 changed only the scalar expression; this section collects the
  invariant they share and hands it to Part IV.
- **Rule.** Every kernel in the chapter is the same seven-line skeleton: `pid`, `offs`,
  `mask`, masked load, scalar body, masked store.
- **Formula.** $\text{offs}=\text{pid}\cdot B+\{0,\dots,B-1\}$,
  $\text{mask}=\text{offs}\lt N$, and $P=\lceil N/B\rceil$.
- **Example.** One kernel with `OP: tl.constexpr` covers add, sub, mul, div, max, and min;
  the JIT emits only the selected arm, so the unused arms cost nothing at runtime.
- **Where.** This is the bridge to the kernel part: comparisons (Chapter 8), activations
  (Chapter 9), broadcasting (Chapter 10), and fused chains (Chapter 20) reuse the skeleton.
- **Failure.** A runtime `OP` would branch per lane and add divergence; keep the selector
  `tl.constexpr`. The denominator uses `other=1.0` because division is the one arm that is
  not safe at zero.
- **Bridge.** Part IV changes the grid and fuses several bodies, but the body remains this
  skeleton.

```python
@triton.jit
def canonical_elementwise_kernel(x_ptr, y_ptr, out_ptr, n_elements,
                                 OP: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    y = tl.load(y_ptr + offs, mask=mask, other=1.0)
    if OP == 0:
        z = x + y
    elif OP == 1:
        z = x - y
    elif OP == 2:
        z = x * y
    elif OP == 3:
        z = x / y
    elif OP == 4:
        z = tl.maximum(x, y)
    else:
        z = tl.minimum(x, y)
    tl.store(out_ptr + offs, z, mask=mask)
```

## Applied examples

| # | Example | Spine result | Sections |
|---|---|---|---|
| 1 | Affine / residual | $(1,3,5,7,9,11)$ with $a=2,b=-1$ | 7.1, 7.3 |
| 2 | RMS normalization | inverse RMS $=0.256776$ | 7.1, 7.3, 7.4, 7.11, 7.13 |
| 3 | Stable log-sum-exp | $6.456193$ | 7.1, 7.2, 7.14, 7.15, 7.18 |
| 4 | Bounds | ReLU, hardtanh, safe log, safe sqrt | 7.18, 7.19, 7.20 |
| **C** | **Capstone: softmax** | $(0.0043,0.0116,0.0315,0.0858,0.2331,0.6337)$ | 7.1, 7.2, 7.4, 7.14, 7.18 |

- **Rule.** The stable log-sum-exp subtracts the row maximum:
  $\operatorname{LSE}(x)=m+\log\sum_j e^{x_j-m}$ with $m=\max_j x_j$.
- **Example.** On the spine, $m=6$ and
  $\operatorname{LSE}(x)=6+\log(e^{-5}+\cdots+1)=6.456193$.
- **Where.** The capstone uses all five families: maximum (reduction), subtraction,
  exponential, sum, and division.
- **Failure.** Without the shift, a logit of 100 overflows before the division; with it,
  the largest exponent is $e^0=1$.

## Forward and backward

- **Rule.** Every operation has a local derivative; the backward pass multiplies the local
  derivatives along each path.
- **Formula.** For a binary forward, $\partial L/\partial x=g\,\partial z/\partial x$ and
  $\partial L/\partial y=g\,\partial z/\partial y$; a broadcast axis becomes a sum in the
  backward.
- **Example.** The notebook checks the row loss $L(x)=\sum_i\ln x_i$ with
  $\partial L/\partial x_i=1/x_i$. On the spine, $L=\ln720=6.579251$ and
  $\nabla L=(1,0.5,0.3333,0.25,0.2,0.1667)$.
- **Where.** The autograd cell checks the forward value, the gradient, and `1/x` with
  assertions; the maximum error is at the level of floating-point rounding.
- **Useful derivatives.** $\partial_x(x/y)=1/y$, $\partial_y(x/y)=-x/y^2$,
  $\partial_x\sqrt{x}=1/(2\sqrt{x})$, $\partial_x x^{-1/2}=-1/(2x^{3/2})$,
  $\partial_x e^x=e^x$, $\partial_x\ln x=1/x$, and
  $\partial_x\operatorname{clamp}(x,l,u)=\mathbf 1[l\lt x\lt u]$.
- **Failure.** A wrong gradient usually shows as a constant factor or a missing term; the
  notebook's derivative table is the checklist.

## Pitfalls

- **Mask both doors.** A masked store with an unmasked load still performs an
  out-of-bounds read.
- **Choose `other` from the next operation.** `0.0` is safe for addition and sums; a
  maximum wants `-inf`, a minimum wants `+inf`, and a logarithm must not be masked to
  zero.
- **Do not pass a `y_ptr` to a unary kernel.** Negation, square, sqrt, exp, and log have
  one input.
- **Respect the domain.** `log(0)=-inf`, `log(-1)=nan`, `sqrt(-1)=nan`, `rsqrt(0)=inf`,
  and a negative base with a fractional exponent is `nan`.
- **Mind the exponent range.** `exp` overflows above about 88.7 and `exp2` above about 128
  in `float32`; `float16` products overflow above 65504.
- **Use the base-2 pair for the hardware path.** `tl.exp2`/`tl.log2` are fast;
  `tl.exp`/`tl.log` are natural. Convert the constant, not the operation.
- **Do not confuse elementwise and reduction.** `tl.maximum(x,y)` compares two tiles;
  `tl.max(x, axis=...)` collapses an axis.
- **Integer signs.** Triton's `//` truncates toward zero and `%` follows the dividend;
  Python floors and follows the divisor.
- **NaN policy.** Triton's `maximum`/`minimum` default to `propagate_nan=NONE`;
  `torch.maximum` propagates NaN.
- **Power on tiles.** Triton 3.x has no tensor `**`. Use `tl.exp2(y * tl.log2(x))` for a
  positive base or `libdevice.pow(x, y)` for a general real power.

## Self-check

1. For the spine $N=6$, `BLOCK=4`, write the offsets, mask, and live lanes of program 1.
   Which two lanes are masked?
2. Compute $x+y$, $x\,y$, and $x^y$ for lanes 1 and 5 by hand.
3. Why is $2^5=32$ but `exp2(5*log2(-2))` not a real number?
4. What is the difference between `tl.maximum(x,y)` and `tl.max(x, axis=0)`?
5. For $x=-7$, $y=3$, what are $x\mathbin{//}y$ and $x\bmod y$ in Python, in C, and in
   Triton?
6. Why does the stable log-sum-exp subtract $\max_j x_j$ before the exponential?
7. Derive $d/dx\,\operatorname{softplus}(x)$ and identify the sigmoid.
8. Why does `other=0.0` work for the addition kernel but not for the maximum in
   log-sum-exp?

## How to run

```bash
cd part3_math/ch07_arithmetic
jupyter notebook 07_complete.ipynb
```

- **CPU.** Every numerical cell, the Method 01 reference, the spine tests, the fused
  reference, and the backward check run. GPU tests print `SKIP <name>: ...` and the
  notebook continues.
- **Interpreter smoke test.** Without a GPU:

```bash
cd 00_Basic
TRITON_INTERPRET=1 PYTHONPATH=$PWD python tools/interpreter_smoke.py
```

- **Power path.** The micro-kernel uses the positive-base `exp2(y * log2(x))` route on the
  spine. The general `libdevice.pow(x, y)` route mentioned for signed bases is GPU-only;
  the Triton interpreter does not implement libdevice.

## Exercises

The notebook ends with ten exercises: the unary skeleton rewrite, Python-versus-Triton
floor semantics, three ways to cube, base-2 exponential accuracy, stable softplus, safe
normalization, NaN policy, the backward of `rsqrt`, fusion traffic accounting, and a
family kernel with an `OP` selector. Hints are in
[`../../docs/SOLUTIONS.md`](../../docs/SOLUTIONS.md).
