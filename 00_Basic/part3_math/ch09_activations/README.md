# Chapter 9 — Activation Functions

Part 3: Small mathematical operations.

> **The thesis of this chapter.** An activation is a scalar nonlinearity lifted
> to a tile. The forward and the backward are both elementwise, so the shape
> never changes and the traffic is the same as a copy. The interesting decisions
> are numerical stability, the boundary convention, and whether the bias add is
> fused into the same pass.

[Open the chapter notebook](09_complete.ipynb)

[Index](../../00_index.ipynb) · [← Previous](../ch08_comparisons/08_complete.ipynb) · [Next →](../../part4_broadcasting/ch10_broadcasting/10_complete.ipynb)

**Estimated time:** 65–80 min · **Difficulty:** Intermediate

## Learning objectives

- Read an activation as an elementwise lift that preserves the input shape.
- State the **same-shape rule** and the **broadcast rule** for a fused bias +
  activation.
- Implement ten activations in one Triton kernel selected by a `tl.constexpr`.
- Choose a numerically stable form for sigmoid, tanh, GELU, and softplus.
- Derive and check the backward pass of every activation in the table.
- Fuse a bias add into the activation without materializing the intermediate.

## Connections

- **Builds on.** Chapter 7: `exp`, `log`, `maximum`, `clamp`, and the domain
  warnings. Chapter 8: `tl.where`, predicates, and masks.
- **Uses.** The activation zoo as a chain: ReLU -> leaky ReLU -> sigmoid ->
  tanh -> GELU/SiLU -> hard variants -> softplus/clamp.
- **Leads to.** Chapter 10 (broadcasting the bias), Chapter 14 (row reductions
  for normalization), and Chapter 16 (stable softmax/LayerNorm recipes).
- **Thread.** A gate keeps, scales, or suppresses a value lane by lane; the
  shape never changes and the backward is another elementwise pass.

## Prerequisites

- Chapter 7: `exp`, `log`, `maximum`, `minimum`, `clamp`, and the domain rules.
- Chapter 8: `tl.where`, Boolean predicates, and masks.
- Familiarity with the sigmoid and the chain rule.

## How the chapter is built

| Part | Contents | Why it is there |
|---|---|---|
| **I** | Activation contract, the zoo, stability, derivatives, worked table, broadcasting | the mathematics and the numerical rules |
| **II** | Method 01: scalar Python activations | the reference implementation |
| **III** | Method 02: one Triton tile kernel with ten `MODE`s | the workhorse kernel |
| **IV** | Method 03: fused bias + activation, SwiGLU, softplus threshold | production-shaped epilogues |
| **★** | Capstone: a fused MLP epilogue | 2-D tile + bias broadcast + selector |
| **B** | Forward and backward | the full derivative table checked with autograd |

## The rules this chapter teaches

### Same-shape rule

An activation is elementwise, so it cannot change the shape:

$$
\operatorname{shape}(f(x))=\operatorname{shape}(x).
$$

A gate is a Hadamard product, not a contraction: `a` and `b` must have the same
shape, and the output has that shape.

### Broadcast rule for a fused bias

For a linear output $X\in\mathbb R^{M\times N}$ and a bias $b\in\mathbb R^{N}$,

$$
\operatorname{shape}(X+b)
=
\operatorname{broadcast\_shapes}((M,N),(N,))
=
(M,N).
$$

The bias broadcasts over the row axis. The kernel loads one bias value per
column and adds it to every row tile.

### Scalar rule

A scalar broadcasts to any compatible shape. A threshold or an epsilon may be a
runtime scalar; it does not change the output shape.

### Shape errors to recognize

| Left | Right | Result | Why |
|---|---|---|---|
| `(7,13)` | `(13,)` | `(7,13)` | per-column bias |
| `(7,13)` | `(7,1)` | `(7,13)` | per-row bias |
| `(7,13)` | `(7,)` | error | last axes are `13` and `7` |
| `(7,13)` | scalar | `(7,13)` | scalar broadcasts |
| `(7,13)` | `(7,13)` | `(7,13)` | same-shape gate |

### Tile rule

A 2-D kernel needs both masks:

```python
rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
mask = (rows < M)[:, None] & (cols < N)[None, :]
```

The bias load uses only the column mask because the bias has no row axis.

### Dtype rule

Load into `float32`, evaluate the activation in `float32`, cast once at the
store. `exp`, `log`, `erf`, and `tanh` are defined for `float32` and `float64`;
a narrow input must be promoted first. The cost is registers, not bytes.

### Stability rule

Use the stable algebraic form, not a clamp:

$$
\sigma(x)=\frac{1}{1+e^{-x}},
\qquad
\tanh x=2\sigma(2x)-1,
$$

$$
\operatorname{softplus}(x)=\max(x,0)+\log(1+e^{-\lvert x\rvert}).
$$

Clamping the output hides overflow and changes the function.

## Chapter map

| § | Activation | Key formula | Shape rule | Applied kernel |
|---|---|---|---|---|
| **9.1** | ReLU | $\max(x,0)$ | same shape | ReLU / leaky ReLU |
| **9.2** | Leaky ReLU | $\max(x,0)+\alpha\min(x,0)$ | same shape | ReLU / leaky ReLU |
| **9.3** | Sigmoid | $1/(1+e^{-x})$ | same shape | sigmoid / tanh |
| **9.4** | Tanh | $\tanh x$ | same shape | sigmoid / tanh |
| **9.5** | GELU | $\frac{x}{2}(1+\operatorname{erf}(x/\sqrt2))$ | same shape | modern activations |
| **9.6** | SiLU / Swish | $x\,\sigma(x)$ | same shape | modern activations |
| **9.7** | Hard sigmoid | $\min(\max((x+3)/6,0),1)$ | same shape | modern activations |
| **9.8** | Hard swish | $x\,\mathrm{hard\_sigmoid}(x)$ | same shape | modern activations |
| **9.9** | Softplus | $\max(x,0)+\ln(1+e^{-\lvert x\rvert})$ | same shape | softplus + threshold |
| **9.10** | Clamp / threshold | $\min(\max(x,l),u)$ | same shape | softplus + threshold |
| **9.11** | Runnable activation | $y_i=\operatorname{act}(x_i)$ | same shape | compile-time selector |
| **★** | Capstone | $y=\operatorname{act}(x+b)$ | bias broadcasts on the last axis | fused MLP epilogue |

## Worked numerical example

The spine is $x=(-2,-1,0,1,2)$. The forward table, to three decimals, is

| $x$ | ReLU | Sigmoid | Tanh | GELU | SiLU | Hard sigmoid | Hard swish | Softplus |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| -2 | 0.000 | 0.119 | -0.964 | -0.045 | -0.238 | 0.167 | -0.333 | 0.127 |
| -1 | 0.000 | 0.269 | -0.762 | -0.159 | -0.269 | 0.333 | -0.333 | 0.313 |
| 0 | 0.000 | 0.500 | 0.000 | 0.000 | 0.000 | 0.500 | 0.000 | 0.693 |
| 1 | 1.000 | 0.731 | 0.762 | 0.841 | 0.731 | 0.667 | 0.667 | 1.313 |
| 2 | 2.000 | 0.881 | 0.964 | 1.955 | 1.762 | 0.833 | 1.667 | 2.127 |

The notebook asserts every entry to `1e-3` and compares the scalar reference
with PyTorch.

The derivative table, checked against autograd, is

| Activation | $f'(x)$ |
|---|---|
| ReLU | $[x\gt 0]$ |
| Leaky ReLU | $1$ if $x\gt 0$, $\alpha$ otherwise |
| Sigmoid | $\sigma(1-\sigma)$ |
| Tanh | $1-\tanh^2$ |
| GELU | $\Phi(x)+x\phi(x)$ |
| SiLU | $\sigma(1+x(1-\sigma))$ |
| Hard sigmoid | $1/6$ on $(-3,3)$, else $0$ |
| Hard swish | $0$ / $(2x+3)/6$ / $1$ on the three pieces |
| Softplus | $\sigma(x)$ |
| Clamp | $1$ on $(l,u)$, else $0$ |

The stability cell shows the naive softplus producing `inf` at $x=89$ and $100$
in `float32`, while the stable form stays finite. It also shows
`exp(x)/(1+exp(x))` producing `nan` at $x=100$.

## The canonical kernels

One kernel, ten compile-time modes:

```python
@triton.jit
def m9_activation_kernel(x_ptr, out_ptr, n_elements,
                         MODE: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    if MODE == 0:
        y = tl.maximum(x, 0.0)
    elif MODE == 3:
        y = 2.0 * tl.sigmoid(2.0 * x) - 1.0
    elif MODE == 5:
        y = x * tl.sigmoid(x)
    elif MODE == 8:
        y = tl.maximum(x, 0.0) + tl.log(1.0 + tl.exp(-tl.abs(x)))
    # ... the other six modes ...

    tl.store(out_ptr + offs, y, mask=mask)
```

The fused epilogue indexes the bias by column:

```python
offsets = rows[:, None].to(tl.int64) * N + cols[None, :]
x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
b = tl.load(bias_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
z = x + b[None, :]
y = tl.sigmoid(z)          # or any other mode
tl.store(out_ptr + offsets, y, mask=mask)
```

## Pitfalls

| Pitfall | What happens | Fix |
|---|---|---|
| naive `log(1+exp(x))` | `inf` for $x\gtrsim88.7$ | stable softplus |
| `exp(x)/(1+exp(x))` | `nan` for large positive $x$ | `tl.sigmoid` or `1/(1+exp(-x))` |
| tanh as a quotient | overflow near $\lvert x\rvert\approx44$ | `2*sigmoid(2x)-1` |
| GELU tanh approximation at `1e-6` | false failure ($\sim10^{-3}$ accurate) | compare with a stated tolerance |
| ReLU at zero | gradient convention differs | state `>` versus `>=` and test the join |
| hard activations at $\pm3$ | derivative jumps | test the boundaries |
| `tl.where` with a dangerous arm | `inf`/`nan` computed first | guard the input |
| bias `(M,)` for `(M,N)` | not broadcastable | use `(N,)` or `(M,1)` |
| missing column mask | out-of-bounds bias/output | outer AND of row and column masks |
| narrow dtype throughout | overflow in `exp` or the product | compute in `float32`, cast once |

## Applied examples and capstone

| # | Example | Sections | What it adds |
|---|---|---|---|
| 1 | ReLU / leaky ReLU | 9.1, 9.2 | two branch-free piecewise maps |
| 2 | Sigmoid / tanh | 9.3, 9.4 | stable logistic and the identity |
| 3 | Modern activations | 9.5–9.8 | GELU, SiLU, hard sigmoid, hard swish |
| 4 | Softplus + threshold | 9.9, 9.10 | stable positive part and a clamp |
| 5 | SwiGLU | 9.6 | a gated elementwise product |
| **★** | **Fused MLP epilogue** | 9.1–9.10 | 2-D tile + bias broadcast + selector |

## Forward and backward

For $y=f(x)$ and upstream gradient $g$,

$$
dx=g\,f'(x).
$$

The backward is another elementwise kernel with the same grid, mask, and traffic
plus one extra input. For a fused bias, the bias gradient is a **reduction** over
the broadcast row axis:

$$
\frac{\partial L}{\partial b_n}
=
\sum_{m=1}^{M}\frac{\partial L}{\partial Y_{mn}}.
$$

The notebook checks every derivative in the table against autograd and checks the
bias-gradient rule on a tiny `(2,3)` example.

## How to run

```bash
cd part3_math/ch09_activations
jupyter notebook 09_complete.ipynb
```

The notebook imports `ttools.py` by walking up to `00_Basic`. On a CPU-only
machine every numerical, stability, and backward cell runs; each Triton test
prints `SKIP ...`. On a GPU every kernel is compared with a PyTorch reference,
including partial-tile `(7,13)` cases.

A quick interpreter smoke test of the kernels, without a GPU:

```bash
cd 00_Basic
TRITON_INTERPRET=1 PYTHONPATH=$PWD python tools/interpreter_smoke.py
```

## Study checklist

- [ ] Compute the eight activations at $x=-1,0,1$ by hand and compare.
- [ ] Explain why `1/(1+e^{-x})` is stable but `e^x/(1+e^x)` is not.
- [ ] Derive the GELU derivative and evaluate it at zero.
- [ ] Derive the SiLU derivative and show the product-rule form.
- [ ] Write the stable softplus and explain each term.
- [ ] State the shape of `(7,13) + (13,)` and `(7,13) + (7,)`.
- [ ] Explain why the bias load uses only the column mask.
- [ ] Derive the bias gradient and explain why it is a reduction.
- [ ] Test the hard-activation boundaries at $\pm3$.
- [ ] Do exercises 2, 6, and 9: stability, SwiGLU backward, and fusion traffic.

## Exercises

The notebook ends with ten exercises: stable sigmoid forms, GELU approximation
error, boundary gradients, a runtime leaky slope, per-row bias, SwiGLU backward,
softplus versus threshold order, mixed precision, fusion traffic, and a
softplus-threshold mode. Hints are in
[`../../docs/SOLUTIONS.md`](../../docs/SOLUTIONS.md).
