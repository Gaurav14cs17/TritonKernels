# Chapter 8 — Comparisons and Logical Operations

Part 3: Small mathematical operations.

> **The thesis of this chapter.** A comparison does not branch; it produces a
> tile of predicates. A mask does not change control flow; it guards lanes.
> `tl.where` does not branch; it selects lane by lane. Once you see that, the
> six comparisons, the three logical operators, and the conditional select are
> one idea with different truth tables.

[Open the chapter notebook](08_complete.ipynb)

[Index](../../00_index.ipynb) · [← Previous](../ch07_arithmetic/07_complete.ipynb) · [Next →](../ch09_activations/09_complete.ipynb)

**Estimated time:** 55–70 min · **Difficulty:** Intermediate

## Learning objectives

- Read a comparison as an elementwise map from numbers to predicates.
- State the **same-shape rule** and the **broadcast rule** for comparisons,
  masks, and `tl.where`.
- Build a band mask, its complement, and a multi-condition mask with parentheses
  and De Morgan's laws.
- Select values with `tl.where` instead of a per-lane branch, and explain why
  both arms are evaluated.
- Write a Triton kernel that stores a mask, selects values, and fuses a mask with
  arithmetic.
- Derive the backward pass of `where` and of a piecewise-linear mask.

## Connections

- **Builds on.** Chapter 7: the elementwise lift, `tl.load`/`tl.store`, the
  tail mask, and the meaning of `other=`.
- **Uses.** The six comparisons as predicates, Boolean combination (`&`, `|`,
  `~`), and `tl.where` as a branch-free select.
- **Leads to.** Chapter 9 uses masks for stable activations and the activation
  kernels; Chapter 10 applies the same broadcast rule to larger shapes.
- **Thread.** Comparison -> mask -> guard -> select -> fuse. Every section adds
  one link and names the earlier link it uses.

## Prerequisites

- Chapter 7: elementwise arithmetic, `tl.maximum`, `tl.minimum`, and the meaning
  of `other=`.
- Basic Boolean algebra: truth tables, `&`, `|`, `~`, and De Morgan's laws.
- No prior GPU experience is assumed.

## How the chapter is built

The notebook is a single reading path, in four parts plus a backward section.

| Part | Contents | Why it is there |
|---|---|---|
| **I** | Comparison contract, six predicates, broadcasting, Boolean algebra, `tl.where` | establishes the rules every later cell uses |
| **II** | Method 01: scalar predicates, masks, and selects | the reference implementation |
| **III** | Method 02: Triton comparison, `where`, and dead-zone kernels | the workhorse kernels |
| **IV** | Method 03: soft-threshold, compile-time mask selector, bit code, masked mean | fusion and production shapes |
| **★** | Capstone: masked normalization | comparison + select + arithmetic in one pass |
| **B** | Forward and backward | the derivative of a select and a mask |

## The rules this chapter teaches

### Same-shape rule

If both operands are full tensors with no broadcasting, every dimension must
match:

$$
\operatorname{shape}(x)=\operatorname{shape}(y)
\quad\Longrightarrow\quad
\operatorname{shape}(m)=\operatorname{shape}(x).
$$

For example, `(4,3)` and `(4,3)` compare element by element and produce a
`(4,3)` Boolean mask.

### Broadcast rule

If the shapes differ, align them on the **right**. For each axis, the sizes must
be equal or one of them must be `1`. The result has the broadcast shape:

$$
\operatorname{shape}(m)
=
\operatorname{broadcast\_shapes}
\bigl(\operatorname{shape}(x),\operatorname{shape}(y)\bigr).
$$

Examples:

| Left | Right | Result | Why |
|---|---|---|---|
| `(4,3)` | `(4,3)` | `(4,3)` | same shape |
| `(4,3)` | `(3,)` | `(4,3)` | row vector broadcasts over rows |
| `(4,3)` | `(4,1)` | `(4,3)` | column vector broadcasts over columns |
| `(4,3)` | scalar | `(4,3)` | scalar broadcasts to any shape |
| `(4,3)` | `(3,4)` | error | last axes are `3` and `4`, neither is `1` |

### Scalar rule

A scalar (0-D) or length-1 tensor broadcasts to any compatible shape. In
`tl.where`, a scalar arm is broadcast against the condition tile.

### `tl.where` rule

$$
y_i=c_i a_i+(1-c_i)b_i,
\qquad
\operatorname{shape}(y)
=
\operatorname{broadcast\_shapes}
\bigl(\operatorname{shape}(c),\operatorname{shape}(a),\operatorname{shape}(b)\bigr).
$$

Both arms are evaluated. If an arm divides by zero or overflows an exponential,
that value is computed before the select; guard the input of the dangerous
expression, not the output.

### Kernel rule

Triton kernels see raw pointers and explicit offsets. A kernel does **not** infer
a broadcast from strides. The host must pass same-length contiguous buffers, or
the kernel must index the broadcast operand explicitly (`offs % N`, a 2-D grid,
or an explicit stride).

### Mask rule

`mask = offs < n_elements` guards both the load and the store. A comparison may
execute on a masked lane; only the memory access is suppressed. When a masked
load feeds a reduction, `other` must be the neutral element of that reduction.

## Chapter map

| § | Topic | Key formula | Shape rule | Applied kernel |
|---|---|---|---|---|
| **8.1** | Equal | $m_i=[x_i=y_i]$ | same shape, or broadcast | six-comparison bit code |
| **8.2** | Not equal | $m_i=[x_i\ne y_i]$ | same shape, or broadcast | six-comparison bit code |
| **8.3** | Greater than | $m_i=[x_i\gt y_i]$ | same shape, or broadcast | six-comparison bit code |
| **8.4** | Greater or equal | $m_i=[x_i\ge y_i]$ | same shape, or broadcast | six-comparison bit code |
| **8.5** | Less than | $m_i=[x_i\lt y_i]$ | same shape, or broadcast | six-comparison bit code |
| **8.6** | Less or equal | $m_i=[x_i\le y_i]$ | same shape, or broadcast | six-comparison bit code |
| **8.7** | Logical AND | $m_i=a_i\land b_i$ | same shape, or broadcast | Boolean mask combinations |
| **8.8** | Logical OR | $m_i=a_i\lor b_i$ | same shape, or broadcast | Boolean mask combinations |
| **8.9** | Logical NOT | $m_i=\lnot a_i$ | unchanged | Boolean mask combinations |
| **8.10** | Boolean masks | $m_i\in\{0,1\}$ | mask shape = guarded tile shape | masked load/store |
| **8.11** | `tl.where()` | $y_i=c_i a_i+(1-c_i)b_i$ | broadcast of `cond`, `a`, `b` | conditional selection |
| **8.12** | Combining conditions | $m_i=a_i\land b_i\land c_i$ | same shape, or broadcast | dead-zone filter |
| **8.13** | Runnable mask/select | $y_i=m_i a_i+(1-m_i)b_i$ | same shape | soft-threshold |
| **★** | Capstone | $y_i=x_i/\hat d_i$ | same shape | masked normalization |

## Worked numerical example

The chapter spine is

$$
x=(-2,-1,-0.5,0,0.5,1,2),
\qquad
y=0.
$$

The six masks are

| Predicate | Mask |
|---|---|
| $x=0$ | `F F F T F F F` |
| $x\ne0$ | `T T T F T T T` |
| $x\gt 0$ | `F F F F T T T` |
| $x\ge0$ | `F F F T T T T` |
| $x\lt 0$ | `T T T F F F F` |
| $x\le0$ | `T T T T F F F` |

The band and its complement are

$$
[-1\lt x\lt 1]=\texttt{F F T T T F F},
\qquad
[x\le-1\ \text{or}\ x\ge1]=\texttt{T T F F F T T}.
$$

The 6-bit code uses `eq=1`, `ne=2`, `gt=4`, `ge=8`, `lt=16`, `le=32` and
produces

```text
[50, 50, 18, 1, 6, 14, 14]
```

The notebook asserts every row and every code.

The broadcast example uses

$$
A=\begin{bmatrix}1&2&3\\4&5&6\end{bmatrix},
\qquad
r=(2,3,4),
\qquad
c=\begin{bmatrix}3\\5\end{bmatrix},
$$

and checks

$$
(A\ge r)\land(A\le c+2)
=
\begin{bmatrix}
0&0&0\\
1&1&1
\end{bmatrix},
$$

with `(2,3)`, `(3,)`, and `(2,1)` all broadcasting to `(2,3)`. It also shows the
error for `(2,3)` with `(3,2)`.

## The canonical kernels

The comparison kernel is the chapter skeleton with a compile-time selector:

```python
@triton.jit
def m8_compare_kernel(x_ptr, y_ptr, out_ptr, n_elements,
                      OP: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements

    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    y = tl.load(y_ptr + offs, mask=mask, other=0.0)

    if OP == CMP_EQ:
        result = x == y
    elif OP == CMP_NE:
        result = x != y
    # ... gt, ge, lt, le ...

    tl.store(out_ptr + offs, result.to(tl.int8), mask=mask)
```

The selection kernel adds the two arms:

```python
z = tl.where(cond, a, b)
tl.store(out_ptr + offs, z, mask=mask)
```

The fused mask-select kernel keeps the whole piecewise function in registers:

```python
y = tl.where(x > t, x - t,
             tl.where(x < -t, x + t, 0.0))
```

## Pitfalls

| Pitfall | What happens | Fix |
|---|---|---|
| `x > 0 & x < 1` | `&` binds before the comparisons | `(x > 0) & (x < 1)` |
| `and` / `or` / `not` on tensors | raises, or short-circuits a scalar | use `&`, `\|`, `~` |
| `~` on an integer tensor | bitwise complement | cast to `bool` first |
| `nan` in a comparison | ordered comparisons are false; `!=` is true | decide the NaN policy explicitly |
| float equality | exact test fails after rounding | compare `abs(x-y) <= eps` |
| storing a predicate without a cast | dtype mismatch or a four-byte mask | `.to(tl.int8)` |
| forgetting the tail mask | out-of-bounds access | `mask = offs < n_elements` |
| `tl.where` with a dangerous arm | `inf`/`nan` computed before selection | guard the input |
| host/kernel broadcast mismatch | wrong stride or length in the kernel | expand + `contiguous()`, or index explicitly |
| boundary mismatch | `>` vs `>=` differs at one value | state the convention and test it |

## Applied examples and capstone

| # | Example | Sections | What it adds |
|---|---|---|---|
| 1 | Six-comparison bit code | 8.1–8.6, 8.10 | six predicates in one integer |
| 2 | Dead-zone filter | 8.7–8.12 | band mask + select |
| 3 | Soft-threshold | 8.11, 8.13 | nested `where` + arithmetic |
| 4 | Masked positive mean | 8.10 | mask + reduction + neutral `other` |
| **★** | **Masked normalization** | 8.11, 8.12 | guard the denominator, not the output |

## Forward and backward

A comparison is piecewise constant, so its derivative is zero almost everywhere.
A select routes the gradient to the chosen arm:

$$
\frac{\partial L}{\partial a}=g\,c,
\qquad
\frac{\partial L}{\partial b}=g\,(1-c),
\qquad
\frac{\partial L}{\partial c}=0
\quad\text{a.e.}
$$

The notebook derives the backward of a piecewise-linear mask, checks it against
autograd, and writes a Triton backward kernel that recomputes the band from `x`
instead of saving a mask.

## How to run

```bash
cd part3_math/ch08_comparisons
jupyter notebook 08_complete.ipynb
```

The notebook imports `ttools.py` by walking up to `00_Basic`. On a CPU-only
machine every numerical and backward cell runs; each Triton test prints
`SKIP ...`. On a GPU every kernel is compared with a PyTorch reference.

A quick interpreter smoke test of the kernels, without a GPU:

```bash
cd 00_Basic
TRITON_INTERPRET=1 PYTHONPATH=$PWD python tools/interpreter_smoke.py
```

## Study checklist

- [ ] Write the six masks and the 6-bit code for the spine by hand.
- [ ] Explain why `x > -1 & x < 1` is wrong and write the parse.
- [ ] Prove De Morgan's law lane by lane for the band mask.
- [ ] State the output shape for `(4,3)` with `(3,)`, `(4,1)`, and `(3,4)`.
- [ ] Break the tail mask and explain which access fails.
- [ ] Rewrite `tl.where(d != 0, x / d, 0)` so it never creates `inf`/`nan`.
- [ ] Derive the gradient of `where(x>0, x, -2*x)` at `x=1` and `x=-1`.
- [ ] Do exercises 2, 5, and 6: they expose the boundary, broadcast, and
      safe-division bugs.

## Exercises

The notebook ends with ten exercises: bit-code extension, open versus closed
bands, the operator trap, De Morgan, bias broadcasting, safe division, NaN
policy, masked reduction, boundary gradients, and fusion traffic. Hints are in
[`../../docs/SOLUTIONS.md`](../../docs/SOLUTIONS.md).
