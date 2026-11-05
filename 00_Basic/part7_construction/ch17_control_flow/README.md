# Chapter 17 — Control Flow

Part 7: Kernel construction and optimization.

> **Goal.** Make control flow a type decision. A `constexpr` condition becomes a compiled
> branch; a uniform runtime scalar becomes a device branch; a per-lane tensor becomes a mask
> or `tl.where`. Static and dynamic loops, unrolling, specialization, and dynamic shapes all
> follow from that one distinction.

[Open the notebook](17_complete.ipynb) · [Setup](../../docs/SETUP.md) · [Notation](../../docs/NOTATION.md) · [Glossary](../../docs/GLOSSARY.md) · [API index](../../docs/API_INDEX.md) · [Formula index](../../docs/FORMULA_INDEX.md) · [Exercise hints](../../docs/SOLUTIONS.md)

[Index](../../00_index.ipynb) · [← Previous](../../part6_reductions/ch16_combining_ops/16_complete.ipynb) · [Next →](../ch18_kernel_construction/18_complete.ipynb)

**Estimated time:** 55–70 min · **Difficulty:** Intermediate

## Learning objectives

- Classify a condition as compile-time (`tl.constexpr`), uniform runtime (scalar), or per-lane (tensor).
- Explain why a Python `if` on a `tl.constexpr` is resolved at compile time, while a data-dependent condition must become a mask or `tl.where`.
- Choose `tl.static_range` for a small fixed trip count and `range` for a dynamic one.
- Read mask tables for lane selection and partial tiles.
- Predict when a launch creates a new specialization, and write a dynamic-shape kernel.

## Prerequisites

- Chapter 1: `@triton.jit`, compile-time versus runtime arguments, and `tl.constexpr`.
- Chapter 5: masked load/store and `other=`.
- Chapter 6: offsets, tiles, and boundary masks.
- Chapter 8: comparison masks, `&` versus `and`, and `tl.where`.

## Connections

- **Recurring idea.** `constexpr` condition → compiled branch; uniform runtime scalar → device branch; per-lane tensor → mask or `tl.where`.
- **Chain.** §17.1 condition type → §17.2 static loop → §17.3 runtime loop → §17.4 unrolling cost → §17.5 execution uniformity → §17.6 masks → §17.7 specialization → §17.8 dynamic shapes.
- **Payoff.** The same distinction explains unrolling, cache keys, and the static-block/dynamic-`N` pattern in production kernels.

## Study ladder

- **Method 01 — scalar reference.** A Python `if`, a Python `for`, and scalar arithmetic: the correctness oracle, intentionally too slow for a GPU.
- **Method 02 — control-flow kernels.** A `constexpr` mode plus `tl.where`, a `tl.static_range` polynomial, and a runtime `range` polynomial.
- **Method 03 — production shapes.** One kernel with `MODE`, `ACT`, and `BLOCK` constexprs; one with a runtime $N$; a grid-stride capstone.

**Study tip.**

- Fix the spine $x=(1,2,3,4,5,6)$ and threshold $T=3$; write the mask, both arms ($x$ and $0$), and the selected vector before reading code.

| $i$ | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| $x_i$ | 1 | 2 | 3 | 4 | 5 | 6 |
| $[x_i\gt T]$ | F | F | F | T | T | T |
| selected | 0 | 0 | 0 | 4 | 5 | 6 |

- For the partial tile with `BLOCK=4`, program 1 owns $4,5,6,7$; the mask is `T,T,F,F`, and only lanes 4 and 5 are live.

---

## Rules

### `if` and compile-time conditions

- **Connection / Rule.** Chapter 16 combined reductions; this chapter starts at the scalar condition that chooses which reduction or epilogue runs. A Python `if` on a `tl.constexpr` is resolved at compile time and only the taken branch is emitted; a runtime scalar becomes a uniform device branch; a tensor condition is a predicate tile, not a Python `bool`.
- **Formula / Example.** `MODE: tl.constexpr` with `if MODE == 0: y = x` / `elif MODE == 1: y = max(x, 0)` compiles one body; per-lane: $m_i=[x_i\gt T]$, $y_i=m_i x_i+(1-m_i)0$. For $x=(1,2,3,4,5,6)$, $T=3$: $m=(\mathrm F,\mathrm F,\mathrm F,\mathrm T,\mathrm T,\mathrm T)$, $y=(0,0,0,4,5,6)$.
- **Where / Triton / Failure / Bridge.** Selects an algorithm, epilogue, dtype path, or block size; annotate `MODE: tl.constexpr`, and for lanes use `tl.where(x > T, x, 0.0)`. `if x > T:` on a tile raises a compile error, and a data-dependent branch serializes both arms. A compile-time condition can also choose a loop, which is §17.2.

### `tl.static_range()`

- **Connection / Rule.** Section 17.1 used a `constexpr` to choose an expression; here it chooses a loop bound. `tl.static_range(start, end, step)` is a compile-time loop; the trip count must be a `constexpr`, and the induction variable is a Python integer.
- **Formula / Example.** Start $y_0=1$, iterate $y_{k+1}=y_kx+1$ four times: $y_4=1+x+x^2+x^3+x^4$. At $x=2$, $y_4=1+2+4+8+16=31$; `tl.static_range(0,4)` is four copies of the body.
- **Where / Triton / Failure / Bridge.** Small fixed $K$: a 3-tap stencil, Newton steps, four accumulators, or a small polynomial; `for k in tl.static_range(0, K):` with `K: tl.constexpr`. Large $K$ bloats code and registers, and a runtime $K$ is a compile error. A static range still needs a fixed count; §17.3 removes that restriction.

### `range()`

- **Connection / Rule.** Section 17.2 fixed the trip count; here the count is a runtime scalar while the body stays one copy. `range(start, end, step)` emits one body, a compare, and a back-edge; the loop-carried value is an SSA value.
- **Formula / Example.** Start $y_0=1$, iterate $y_{k+1}=c y_k$ for runtime $n$: $y_n=c^n$. For $c=1.1$, $n=4$: $y_4=1.1^4=1.4641$; $n=0$ returns $1$.
- **Where / Triton / Failure / Bridge.** Long reductions over $K$ blocks, dynamic sequence lengths, grid-stride loops, and any launch-time trip count; `for k in range(0, tl.cdiv(K, BLOCK_K)):`. An unbounded bound hangs, and a tensor bound with more than one element is not a scalar trip count. Sections 17.2–17.3 give the two loop forms; §17.4 compares their cost.

### Loops and unrolling

- **Connection / Rule.** Sections 17.2–17.3 defined the static and runtime loops; this section compares what each costs. Unrolling replicates the body and removes the back-edge; a runtime loop keeps one body and pays one branch per trip.
- **Formula / Example.** $C_{\text{loop}}=c_b+c_j$, $C_{\text{unroll}}=Kc_b$, $I_{\text{loop}}=K(c_b+c_j)$, $I_{\text{unroll}}=Kc_b$. For $K=4$, $c_b=8$, $c_j=2$: $C_{\text{loop}}=10$ vs $C_{\text{unroll}}=32$; $I_{\text{loop}}=40$ vs $I_{\text{unroll}}=32$; unrolling saves $8$ branch instructions and costs $22$ extra static instructions.
- **Where / Triton / Failure / Bridge.** Unroll small, hot, independent loops; keep long, memory-bound, or dependent loops rolled. `tl.static_range` is explicit unrolling, while a plain `range` may be partially unrolled. A dependent recurrence gains no parallelism, and a large unroll can spill registers. The loop decision is one axis; §17.5 adds execution uniformity.

### Conditional execution

- **Connection / Rule.** Section 17.1 classified conditions; here the classification becomes an execution strategy. A compile-time condition selects code; a uniform runtime scalar is a device branch shared by every lane; a per-lane condition is a predicate tile.
- **Formula / Example.** Uniform early exit: `if pid * B >= N: return`; per-lane select: $y_i=m_i a(x_i)+(1-m_i)b(x_i)$. For $x=(1,\dots,6)$, $T=3$, $a(x)=x$, $b(x)=0$: $m=(\mathrm F,\mathrm F,\mathrm F,\mathrm T,\mathrm T,\mathrm T)$, $y=(0,0,0,4,5,6)$.
- **Where / Triton / Failure / Bridge.** Uniform branches skip whole tiles; per-lane conditions implement ReLU, dropout, attention masks, and piecewise activations. Use `if pid * BLOCK >= n_elements: return` for tiles and `tl.where(cond, a, b)` for lanes. A tile branch is rejected, and both arms of `tl.where` are evaluated. Once the per-lane condition is a select, §17.6 makes it a mask for values and memory.

### Mask-based execution

- **Connection / Rule.** Section 17.5 turned a per-lane condition into a select; here the predicate guards memory and chooses values. A mask is a Boolean tile shaped like the guarded tile; it guards memory with `mask=` on `tl.load`/`tl.store` and selects values with `tl.where`.
- **Formula / Example.** Bounds mask `offs < n_elements`; value mask $m_i=[x_i\gt T]$, $y_i=m_i x_i$. With `BLOCK=4`, $N=6$: program 1 owns $4,5,6,7$, mask `T,T,F,F`, loads $5,6$ and two zeros; threshold select: $m=(\mathrm F,\mathrm F,\mathrm F,\mathrm T,\mathrm T,\mathrm T)$, $y=(0,0,0,4,5,6)$.
- **Where / Triton / Failure / Bridge.** Tail handling, padding, variable sequence lengths, causal attention, dropout, and gated activations; `mask = offs < n_elements`, `tl.load(p + offs, mask=mask, other=0.0)`, and `tl.store(p + offs, y, mask=mask)`. Wrong mask shape broadcasts or fails, a forgotten store mask writes past the end, and a dangerous `tl.where` arm is evaluated before selection. Masks handle one launch; §17.7 shows how `constexpr` choices become separate compiled bodies.

### Compile-time specialization

- **Connection / Rule.** Section 17.1 chose a branch and §17.6 chose a mask; here the choice itself is a compilation boundary. A kernel is compiled per specialization key: `constexpr` arguments, argument dtypes, and compiler options; runtime arguments are not part of the key.
- **Formula / Example.** $\kappa=(\texttt{MODE},\texttt{ACT},\texttt{BLOCK},\text{dtypes})$. $(0,0,4)$, $(1,0,4)$, $(0,1,4)$, $(0,0,8)$ produce four binaries; $n=17$ and $n=1000$ still produce four.
- **Where / Triton / Failure / Bridge.** One source serves fp16/bf16/fp32, several activation epilogues, and several block sizes; annotate `BLOCK`, `MODE`, and `ACT` as `tl.constexpr` and keep `n_elements` a normal argument. `n_elements` as `constexpr` causes a JIT storm, and a tensor where a `constexpr` is expected fails at launch. A key fixes the block; §17.8 keeps the block fixed while the tensor size varies.

### Dynamic vs static shapes

- **Connection / Rule.** Section 17.7 fixed the `constexpr` block; here the block stays static while $N$ becomes a runtime argument. Tile shapes must be compile-time; tensor sizes may be runtime scalars.
- **Formula / Example.** Host grid $\lceil N/B\rceil$; device mask `offs < n_elements`; signature `(x_ptr, out_ptr, n_elements, BLOCK: tl.constexpr)`. With $B=4$: $N=6$ → grid 2, offsets 4–7, mask `T,T,F,F`; $N=10$ → grid 3, offsets 8–11, mask `T,T,F,F`; $N=16$ → grid 4, offsets 12–15, mask `T,T,T,T`.
- **Where / Triton / Failure / Bridge.** Batch size, sequence length, vocabulary size, and token count are dynamic; use host grid `(triton.cdiv(n, BLOCK),)` and never write `tl.arange(0, n_elements)`. `tl.arange(0, N)` with runtime `N` is a compile error, and assuming $N\bmod B=0$ reads or writes out of bounds. **Where this leads.** Parts II–IV turn these rules into a scalar reference, kernel controls, and specialization plus dynamic shapes.

---

## Worked numerical example

### Branch and mask

- Spine: $x=(1,2,3,4,5,6)$, threshold $T=3$; mask $m_i=[x_i\gt T]$.
- Branch result: $y_i=m_i x_i+(1-m_i)0=(0,0,0,4,5,6)$.

| $i$ | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| $x_i$ | 1 | 2 | 3 | 4 | 5 | 6 |
| $m_i$ | F | F | F | T | T | T |
| $y_i$ | 0 | 0 | 0 | 4 | 5 | 6 |

- Two-arm select: $a_i=x_i+1=(2,3,4,5,6,7)$, $b_i=2x_i=(2,4,6,8,10,12)$, $z_i=m_i b_i+(1-m_i)a_i=(2,3,4,8,10,12)$; both arms are evaluated.

### Unrolled loop

- Start $y_0=1$; iterate $y_{k+1}=y_kx+1$ four times: $y_4=1+x+x^2+x^3+x^4$.
- At $x=2$: $y_4=16+8+4+2+1=31$; `tl.static_range(0,4)` is four copies of the body.

### Dynamic bound

- Start $y_0=1$; iterate $y_{k+1}=1.1y_k$ for $n$ trips: $y_n=1.1^n$.
- $n=4$: $y_4=1.4641$; $n=0$: $1$; $n=8$: $2.14358881$; the source does not change.

### Dynamic shape

| $N$ | grid | last offsets | tail mask | live |
|---:|---:|---|---|---:|
| 6 | 2 | 4,5,6,7 | T,T,F,F | 2 |
| 10 | 3 | 8,9,10,11 | T,T,F,F | 2 |
| 16 | 4 | 12,13,14,15 | T,T,T,T | 4 |

- Every row is checked by `assert` in the notebook; no table is printed.

### Compile-time versus runtime

- `BLOCK`, `MODE`, `ACT`: `constexpr`; changing one creates a new binary.
- `n_elements`, `threshold`, `shift`, `n_steps`: runtime scalars; changing one does not.
- Pointer arguments are runtime addresses; argument dtypes are part of the key.

---

## Chapter map

| § | Topic | Rule | Example | Kernel/demo |
|---|---|---|---|---|
| **17.1** | `if` and compile-time conditions | `constexpr` condition → one emitted branch | `MODE` chooses identity/threshold/select | `mode_mask_kernel` |
| **17.2** | `tl.static_range()` | fixed trip count → body replicated | $y\leftarrow yx+1$ four times | `poly_static_kernel` |
| **17.3** | `range()` | runtime trip count → one body + back-edge | $y\leftarrow yc$ for $n$ trips | `poly_range_kernel` |
| **17.4** | Loops and unrolling | fewer branches vs code size and registers | $C_{\text{rolled}}$ vs $C_{\text{unrolled}}$ | trade-off model |
| **17.5** | Conditional execution | scalar branch uniform; lane condition select | `if pid*BLOCK >= n: return` | `mode_mask_kernel` |
| **17.6** | Mask-based execution | mask guards memory and selects values | $m=[x\gt 3]$, $y=mx$ | `mode_mask_kernel` |
| **17.7** | Compile-time specialization | new binary per `constexpr` key | `(MODE, ACT, BLOCK)` | `specialize_kernel` |
| **17.8** | Dynamic vs static shapes | static block, runtime $N$, masked tail | $N=6,10,16$ with $B=4$ | `dynamic_scale_kernel` |
| **★** | Grid-stride capstone | fixed grid loops over many tiles | `for start in range(pid*B, n, P*B)` | `grid_stride_kernel` |

## Pitfalls

| Pitfall | What happens | Fix |
|---|---|---|
| `if x > T:` on a tile | compile error | `tl.where(x > T, ...)` or a mask |
| `and` / `or` / `not` on tensors | raises or short-circuits | `&`, `\|`, `~` with parentheses |
| `constexpr` on a runtime value | recompiles per value | keep $N$ a normal argument |
| `tl.static_range` with a runtime bound | compile error | `range(0, n_steps)` |
| unbounded loop | hang | clamp the bound; test $n=0,1,N$ |
| dynamic shape as `constexpr` | JIT storm | static `BLOCK`, runtime $N$, tail mask |
| mask shape mismatch | wrong lanes or broadcast error | mask shape = guarded tile |
| large unroll | code bloat, spills | unroll small independent loops |
| forgotten store mask | out-of-bounds writes | mask load and store |
| dangerous `tl.where` arm | `inf`/`nan` before select | guard the input |

## How to run

- CPU: `cd part7_construction/ch17_control_flow && jupyter notebook 17_complete.ipynb`. The numerical asserts, Method 01 reference, unrolling model, and mask-versus-branch check run; GPU tests print `SKIP ...`.
- Interpreter: `cd 00_Basic && TRITON_INTERPRET=1 PYTHONPATH=$PWD python tools/interpreter_smoke.py`. The kernels run on the CPU; they check values and masks, not speed.
- The final cell dispatches every test.

## Study checklist

- [ ] Write the mask and both arms for $x=(1,\dots,6)$, $T=3$.
- [ ] Explain why `if x > T` fails on a tile but `if MODE == 0` is fine.
- [ ] Expand `tl.static_range(0,4)` for $y\leftarrow yx+1$ and evaluate at $x=2$.
- [ ] Compute rolled and unrolled instruction counts for $K=8$, $c_b=8$, $c_j=2$.
- [ ] Give the grid and tail mask for $N=10$, $B=4$.
- [ ] List two arguments that create a specialization and two that do not.
- [ ] Rewrite a per-lane `if` as `tl.where` without evaluating a dangerous arm.
- [ ] Explain why a grid-stride loop can cover $N=37$ with two programs.

**Where this leads.** The exercises extend each rule: scalar-to-mask, unroll factor, dynamic bound, specialization count, grid-stride, safe select, and masked reduction.

## Exercises

1. Rewrite `method01_branch` as a Triton kernel with a runtime threshold $T$; test $T=0,3,10$.
2. Add `MODE=3` (signed square root) to `mode_mask_kernel` and test both signs.
3. Run `poly_static_kernel` with $K=1,2,4,8,16$; describe where code size and registers grow.
4. Write a dynamic-bound accumulator with `n_steps` runtime; check $0,1,4,8$ against Python.
5. For $B=4$, tabulate grid, last offsets, and tail mask for $N=6,7,8,9,10$.
6. Count the binaries for six `(MODE, ACT)` pairs plus two values of $N$; justify the count.
7. Run the grid-stride kernel with one program and $N=37$; how many trips does it take?
8. Add a uniform early exit and prove it does not change the output.
9. Implement a safe select $y=\texttt{tl.where}(x\ne0,1/x,0)$ with no `inf`/`nan`.
10. Use `other=-inf` in a masked maximum; why is `other=0` wrong for negative data?

Hints are in [`../../docs/SOLUTIONS.md`](../../docs/SOLUTIONS.md).
