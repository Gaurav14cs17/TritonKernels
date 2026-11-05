# Chapter 3 — Triton Vector Construction

Part 1: Foundations: what Triton is, how programs run, and how vectors are built.

**Goal.**
- A Triton vector is a **register tile** whose shape and dtype are fixed at trace time.
- Learn the constructor vocabulary: lane indices, constant tiles, like-shaped sentinels, broadcasts, and casts.
- Keep shape and dtype straight from the host reference to the fused kernel.

**Thread.** One idea recurs in every section: the shape and dtype are part of the tile type.
- A tile is built from static shapes and explicit dtypes, never from runtime integers.
- `tl.arange` supplies indices; `tl.full`, `tl.zeros`, and `*_like` supply constants.
- `tl.broadcast_to` changes shape without moving storage; `tl.cast` changes dtype without changing shape.
- `BLOCK: tl.constexpr` and `tl.static_range` keep shape and loop count on the compile-time side.

[Open the chapter notebook](03_complete.ipynb)

[Index](../../00_index.ipynb) · [← Previous](../ch02_program_execution/02_complete.ipynb) · [Next →](../../part2_memory/ch04_pointers/04_complete.ipynb)

**Estimated time:** 55–70 min · **Difficulty:** Beginner–Intermediate

## Connections

- **Chapter 2 → 3.1.** The launch skeleton `offs = pid * BLOCK + tl.arange(0, BLOCK)` supplies the first constructor.
- **3.1 → 3.6.** `arange` makes indices; `full` / `zeros` / `*_like` make constant tiles from a static shape.
- **3.7 → 3.8.** `broadcast_to` changes shape without moving storage; `cast` changes dtype without changing shape.
- **3.9 → 3.12.** Dtypes, compile-time dimensions, scalars, and static ranges explain what the compiler knows before launch.
- **Part I → Part II.** The host PyTorch reference is the specification for every device tile.
- **Part II → Parts III–IV.** The same constructions move into kernels, then fuse with selection and casts.
- **Chapter 3 → Chapter 4.** `base + offset` becomes a memory address, and `BLOCK` becomes the tile a load or store covers.

## Learning objectives

- Build tiles with `tl.arange`, `tl.full`, `tl.zeros`, `tl.ones`, `tl.full_like`, and `tl.zeros_like`.
- State the shape and dtype rule for each constructor, and predict the asserted `shape` and `dtype` before the cell runs.
- Expand a size-1 axis with `tl.broadcast_to`; explain the zero stride on the host and the register reuse in a kernel.
- Cast between `int32`, `int64`, `float16`, `bfloat16`, and `float32`, and say where precision or range is lost.
- Separate compile-time values (`BLOCK`, `tl.constexpr`, `tl.static_range`) from runtime scalars (`pid`, `n_elements`).
- Write and fuse kernels that construct their own tiles inside device code.

**Prerequisites.**
- Chapter 2: program IDs, grids, and `offs = pid * BLOCK + tl.arange(0, BLOCK)`.
- NumPy or PyTorch broadcasting.
- No prior GPU experience.

**Ladder.**
- **Concept.** Part I gives a signature, shape rule, dtype rule, use case, hint, and failure mode for each constructor.
- **Numerical demo.** The worked example fixes every shape, dtype, and value.
- **Method 01.** Part II writes the host PyTorch reference and the zero-stride broadcast view.
- **Method 02.** Part III constructs tiles and broadcasts a column inside kernels.
- **Method 03.** Part IV fuses masked affine, a bias epilogue, and dtype casts.
- **Applied use.** The capstone builds a causal attention mask from two `arange` vectors.

**Study tip.**
- Write the shape and dtype of each constructor on paper before running the notebook.
- `tl.arange(0, 8)` is `(8,)` `int32`; `tl.full((2, 4), 3.0, tl.float32)` is `(2, 4)` `float32`.
- `tl.zeros_like(x)` copies the shape and dtype of `x`.
- `tl.cast(x, tl.float16)` preserves the shape and changes the element width.

**Version note.**
- Triton 3.7.1 exports `tl.full` and `tl.zeros_like` but not `tl.ones` or `tl.full_like`.
- The portable forms are `tl.full(shape, 1.0, dtype)` and `tl.full(x.shape, value, x.dtype)`.
- The host functions `torch.ones`, `torch.full_like`, and `torch.zeros_like` all exist and are used in Method 01.

## Mathematics of a tile

- **Shape.** A tile type is a pair `(shape, dtype)`; with static dimensions $B_0,\ldots,B_{d-1}$, the number of logical lanes is $T=\prod_{k=0}^{d-1} B_k$, and every $B_k$ is a power of two in this backend.
- **Scalar lift.** A scalar operation lifts lane by lane: $Y_{ij}=X_{ij}s$.
- **Cast.** A cast changes only the dtype: $\operatorname{shape}(\operatorname{cast}_d(x))=\operatorname{shape}(x)$.
- **Broadcast.** A broadcast aligns on the right and expands only size-1 axes: $\operatorname{shape}(\operatorname{broadcast\_to}(x,S))=S$ with each $S_k\in\{x_k,1\}$.
- **View.** The host view has a zero stride on each expanded axis, so logical size grows while storage stays fixed; `tl.broadcast_to` does the same in registers.

## Rules

### Shape rule
- Every tile dimension is a `tl.constexpr` power of two.
- Shapes come from static literals such as `tl.full((2, 4), ...)` or from another tile such as `x.shape`, never from a runtime integer.
- A runtime length belongs in the mask and the grid: `offs = pid * BLOCK + tl.arange(0, BLOCK)` uses the static `BLOCK`, while `mask = offs < n_elements` uses the runtime length.

### Dtype rule
- Storage dtype is a bandwidth decision; arithmetic dtype is an accuracy decision.
- Load in `float16`, accumulate in `float32`, cast once at the store.
- `tl.arange` produces `int32`; use `int64` only when an offset can exceed $2^{31}-1$.
- Mixed arithmetic promotes to the wider participating type.

### Broadcast rule
- `tl.broadcast_to(x, shape)` expands only axes whose input size is `1`.
- Dtype does not change and no values move.
- The host result has a zero stride and shares storage.
- `(3,) -> (4,)` is an error because neither axis is `1`.

### Constexpr rule
- `BLOCK`, `BLOCK_M`, `BLOCK_N`, and other `tl.constexpr` arguments are part of the specialization key.
- `BLOCK=8` and `BLOCK=16` are two compiled kernels.
- Runtime `n_elements` does not recompile the tile type; it changes only the mask and the grid, $P=\lceil N/B\rceil$.

### Store rule
- The tile you store must have the pointer's element type.
- Cast on the last line: `tl.store(out_ptr + offs, tl.cast(y, tl.float16), mask=mask)`.

## Worked numerical example

- **Rule.** Set `BLOCK = 8`; the notebook asserts every value below without printing it.

| expression | shape | dtype | value |
|---|---|---|---|
| `tl.arange(0, 8)` | `(8,)` | `int32` | `0..7` |
| `tl.full((8,), 10.0, tl.float32)` | `(8,)` | `float32` | all `10` |
| `tl.zeros((8,), tl.float32)` | `(8,)` | `float32` | all `0` |
| `tl.full((8,), 1.0, tl.float32)` | `(8,)` | `float32` | all `1` |
| `arange + full + zeros + ones` | `(8,)` | `float32` | `11..18` |
| `tl.full((2, 4), 3.0, tl.float32)` | `(2,4)` | `float32` | all `3` |
| `tl.full(x.shape, -1.0, x.dtype)` | `(8,)` | `float32` | all `-1` |
| `tl.zeros_like(x)` | `(8,)` | `float32` | all `0` |
| `tl.broadcast_to(col, (4,4))` | `(4,4)` | `float32` | rows `5,6,7,8` |
| `tl.cast(y, tl.float16)` | `(8,)` | `float16` | rounded |

- **Vector.** $v=(0,1,\ldots,7)+(10,\ldots,10)+(0,\ldots,0)+(1,\ldots,1)=(11,12,\ldots,18)$.
- **Constant matrices.** $F$ is `(2,4)` with rows `(3,3,3,3)` and `(3,3,3,3)`; $Z$ is `(2,4)` with two zero rows.
- **Host broadcast.** $C=(10,20,30)^T$ has shape `(3,1)`; $B=\operatorname{broadcast\_to}(C,(3,4))$ has rows `(10,10,10,10)`, `(20,20,20,20)`, and `(30,30,30,30)`.
- **Strides.** `stride(C) = (1,1)`, `stride(B) = (1,0)`, and `B.data_ptr() == C.data_ptr()`; the four columns reuse one storage element and no copy is made.
- **Device broadcast.** Triton axes are powers of two, so the kernel uses `(4,1) -> (4,4)`; $C_d=(5,6,7,8)^T$ and $B_d$ has rows `(5,5,5,5)`, `(6,6,6,6)`, `(7,7,7,7)`, and `(8,8,8,8)`.
- **Cast.** $h=\operatorname{fp16}(0.1\,(0,\ldots,7))=(0, 0.0999755859375, 0.199951171875, 0.300048828125, 0.39990234375, 0.5, 0.60009765625, 0.7001953125)$; casting back reproduces those eight values exactly.

## Chapter map

| § | Constructor | Formula/behavior | Shape rule | Kernel/demo |
|---|---|---|---|---|
| **3.1** | `tl.arange()` | $[s,s+1,\ldots,e-1]$ | `(e-s,)` | offsets + tail mask |
| **3.2** | `tl.full()` | $X_{ij}=c$ | exactly `shape` | constant tile |
| **3.3** | `tl.zeros()` | $X_{ij}=0$ | exactly `shape` | accumulator seed |
| **3.4** | `tl.ones()` | $X_{ij}=1$ | exactly `shape` | `tl.full(..., 1.0, ...)` |
| **3.5** | `tl.full_like()` | $X_{ij}=c$ | same as `x` | sentinel like a loaded tile |
| **3.6** | `tl.zeros_like()` | $X_{ij}=0$ | same as `x` | zero branch |
| **3.7** | `tl.broadcast_to()` | $Y_{ij}=b_j$ | target; size-1 expands | column-bias add |
| **3.8** | `tl.cast()` | $y=\operatorname{cast}_d(x)$ | unchanged | masked affine |
| **3.9** | Triton dtypes | widths and promotion | unchanged | dtype table |
| **3.10** | compile-time dimensions | $T=\prod_k B_k$ | all axes static | `BLOCK` grid demo |
| **3.11** | scalar vs vector | $Y_{ij}=X_{ij}s$ | unchanged | `pid * BLOCK + arange` |
| **3.12** | static ranges | unrolled $K$ steps | unchanged | polynomial kernel |
| **★** | causal mask | $[j\le i]$ | `(T,T)` | attention mask |

## The kernels

- **Method 02.** Constructs tiles inside device code and broadcasts a loaded column with `tl.broadcast_to(col[:, None], (BLOCK_M, BLOCK_N))`.
- **Method 03.** Fuses construction, selection, and a cast in the masked affine kernel and the bias epilogue.
- **Capstone.** Builds a causal mask from two `arange` vectors and `tl.where`; for $T=5$ it is lower triangular.

## Dtypes and compile-time control

| dtype | bytes | role | key limit |
|---|---:|---|---|
| `tl.int32` | 4 | lane indices | $2^{31}-1$ |
| `tl.int64` | 8 | very large offsets | twice the register width |
| `tl.float16` | 2 | storage | max `65504`, 10 mantissa bits |
| `tl.bfloat16` | 2 | storage, wide range | 7 explicit mantissa bits |
| `tl.float32` | 4 | accumulation | 23 mantissa bits |
| `tl.int1` | 1 bit (1 byte stored) | predicates | not numeric |

- **Static range.** `tl.static_range(0, K)` unrolls a fixed loop at trace time.
- **Example.** The polynomial kernel computes `1 + x + x^2 + x^3`; for `x = (2,-1,0.5)` the output is `(15,0,1.875)`.
- **Failure.** A runtime bound is not a static range.

## Pitfalls

| Pitfall | What happens | Fix |
|---|---|---|
| shape/dtype mismatch | fp16 pointer with fp32 tile fails or stores garbage | cast at the store |
| runtime `tl.arange` bound | tracing fails | use `BLOCK` and mask |
| non-power-of-two dimension | trace error | pad and mask |
| `broadcast_to` treated as a copy | host view is read-only; storage is shared | write a fresh tile, or `.contiguous()` |
| fp16 overflow | above `65504` becomes `inf` | accumulate in fp32 |
| implicit promotion | int/float and width changes surprise you | cast explicitly |
| changing `BLOCK` | new specialization and cache entry | benchmark the shipped value |
| missing tail mask | out-of-bounds load/store | mask both doors |

## How to run

- **Notebook.** `cd part1_foundations/ch03_vector_construction` then `jupyter notebook 03_complete.ipynb`.
- **Path.** The notebook walks up to `00_Basic`, imports `ttools.py`, and prints the active device.
- **CPU.** Every numerical cell, the Method 01 reference, the broadcast-view test, and the autograd check run; each Triton test prints `SKIP ...`.
- **GPU.** Every kernel is compared with a PyTorch reference.
- **Interpreter.** Run from `00_Basic` with the command below.

```bash
cd 00_Basic
TRITON_INTERPRET=1 PYTHONPATH=$PWD python tools/interpreter_smoke.py
```

## Study checklist
- [ ] Write the shape and dtype of all six constructors before running anything.
- [ ] Explain why `tl.zeros((n,), ...)` fails but `tl.zeros((BLOCK,), ...)` works.
- [ ] State the stride of `torch.broadcast_to(col, (3, 4))` when `col` is `(3, 1)`.
- [ ] Prove that `.contiguous()` copies and that `broadcast_to` does not.
- [ ] Predict the fp16 result for `0.1`, `0.3`, and `70000.0`.
- [ ] Explain why the backward of a broadcast is a sum over the expanded axis.
- [ ] Change `BLOCK` from 8 to 16 and say what recompiles and what does not.
- [ ] Run the capstone for `T=13` and check the masked tail.

**Where this leads.** Chapter 4 uses these constructors to build pointers: `base + offset` becomes a memory address, and `BLOCK` becomes the tile that a load or store covers.

## Exercises
1. Build `tl.arange(0, 16)`, print shape and dtype, then cast to `int64` and `float32`.
2. Build a `(4, 8)` tile with `tl.full`, `tl.zeros`, and the portable ones form.
3. Build an `-inf` padding tile with `tl.full(x.shape, -float("inf"), x.dtype)`.
4. Test the host broadcasts `(3,1)->(3,4)`, `(1,4)->(3,4)`, and `(3,)->(4,)`.
5. Prove no copy by comparing `data_ptr()` before and after `.contiguous()`.
6. Cast `65504.0`, `65520.0`, and `70000.0` to fp16 and identify the `inf`.
7. Classify `pid`, `BLOCK`, `offs`, and the literal `0` as compile-time or runtime.
8. Run the construct kernel with `BLOCK=8` and `BLOCK=16` for `n=13`.
9. Replace the polynomial with a five-tap `tl.static_range(0, 5)` recurrence.
10. Build a causal mask for `T=17`, `BLOCK=32`, then an anti-diagonal mask.

- **Hints.** See [`../../docs/SOLUTIONS.md`](../../docs/SOLUTIONS.md).
