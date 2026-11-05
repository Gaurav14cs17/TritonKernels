# Chapter 1 — Introduction to Triton

Part 1: Foundations — what Triton is, how programs run, and how vectors are built.

- **Goal.** A Triton launch is a grid of independent programs, and each program owns one tile. Once you can draw `pid -> offs -> mask`, every kernel in the book is the same skeleton with a different expression or grid.
- **Thread.** A launch is a grid; a program owns a tile; the tile is `pid*BLOCK + arange(BLOCK)`; the mask `offs < N` guards the tail; the host chooses shapes and grid.
- **Ladder.** Concept → numerical demo → micro-kernel → applied use.

[Open the chapter notebook](01_complete.ipynb) · [Index](../../00_index.ipynb) ·
[Next →](../ch02_program_execution/02_complete.ipynb)

**Estimated time:** 45–60 min · **Difficulty:** Beginner

## Learning objectives

- Explain what Triton is and which parts of a GPU program the compiler owns.
- Place allocation, grid choice, JIT specialization, and tile math on the correct side of the host/device boundary.
- Compute $P=\lceil N/B\rceil$, `pid*BLOCK + arange(BLOCK)`, and `offs < N`.
- Classify arguments as runtime or `tl.constexpr`, and predict recompilation.
- Write a minimal `@triton.jit` vector add with masked load and store.

**Prerequisites:** Python functions and loops; `torch.empty`, `torch.arange`,
`.shape`, `.numel`. No GPU or CUDA experience is required.

## Connections

- **1.1 → 1.2.** The tile model is defined, then contrasted with CUDA's thread model.
- **1.2 → 1.3.** CUDA and Triton are placed beside PyTorch, the layer above both.
- **1.3 → 1.4.** A tile kernel launch is expanded into a grid of programs.
- **1.4 → 1.5.** The grid is split into host-side allocation and device-side tile math.
- **1.5 → 1.6.** The host's first launch is followed through JIT specialization and caching.
- **1.6 → 1.7.** The cache key is triggered by the concrete launch syntax.
- **1.7 → 1.8.** The call's arguments are separated into runtime and compile-time parts.
- **1.8 → 1.9.** `BLOCK` becomes the first detailed `tl.constexpr` example.
- **1.9 → 1.10.** Every rule is assembled into the first working kernel.
- **Part I → Part IV.** The scalar reference, tiled kernel, and fused epilogue follow in Methods 01–03.
- **Chapter 1 → Chapter 2.** The single `pid` axis becomes 1-D, 2-D, and 3-D grids.

## Mental model

### Host and device

```text
 HOST (ordinary Python)                    DEVICE (compiled kernel)
 allocate x, y, out        launch          program pid reads/writes
 choose grid P=ceil(N/B)  -------->        offsets = pid*B + arange(B)
 choose BLOCK, dtype                       mask = offsets < N
 inspect output           <--------        load -> compute -> store
```

- **Rule.** The host runs once per launch; the device body runs once per program, concurrently.
- **Rule.** The host owns **shapes**, the device owns **addresses**.

### One program, one tile

- **Rule.** Program $p$ owns $\text{offs}_p=pB+\{0,\ldots,B-1\}$, mask $\text{offs}_p\lt N$, and the grid is $P=\lceil N/B\rceil$.
- **Rule.** The same body runs for every `pid`; only `pid` changes.
- **Rule.** Programs have no ordering, so two programs must not write one address unless the write is atomic.

### Compile time and runtime

- **Rule.** The first launch parses the AST, specializes on pointer types, scalar types, and `constexpr` values, then compiles and caches one binary.
- **Rule.** A new `N` hits the cache; a new `BLOCK`, `ACT`, or dtype creates a second binary.

## Rules this chapter teaches

### Shape and tile rules

- A kernel operates on tiles, not scalars: `tl.arange(0, BLOCK)` is the index vector, `tl.load` returns a tile, arithmetic is lane by lane.
- A 1-D kernel takes equal-length input buffers and writes an output of that length; the kernel never receives `.shape`, so pass `n_elements`.
- For a binary op, `shape(x) == shape(y)`; broadcasting is a host (`expand` + `contiguous`) or explicit-indexing decision inside a kernel.
- `BLOCK` is the tile length, is a `tl.constexpr`, and must be a power of two for `tl.arange`; use `triton.next_power_of_2(n)` for arbitrary `n`.

### Grid and mask rules

- The first bracket is a tuple: `kernel[(P,)]`, with $P=\text{triton.cdiv}(N,B)$.
- An undersized grid drops the tail; an oversized grid is safe only because the mask protects the extra lanes.
- Mask **both** the load and the store with `offs < n_elements`; the compute still runs on a masked lane, but the memory access is suppressed.
- Choose `other=` from the consuming operation: `0` for sum, `1` for product, `-inf` for max, `+inf` for min.

### Argument rules

| Argument | Kind | Changes the binary? |
|---|---|---|
| `x_ptr`, `y_ptr`, `out_ptr` | runtime pointer | no (same dtype) |
| `n_elements`, `a`, `b`, `scale` | runtime scalar | no |
| `BLOCK`, `MODE`, `ACT` | `tl.constexpr` | **yes** |
| dtype (fp32 vs fp16) | pointer type | **yes** |
| `num_warps`, `num_stages` | launch meta-parameter | **yes** |

- **Rule.** If a value appears in a shape, an `arange`, a `range`, or a Python `if` inside the kernel, it must be a `tl.constexpr` and it is part of the cache key.

## Chapter map

| § | Topic | Key idea/formula | Shape or argument rule | Kernel/demo |
|---|---|---|---|---|
| **1.1** | What is Triton? | tile-level Python DSL: $Y=f(\text{tile}(X))$ | tile length = `BLOCK` | tile lift of the spine |
| **1.2** | Why Triton instead of CUDA? | thread indexing → tile expressions | $P=\lceil N/B\rceil$ programs | thread vs program count |
| **1.3** | Triton vs PyTorch vs CUDA | three levels of control | PyTorch broadcasts; Triton does not | three-layer comparison |
| **1.4** | Triton execution model | grid → program → tile | `pid*BLOCK + arange(BLOCK)` | numeric tile map |
| **1.5** | Host code vs device code | host shapes, device addresses | device sees `x_ptr`, not `x` | simulated split |
| **1.6** | JIT compilation with `@triton.jit` | specialize + cache | pointers runtime; `BLOCK` constexpr | signature inspection |
| **1.7** | Kernel launch syntax | `kernel[grid](...)` | $P=\lceil N/B\rceil$ | grid arithmetic |
| **1.8** | Compile-time vs runtime arguments | runtime data vs binary key | `n_elements` runtime; `BLOCK` constexpr | argument classifier |
| **1.9** | `tl.constexpr` | compile-time value, one binary each | power-of-two `tl.arange` | two `BLOCK` values |
| **1.10** | First working Triton kernel | $y_i=x_i+y_i$ | same shape `(N,)`, masked tail | `add_kernel` |

## Worked numerical example

- **Spine.** $x=(1,2,3,4,5,6,7,8)$ and $y=10x=(10,20,30,40,50,60,70,80)$.
- **Grid.** $N=8$, $B=\texttt{BLOCK}=4$, $P=\lceil N/B\rceil=2$.

| pid | offs | mask | x tile | y tile | out tile |
|---:|---|---|---|---|---|
| 0 | 0,1,2,3 | T,T,T,T | 1,2,3,4 | 10,20,30,40 | 11,22,33,44 |
| 1 | 4,5,6,7 | T,T,T,T | 5,6,7,8 | 50,60,70,80 | 55,66,77,88 |

- **By hand.** $x_0+y_0=1+10=11$ and $x_7+y_7=8+80=88$.
- **Output.** $\text{out}=x+y=11x=(11,22,33,44,55,66,77,88)$.
- **Tail.** For $N=10$, $P=3$; program 2 has $\text{offs}_2=(8,9,10,11)$ and $\text{mask}_2=(T,T,F,F)$.
- **Check.** The notebook asserts the shapes, strides, grid, masks, and values rather than printing numerical tables.

## The canonical kernel

```python
@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    y = tl.load(y_ptr + offs, mask=mask, other=0.0)
    tl.store(out_ptr + offs, x + y, mask=mask)
```

- **Rule.** The host allocates `out`, chooses `grid = (triton.cdiv(N, BLOCK),)`, and launches.
- **Rule.** The notebook wrapper adds `.contiguous()` and a same-shape assert.

## What is saved and what is recompiled

- **Rule.** For `out = x + y`, $\partial L/\partial x=\partial L/\partial y=g$; the forward saves **nothing**.
- **Rule.** The backward reads `g` once and writes `dx` and `dy`.

| Change between launches | Recompile? | Why |
|---|---|---|
| pointer values | no | same pointer type |
| `n_elements`, `a`, `b` | no | runtime scalars |
| `BLOCK` | **yes** | constexpr used by `tl.arange` |
| `ACT`, `MODE` | **yes** | constexpr controls `if` |
| input dtype | **yes** | different pointer type |
| grid size | no | grid is not in the binary |
| `num_warps`, `num_stages` | **yes** | launch meta-parameters |

## Applied example: fused residual + ReLU

- **Rule.** Method 03 computes $z_i=a x_i+y_i+b$ and applies $\max(z_i,0)$ when `ACT=1`.
- **Rule.** One kernel keeps `z` in registers, applies the compile-time activation, and writes once.
- **Triton.** Since `ACT` is a `tl.constexpr`, the inactive branch is absent from each binary.

## Pitfalls

| Pitfall | What happens | Fix |
|---|---|---|
| grid uses floor division | the tail is never launched | `triton.cdiv(N, BLOCK)` |
| no store/load mask | out-of-bounds access | mask every access |
| `BLOCK` not a power of two | `tl.arange` rejects it | `next_power_of_2` |
| `x.shape` inside a kernel | the kernel sees only a pointer | pass `n_elements` |
| wrong `other=` | wrong reduction values | use the operation's identity |
| non-contiguous view | wrong addresses | `.contiguous()` or pass strides |

## How to run

```bash
cd part1_foundations/ch01_introduction
jupyter notebook 01_complete.ipynb
```

- **Rule.** The notebook imports `ttools.py` by walking up to `00_Basic`.
- **CPU.** Every numerical, autograd, and reference cell runs; Triton tests print `SKIP ...`.
- **GPU.** Every kernel is compared with PyTorch.
- **Interpreter.** From `00_Basic`, run `TRITON_INTERPRET=1 PYTHONPATH=$PWD python tools/interpreter_smoke.py`.

## Study checklist

- [ ] Write the spine tile map ($N=8$, `BLOCK=4`) and repeat it for $N=10$.
- [ ] Explain why the grid is `ceil(N/BLOCK)`, not `N // BLOCK`.
- [ ] Classify every argument of `add_kernel` as runtime or `constexpr`.
- [ ] State what recompiles when `N` changes and when `BLOCK` changes.

**Where this leads.** [Chapter 2](../ch02_program_execution/02_complete.ipynb) expands the grid to 1-D, 2-D, and 3-D and maps programs to matrix coordinates.

## Exercises

1. **Block size.** Rerun the spine with `BLOCK=8`; recompute $P$ and offsets.
2. **Edge case.** Run `add` on $N=1$ with `BLOCK=4`; list the masked lanes.
3. **New expression.** Change `x + y` to `x - y` and update the reference.
4. **Runtime scalar.** Add `scale`; is it runtime or `constexpr`?
5. **Compile-time flag.** Add `SQUARE: tl.constexpr` and count the binaries.
6. **Specialization.** List the four cache keys for `BLOCK` in `{4, 8}` and `ACT` in `{0, 1}`.

Hints are in [`../../docs/SOLUTIONS.md`](../../docs/SOLUTIONS.md).
