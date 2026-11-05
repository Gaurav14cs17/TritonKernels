# Chapter 5 — Loading and Storing Data

Part 2: Pointer arithmetic and memory operations.

**Thesis.**
- A Triton program reaches global memory through exactly two doors: `tl.load` and `tl.store`.
- A tile has a compile-time shape (`BLOCK`); a mask has the same shape.
- `other` is a scalar chosen by the operation that consumes the tile.
- Get offsets, mask, value, and `other` right, and every later kernel is a variation on the same skeleton.

[Open the chapter notebook](05_complete.ipynb)

[Index](../../00_index.ipynb) · [← Previous](../ch04_pointers/04_complete.ipynb) · [Next →](../ch06_indexing/06_complete.ipynb)

**Estimated time:** 70–90 min · **Difficulty:** Intermediate

## Goal and objectives

- Write kernels that read and write global memory safely; derive the tile table, mask both doors, and choose the neutral `other`.
- Write `tl.load(ptr + offs, mask=..., other=...)` and `tl.store(ptr + offs, value, mask=...)`.
- Compute $P=\lceil N/\text{BLOCK}\rceil$ and the exact 1-D and 2-D masks; choose `other` for sum, product, max, min, log, and a denominator.
- Guard the load **and** the store, and recognise RMW, dtype width, and coalesced access.

**Prerequisites:** Chapter 4 (pointers and strides), Chapter 3 (`tl.arange`, tile shapes), Chapter 2 (`tl.program_id(0)` and launch grids).

## Thread

- Offsets first: `pid * BLOCK + tl.arange(0, BLOCK)` fixes the lane map.
- The mask `offs < N` decides which lanes exist and guards both memory doors.
- `other` gives non-existent lanes the identity of the next reduction.
- The same four objects — offsets, mask, value, `other` — recur in copy, add, RMW, and reduction kernels.
- The address pattern decides the cost: contiguous coalesces, strided multiplies transactions.

## Connections

- Chapter 4 → base pointer plus element offset; this chapter adds the read and write operations.
- §5.1–§5.6 → load, store, scalar/vector forms, and the mask.
- §5.7–§5.9 → the neutral `other` and the 1-D/2-D boundary.
- §5.10–§5.13 → dtypes, multiple pointers, and read-modify-write.
- §5.14 → address patterns; Chapter 6 → 2-D and 3-D index maps.

## The ladder

| Method | What it is | Where it is used |
|---|---|---|
| **01** | Eager PyTorch: pad to `P*BLOCK`, then `torch.where` | the reference implementation |
| **02** | Triton `tl.load` / `tl.store` with masks, scalar and vector tiles | copy, scalar scale, two-input add |
| **03** | Read-modify-write and fused masked reductions | in-place add, masked sum/max |

- **Rule.** Method 01 is the CPU specification; Method 02 is the workhorse; Method 03 adds aliasing and the neutral-element rule.

## Mathematics

- **Rule.** A load is a masked gather; a store is a masked scatter.
- **Formula.**

$$
x_i=\begin{cases}\operatorname{load}(p_i),&m_i=1\\ \text{other},&m_i=0\end{cases}
\qquad
\operatorname{store}(p_i,y_i,m_i)=\begin{cases}y_i,&m_i=1\\ \text{skip},&m_i=0.\end{cases}
$$

- **Rule.** The tile shape is compile-time: $\text{offs}_i=\text{pid}\cdot B+i$ and $P=\lceil N/B\rceil$.
- **Rule.** A 1-D mask is $m_i=[\text{offs}_i\lt N]$; a 2-D mask is $m_{ij}=[r_i\lt M]\,\&\,[c_j\lt N]$.
- **Rule.** Traffic is $\text{bytes}=N_{\text{active}}(b_{\text{load}}+b_{\text{store}})$.

## Rules

### Mask rule

- `mask = offs < N` guards the load and the store, and both use the **same** predicate.
- A masked store leaves the destination untouched; it does not write zero or `other`; a comparison may run on every lane, but only the memory access is suppressed.

### Other rule

- `other` is the identity of the operation that follows the load.
- It is not a clamp and not a default for missing data.
- It is a scalar, broadcast to the tile; it never changes the shape.

| Consumer | `other` | Reason |
|---|---|---|
| `tl.sum` | `0.0` | $x+0=x$ |
| product | `1.0` | $x\cdot1=x$ |
| `tl.max` | `-inf` | it can never win |
| `tl.min` | `+inf` | it can never win |
| `tl.log` then sum | `1.0` | $\log 1=0$ |
| denominator | `1.0` | never $0/0$ |
| copy with masked store | `0.0` | the value is discarded |

### Dtype rule

- The pointer's element type sets transaction width and interpretation.

| dtype | bytes/element | read + write for $N=10$ |
|---|---|---|
| `float64` | 8 | 160 B |
| `float32` | 4 | 80 B |
| `float16` / `bfloat16` | 2 | 40 B |
| `int8` | 1 | 20 B |

- Cast to `float32` before a long accumulation; cast explicitly before a narrow store.
- `other=-inf` is only representable in a floating dtype.

### Boundary rule

- The mask is the **only** bounds check inside a kernel.
- A true lane past the allocation can read garbage, fault, or corrupt a neighbour buffer; host-side `IndexError` does not exist in device code.

### Shape rule

- Offsets, mask, and value share `(BLOCK,)` or `(BLOCK_M, BLOCK_N)`.
- A scalar load has shape `()` and broadcasts; different input lengths need different masks.

### Read-modify-write rule

- Load and store the same address with the same offsets and mask.
- Each address needs one owner; cross-program accumulation needs `tl.atomic_add` (Chapter 21).

### Coalescing rule

- Consecutive lanes should touch consecutive addresses: `offs = r*N + c`.
- Column-major indexing (`offs = r + c*M`) makes each lane jump by `M` elements.

## Worked numerical example — $N=10$, `BLOCK=4`

- **Connection.** Applies §5.7 (`other`) and §5.8 (tail mask) to the chapter spine.
- **Example.** Take $x=(-1,-2,-3,-4,-5,-6,-7,-8,-9,-10)$; then $P=\lceil 10/4\rceil=3$, with 12 declared lanes and 10 real elements.

| pid | offsets | mask | lanes | load, `other=0.0` | load, `other=-inf` |
|---|---|---|---|---|---|
| 0 | $0,1,2,3$ | $\mathrm{T,T,T,T}$ | 4 | $(-1,-2,-3,-4)$ | $(-1,-2,-3,-4)$ |
| 1 | $4,5,6,7$ | $\mathrm{T,T,T,T}$ | 4 | $(-5,-6,-7,-8)$ | $(-5,-6,-7,-8)$ |
| 2 | $8,9,10,11$ | $\mathrm{T,T,F,F}$ | 2 | $(-9,-10,0,0)$ | $(-9,-10,-\infty,-\infty)$ |

- **Formula.**

$$
S_0=(-10)+(-26)+(-19)=-55,\quad S_{-\infty}=-\infty,\quad
M_{-\infty}=-1,\quad M_0=0.
$$

- **Rule.** `other=0.0` is right for the sum and wrong for the max; `other=-inf` is right for the max and destroys the sum.
- **Rule.** The notebook asserts every hand-computed row.

## Chapter map

| § | Topic | Key formula/rule | Shape/mask rule | Kernel/demo |
|---|---|---|---|---|
| **5.1** | `tl.load()` | $x_i=\operatorname{load}(p_i)$ if $m_i$, else `other` | `shape(x) = shape(offs)` | masked copy |
| **5.2** | `tl.store()` | write $y_i$ only if $m_i$ | `shape(value) = shape(offs)` | masked copy |
| **5.3** | Scalar load | $v=\operatorname{load}(p)$ | result shape `()` | scalar scale |
| **5.4** | Vector load | $x_i=\operatorname{load}(p_i)$ | `(BLOCK,)` or `(BM,BN)` | masked copy |
| **5.5** | Masked load | `other` fills false lanes | mask shape = offset shape | copy with `other=0.0` |
| **5.6** | Masked store | false lanes are not written | store mask = load mask | sentinel-tail test |
| **5.7** | The `other` parameter | identity of the next reduction | scalar, broadcast | neutral-element table |
| **5.8** | Boundary checking | $m_i=[\text{offs}_i\lt N]$ | outer product for 2-D | 1-D and 2-D tail masks |
| **5.9** | Out-of-bounds prevention | mask both doors | same predicate on both | all three programs |
| **5.10** | Load and store dtypes | bytes $=N_{\text{active}}(b_{\text{load}}+b_{\text{store}})$ | value dtype = pointer dtype | dtype traffic table |
| **5.11** | Multiple input pointers | $y_i=a_i x_i+b_i$ | one mask per input length | two-input add |
| **5.12** | Multiple output pointers | $(y_i,s_i)=f(x_i)$ | one store per output | two-output host demo |
| **5.13** | Read-modify-write | $y_i\leftarrow y_i+x_i$ | same offsets and mask | in-place add |
| **5.14** | Memory access patterns | coalesced if $\Delta p=1$ | last axis contiguous | stride experiment |
| **★** | Worked example | $N=10$, `BLOCK=4`, $P=3$ | tail mask `T T F F` | `test_worked_example` |

## Pitfalls

| Pitfall | What happens | Fix |
|---|---|---|
| mask only the store | the load still reads out of bounds | mask both doors |
| `other=0.0` for a max | a masked lane wins with zero | `other=-inf` |
| `other=0.0` for a log/denominator | `-inf` or `inf`/`nan` | `other=1.0` |
| 1-D mask for a 2-D tile | shape error or wrong lanes | outer-product mask |
| `offs <= N` | one lane past the end is active | `offs < N` |
| `other=-inf` through an integer pointer | not representable | use a float dtype |
| RMW from two programs | last write wins | one owner, or atomics |
| non-contiguous tensor without strides | wrong elements are read | pass strides or `.contiguous()` |
| wide value through a narrow pointer | silent truncation | cast before the store |

## How to run

- Open the notebook from this directory:

```bash
cd part2_memory/ch05_load_store
jupyter notebook 05_complete.ipynb
```

- The notebook imports `ttools.py` by walking up to `00_Basic`.
- On CPU every numerical, reference, and backward cell runs; Triton tests print `SKIP ... no GPU`.
- On GPU every kernel is compared with a PyTorch reference.
- GPU-free interpreter run: prepend `import ttools; ttools.HAS_GPU = True`, then use `TRITON_INTERPRET=1` and `PYTHONPATH=00_Basic`.

## Study checklist

- [ ] Write the $N=10$, `BLOCK=4` tile table, including the mask, from memory.
- [ ] Compute $P$, the declared lanes, and the out-of-bounds lane count.
- [ ] State the neutral `other` for sum, product, max, min, log, and a denominator.
- [ ] Explain why a masked store leaves the old value, and why masking only the store still risks an out-of-bounds load.
- [ ] Write the 2-D mask for the corner tile of a `(3,5)` matrix with a `(2,4)` block.
- [ ] Derive the gradient of a masked scale and say what a masked-out lane gets.
- [ ] Explain why a column-major access is slower with the same element count.
- [ ] Do exercises 2, 4, and 10: they expose the `other`, mask, and race bugs.

## Exercises

- **Mask table.** $N=13$, `BLOCK=4`; find the program with two active lanes.
- **Wrong neutral.** Use `other=0.0` for max and find the smallest all-negative input that exposes it.
- **Store-only mask.** Delete the load mask in `m5_copy_kernel`; describe what the third program reads.
- **2-D boundary.** Tile-copy `(5,7)` with `BLOCK_M=2`, `BLOCK_N=4`; count the corner tile.
- **Sentinel audit.** Assert that exactly `P*BLOCK-N` lanes keep `fill=-7.0`.
- **Dtype traffic.** Compute the bytes for $N=2^{20}$ in `float32`, `float16`, and `int8`.
- **Scalar reuse.** Replace the scalar load with a per-lane load of the same address; count the extra transactions.
- **Two outputs.** Add `a*b` to the add kernel and count the traffic.
- **RMW race.** Show an order-dependent update, then rewrite it with `tl.atomic_add`.
- **Column-major copy.** Extend `strided_copy` to `(7,5)` and compare both layouts.

## Where this leads

- Chapter 6 turns the 1-D offset map into 2-D and 3-D indexing, including tiled and strided addresses.
- Continue with [Chapter 6 — Indexing and Offset Calculations](../ch06_indexing/06_complete.ipynb).
