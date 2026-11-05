# Chapter 4 — Pointer Fundamentals
Part 2: Pointer arithmetic and memory operations.

> **Thread.** A logical index becomes an element offset through `shape` and `stride`; the offset becomes a byte address through `itemsize`; the byte address becomes a typed pointer in a Triton kernel. The kernel receives only that pointer, so layout is the host's responsibility.

[Open the chapter notebook](04_complete.ipynb)

[Index](../../00_index.ipynb) · [← Previous](../../part1_foundations/ch03_vector_construction/03_complete.ipynb) · [Next →](../ch05_load_store/05_complete.ipynb)

**Estimated time:** 60–75 min · **Difficulty:** Intermediate
## Thread
- A logical index maps to an element offset through `shape` and `stride`.
- An element offset maps to a byte address through `itemsize`.
- A byte address maps to a typed pointer expression in a Triton kernel.
- The kernel sees only that pointer; every layout decision stays on the host.
## Learning objectives
- Read `x_ptr` as a **typed base pointer** and write `addr = base + offset`.
- Convert an element offset to a byte address with `itemsize` (fp32 4, fp16 2, int8 1, int64 8).
- Derive row-major and column-major offsets from `shape` and `stride`.
- Read `shape`, `stride()`, `storage_offset()`, and `data_ptr()` and map an index to storage.
- Write 1-D and 2-D Triton kernels with explicit offsets, including an `int64` path for `numel > 2**31`.
- Pass strides as runtime arguments and run a stride-aware kernel on a transposed and sliced tensor.
## Prerequisites
- Chapter 3: `tl.arange`, `tl.constexpr`, tiles, and the idea of a program owning a block.
- Basic linear indexing, zero-based counting, and `torch.Tensor.shape`, `dtype`, `stride()`, `data_ptr()`.
## Connections
- **Backward:** Chapter 3 builds the tiles; this chapter supplies the addresses those tiles read and write.
- **Forward:** Chapter 5 — Loading and Storing Data turns these addresses into `tl.load`, `tl.store`, masks, and boundary checks.
- **Within Part I:** §4.1–§4.5 build the address chain; §4.6–§4.9 turn layouts into strides and bytes; §4.10–§4.12 apply the result to arbitrary views.
- **Within the methods:** Method 01 computes the map in Python, Method 02 moves it into 1-D/2-D Triton kernels, and Method 03 passes explicit strides for non-contiguous inputs.
- **Capstone:** the panel-copy kernel combines the source strides with a contiguous destination.
## The ladder
| Method | What it does | Why it is there |
|---|---|---|
| **01** | Python loop computes `storage_offset + sum(i*stride)` and gathers | makes the address map the specification |
| **02** | Triton adds `pid*BLOCK + arange` to a base pointer | the workhorse 1-D and 2-D kernels |
| **03** | Triton receives `stride_m`, `stride_n`, and output strides | reads transposed and sliced views without a copy |
| **★** | Panel-copy kernel reads a strided source into a contiguous buffer | the applied shape of the same formula |
## Part I — the address chain
- **Connection.** Chapter 3 gives the tiles; this part gives each tile a storage address.
- **Thread.** Logical index → element offset → byte address → pointer expression.
- **Rule.** The element offset is what a Triton kernel adds to a pointer.
- **Where.** The byte address is what `data_ptr()` and external tools report; the stride converts an index to an offset.
- **Bridge.** Keep those three objects separate and the twelve sections collapse into one rule.
### Address rule
- For a $d$-dimensional index, $\text{off}=\sum_{k=0}^{d-1} i_k s_k$ and $\text{addr}=\text{base}+\text{off}$.
- The map is affine: a base plus a linear combination of indices, never a multiplication of pointers.
### Stride rule
- PyTorch reports `stride(k)` and `storage_offset()` in **elements**, not bytes.
- A `float32` tensor with stride `6` has a byte stride of `24`.
- NumPy's `ndarray.strides`, by contrast, are byte quantities.
### Itemsize rule
| dtype | itemsize | offset 11 in bytes |
|---|---:|---:|
| `float64` / `int64` | 8 | 88 |
| `float32` / `int32` | 4 | 44 |
| `float16` / `bfloat16` | 2 | 22 |
| `int8` / `uint8` | 1 | 11 |

- Itemsize is a property of the dtype.
- Reinterpreting a pointer as another dtype also changes the meaning of every offset.
### Contiguous rule
- A tensor is contiguous (row-major, C order) when $s_k=\prod_{j\gt k}n_j$.
- The offset is then $\sum_k i_k\prod_{j\gt k}n_j$.
- For `(M,N)` this is stride `(N,1)`; for `(B,M,N)` it is `(M*N,N,1)`.
- Only a contiguous tensor may be addressed by one flat index.
### Non-contiguous rule
- A non-contiguous view has arbitrary non-negative strides and a `storage_offset`.
- Read them and pass them to the kernel: $\text{off}(i,j)=i\,\text{stride}(0)+j\,\text{stride}(1)$.
- `data_ptr()` already includes `storage_offset`, so kernel offset `0` is the view's first logical element.
- Do not add `storage_offset` again in the kernel.
### Kernel rule
- The host owns shape and stride; the kernel owns the affine map.
- If the host materializes `.contiguous()`, the kernel may use a flat offset.
- If not, the kernel must receive strides.
- `tl.load` performs no hidden contiguity check and no hidden broadcast.
## Chapter map
| § | Topic | Key formula | Shape/stride rule | Kernel/demo |
|---|---|---|---|---|
| **4.1** | Pointer arguments | $\text{addr}=\text{base}+\text{off}$ | offset relative to the view's `data_ptr` | 1-D `+1` kernel |
| **4.2** | Base pointer | `base = x_ptr` | `data_ptr` includes `storage_offset` | worked example |
| **4.3** | Pointer plus offset | `x_ptr + offs` | `offs` has the tile shape | 1-D and 2-D kernels |
| **4.4** | Element offset vs byte offset | $\text{addr}_\text{byte}=\text{base}_\text{byte}+b\cdot\text{off}$ | `b = itemsize(dtype)` | offset table |
| **4.5** | Pointer arithmetic | $\text{base}+\sum_k i_k s_k$ | one affine map per element | 1-D kernel |
| **4.6** | Contiguous memory | $s_k=\prod_{j\gt k}n_j$ | `is_contiguous()` is true | 2-D row-major kernel |
| **4.7** | Row-major memory | $\text{off}(i,j)=iN+j$ | stride `(N,1)` | 2-D kernel |
| **4.8** | Column-major memory | $\text{off}(i,j)=i+jM$ | stride `(1,M)` | transpose demo |
| **4.9** | Element size and dtype | $\text{addr}_\text{byte}=b\cdot\text{off}$ | fp32 4 B, fp16 2 B, int8 1 B | itemsize table |
| **4.10** | Tensor strides | $\text{off}=\sum_k i_k s_k$ | PyTorch strides are in elements | view table |
| **4.11** | Multidimensional address calculation | $\text{off}(b,i,j)=b s_0+i s_1+j s_2$ | one stride per axis | 3-D numeric check |
| **4.12** | Pointer arithmetic for non-contiguous tensors | $\text{off}=i s_0+j s_1$ | pass `stride()` at runtime | stride-aware kernel |
| **★** | Capstone | $\text{src}=i s_0+j s_1$ | strided source, contiguous destination | panel copy |
## Worked numerical example
- **Connection.** Applies §4.5–§4.12 to one matrix; the notebook asserts every value below.
- **Matrix.** Take

$$
A=\begin{bmatrix}0&1&2&3\\4&5&6&7\\8&9&10&11\end{bmatrix}
\in\mathbb R^{3\times4},
\qquad
\text{shape}=(3,4),\quad
\text{stride}=(4,1),\quad
\text{storage\_offset}=0,\quad
\text{itemsize}=4.
$$

- **Address of `(2,3)`.** $\text{off}(2,3)=2\cdot4+3\cdot1=11$, so the byte offset is $11\cdot4=44$, and `A[2,3] = A.reshape(-1)[11] = 11`.
- **Transpose.** $A^\top$ has shape `(4,3)` and stride `(1,4)`, so $\text{off}(3,2)=3\cdot1+2\cdot4=11$ and `A.t()[3,2] = A[2,3]`.
- **Slice.** $S=A[:,1{:}3]$ has shape `(3,2)`, stride `(4,1)`, and `storage_offset = 1`; the view-relative offset of `S[2,1]` is $2\cdot4+1\cdot1=9$, the storage index is $1+9=10$, and `S[2,1] = A.reshape(-1)[10] = 10`.
- **Copy.** `A.t().contiguous()` has shape `(4,3)` and stride `(3,1)`: values are unchanged, `data_ptr` changes, and storage order is rearranged.

| view | shape | stride | storage_offset | an index | offset | value |
|---|---|---|---|---|---|---|
| `A` | `(3,4)` | `(4,1)` | `0` | `(2,3)` | `11` | `11` |
| `A.t()` | `(4,3)` | `(1,4)` | `0` | `(3,2)` | `11` | `11` |
| `A[:, 1:3]` | `(3,2)` | `(4,1)` | `1` | `(2,1)` | `9` (storage `10`) | `10` |
| `A.t().contiguous()` | `(4,3)` | `(3,1)` | `0` | `(3,2)` | `3*3+2*1 = 11` | `11` |
## Method 01 — host-side address arithmetic
- **Connection.** Implements §4.5–§4.12 in Python.
- **Rule.** For each flat index, reconstruct the multi-index, compute the storage index, and gather.
- **Formula.** $\text{storage index}=\text{storage\_offset}+\sum_k i_k\,\text{stride}(k)$.
- **Where.** `as_strided` provides a flat view of storage; the gather is driven by the offset formula.
- **Triton.** On CPU this is acceptable for small tensors; on a GPU it is one Python iteration per element, which motivates Method 02.
## Method 02 — pointer + element offsets
- **Connection.** Implements Part II with a tile instead of a Python loop.
- **Rule.** Each program owns `BLOCK` consecutive elements and masks the tail.
- **Formula.** $\text{offs}=\text{pid}\cdot B+\{0,\ldots,B-1\}$, $\text{mask}=\text{offs}\lt N$, and $P=\lceil N/B\rceil$.
- **Example.** The 2-D kernel uses $\text{offs}(i,j)=iN+j$.
- **Offset dtype.** `tl.arange` and `tl.program_id` are 32-bit; cast `offs` to `tl.int64` when `numel > 2**31`.
- **Where.** The 1-D and 2-D kernels are the workhorses for contiguous inputs.
- **Bridge.** Method 03 removes the contiguity assumption.
## Method 03 — stride-aware kernels
- **Connection.** Implements Part IV and §4.12 by receiving explicit strides.
- **Rule.** Pass four runtime strides: two for the source view and two for the output.
- **Formula.** $\text{src}(i,j)=i s^{(x)}_0+j s^{(x)}_1$ and $\text{dst}(i,j)=i s^{(o)}_0+j s^{(o)}_1$.
- **Where.** The test covers a contiguous matrix, a transpose, a sliced column band, and a materialized copy.
- **Triton.** The transpose has stride `(1,4)` and the slice has stride `(4,1)` with `storage_offset = 1`; both run without a copy.
- **Failure.** `data_ptr` already includes `storage_offset`; do not add it again.
## Applied example — column panel
- **Connection.** Combines Method 03 with a fixed output stride.
- **Rule.** Extract $P=A[:,c_0{:}c_1]$ and write a contiguous destination.
- **Formula.** $\text{src}(i,j)=\text{base}+iN+(c_0+j)$ and $\text{dst}(i,j)=iW+j$, where $W=c_1-c_0$.
- **Example.** For `A = torch.arange(20).reshape(4,5)`, `A[:,1:4]` has shape `(4,3)`, stride `(5,1)`, and `storage_offset = 1`; `A.t()` has shape `(5,4)` and stride `(1,5)`.
- **Where.** The panel is non-contiguous whenever its width is less than `N`.
- **Triton.** The kernel reads with the source strides and writes with `N` as the row stride.
## Byte traffic
- **Rule.** For an `(M,N)` tensor with `b` bytes per element, the stride-aware path moves $2MNb$ bytes and the copy path moves $4MNb$.

| Path | Traffic | For `(1024,2048)` fp32 |
|---|---:|---:|
| stride-aware kernel | $2MNb$ | 16.78 MB |
| `.contiguous()` + kernel | $4MNb$ | 33.55 MB |

- **Where.** The stride-aware path moves half the bytes because it skips the materializing copy.
- **Failure.** It can still lose when the strided access is badly uncoalesced; the notebook states that trade-off rather than hiding it.
## Pitfalls
| Pitfall | What happens | Fix |
|---|---|---|
| Element vs byte offset | a byte offset added to `x_ptr` skips elements | keep Triton offsets in elements |
| Wrong stride | a transposed tensor is read as row-major | read and pass `x.stride()` |
| Transposed view | values come back transposed with no fault | test a rectangular transpose |
| `int32` overflow | offsets wrap for `numel > 2**31` | cast `offs` to `tl.int64` |
| Assuming contiguous | slicing and `expand` change the map | check `is_contiguous()` |
| Negative strides | a reversed view can walk before the base | materialize with `contiguous()` |
| Forgetting `storage_offset` | an absolute index is off by the view offset | use `data_ptr()` as the base |
| NumPy strides in bytes | offsets are `itemsize` times too large | divide by `itemsize` |
| Zero-stride `expand` | every row touches the same element | materialize or guard the store |
## How to run
```bash
cd part2_memory/ch04_pointers
jupyter notebook 04_complete.ipynb
```

- The notebook imports `ttools.py` by walking up to `00_Basic`.
- On CPU the numerical cells and all five CPU tests run; each GPU test reports `SKIP ...`.
- On a GPU every kernel is compared with PyTorch.

```bash
cd 00_Basic
TRITON_INTERPRET=1 PYTHONPATH=$PWD python tools/interpreter_smoke.py
```

- The interpreter forces `ttools.HAS_GPU = True`.
- The stride-aware and panel kernels then run on CPU tensors with the same code path as on a GPU.
## Study checklist
- [ ] Compute the element and byte address of `(2,3)` for a `(3,4)` fp32 matrix.
- [ ] Do the same for the transpose and explain why the offset is unchanged.
- [ ] Write the row-major and column-major offset tables and compare them.
- [ ] Read `shape`, `stride`, `storage_offset`, and `data_ptr` for `A.t()` and `A[:,1:3]`.
- [ ] Explain why kernel offset `0` is the view's first element, not the storage start.
- [ ] State when `tl.arange` offsets overflow and how to fix it.
- [ ] Run the stride-aware kernel on a transpose and a slice; compare with `.contiguous()`.
- [ ] Derive the 16.78 MB vs 33.55 MB traffic numbers for `(1024,2048)` fp32.
- [ ] Do exercises 2, 4, and 8: they expose byte/element, stride, and view bugs.
## Exercises
- **Connection.** These revisit the address chain and the Method 01–03 kernels.

The notebook ends with ten exercises: hand-computed addresses, a byte-address helper, column-major storage order, a stride table, a 1-D kernel variant, the int64 path, a 2-D kernel variant, stride-aware runs, byte-traffic derivation, and a 3-D panel generalization. Hints are in [`../../docs/SOLUTIONS.md`](../../docs/SOLUTIONS.md).

**Where this leads.** Chapter 5 — Loading and Storing Data turns these addresses into masked loads, stores, and boundary checks.
