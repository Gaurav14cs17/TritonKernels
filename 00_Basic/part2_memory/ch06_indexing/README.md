# Chapter 6 — Indexing and Offset Calculations

Part 2: Pointer arithmetic and memory operations.

- **Thread.** Every address is a dot product between a coordinate and element strides, plus a base.
- **Thread.** Rank adds terms, and flatten/unflatten convert a coordinate to one integer and back.
- **Thread.** Strides replace hard-coded widths; tiling and masks only build the index vectors.
- **Thesis.** An index is not a coordinate and a coordinate is not an address.

**Connections**

- **Builds on:** Chapters 4 and 5 (pointers, element versus byte offsets, row-major layout, `tl.load`, `tl.store`, masks, `other=`).
- **Uses:** `tensor.stride()`, `tl.arange`, `tl.program_id`, and `tl.expand_dims`.
- **Leads to:** Chapter 7 (elementwise arithmetic) and the tiled kernels of later parts.

[Notebook](06_complete.ipynb) · [Helpers](../../ttools.py) · [Setup](../../docs/SETUP.md) · [Notation](../../docs/NOTATION.md) · [Glossary](../../docs/GLOSSARY.md) · [API](../../docs/API_INDEX.md)
[Formula](../../docs/FORMULA_INDEX.md) · [Diagrams](../../docs/DIAGRAMS.md) · [Troubleshooting](../../docs/TROUBLESHOOTING.md) · [Exercise hints](../../docs/SOLUTIONS.md)

[Index](../../00_index.ipynb) · [← Previous](../ch05_load_store/05_complete.ipynb) · [Next →](../../part3_math/ch07_arithmetic/07_complete.ipynb)

**Estimated time:** 60–75 min · **Difficulty:** Intermediate

## Goal

- Read the address expression inside a Triton kernel.
- Turn a logical index $(i,j,k)$, a shape, and a stride tuple into an element offset, and invert the map when a flat program id is all the kernel receives.

## Learning objectives

- Build 1-D, 2-D, and 3-D offset maps and state the shape of each offset tile.
- Convert between $(i,j)$ and a flat index in both directions.
- Distinguish element offsets from byte offsets and drive an address with `tensor.stride()`.
- Build tiled offsets with `rows[:, None]` and `cols[None, :]`, and state the `(BM, BN)` mask shape.
- Write a kernel whose address is recovered from a flat program id.
- Recognize swapped axes, off-by-one tails, wrong mask rank, int32 overflow, and non-contiguous strides.

## Prerequisites

- Chapter 4: pointers, element versus byte offsets, row-major layout.
- Chapter 5: `tl.load`, `tl.store`, masks, `other=`, and modular arithmetic.

## The ladder

| Method | Implementation | What it establishes |
|---|---|---|
| **01** | Python integers and loops | flatten/unflatten and row/column gather as definitions |
| **02** | Triton 2-D tiles, tile copy, row/column gather | `[:, None]`, `[None, :]`, rank-2 masks, 2-D grids |
| **03** | 3-D recovery, stride arguments, row scale, `tl.static_range` | non-contiguous inputs, arbitrary rank, applied kernels |

**Study tip.**

- Fix a $3\times4$ matrix of values $0,\ldots,11$.
- Write $f(i,j)=iN+j=4i+j$.
- Write the inverse $i=\lfloor f/4\rfloor$ and $j=f\bmod4$.
- Check $f(1,2)=6$ and $f=6\mapsto(1,2)$; every later formula is this one with another index or another stride.

## The rules this chapter teaches

### Offset rule

- **Rule.** For rank $r$ with element strides $s_0,\ldots,s_{r-1}$, the address adds one strided index per axis.
- **Formula.** $o(i_0,\ldots,i_{r-1})=\text{base}+\sum_{k=0}^{r-1} i_k s_k$.
- **Example / where.** $o_i=\text{pid}\cdot B+i$, $o(i,j)=i s_0+j s_1=iN+j$, and $o(i,j,k)=(iD_1+j)D_2+k$; strides are in elements, so a byte address multiplies by `tensor.element_size()`.
- **Failure.** Passing byte strides to an element-offset kernel scales every address.

### Flatten and unflatten rule

- **Rule.** Row-major flattening is a quotient and remainder by the row width.
- **Formula.** $f(i,j)=iN+j$, $i=f//N$, and $j=f\%N$.
- **Example / where.** For $N=4$, $f(1,2)=6$ and $6\mapsto(1,2)$; the width is $N$, the number of columns, not `numel()` and not the number of rows.
- **Failure.** Column-major storage uses $f=jM+i$; `view(-1)` raises on a non-contiguous tensor.

### Row and column rule

- **Rule.** A row is contiguous; a column is strided.
- **Formula.** $\text{row}_i=\text{base}+iN+\{0,\ldots,N-1\}$ and $\text{col}_j=\text{base}+j+\{0,N,2N,\ldots\}$.
- **Example / where.** In a $3\times4$ matrix, row 2 is $\{8,9,10,11\}$ and column 3 is $\{3,7,11\}$; a column gather writes a contiguous $(M,K)$ output, so its row stride is $K$, not $N$.
- **Failure.** A column read touches one element per cache line but is still correct.

### Stride rule

- **Rule.** Read the strides from the tensor and use them in the address.
- **Formula.** $o=i s_0+j s_1$.
- **Example / where.** A contiguous $3\times4$ matrix has $(s_0,s_1)=(4,1)$; its transpose has $(1,4)$; one kernel then serves contiguous, transposed, and sliced tensors.
- **Failure.** A zero stride is a broadcast; it is legal but not a copy.

### Tiling rule

- **Rule.** A 2-D grid assigns each program one tile of fixed shape.
- **Formula.** $\text{rows}=p_mB_M+\{0,\ldots,B_M-1\}$, $\text{cols}=p_nB_N+\{0,\ldots,B_N-1\}$, and $O=\text{rows}[:,None]\,s_0+\text{cols}[None,:]\,s_1$.
- **Example / where.** The grid is $(\lceil M/B_M\rceil,\lceil N/B_N\rceil)$; `rows[:, None]` has shape $(B_M,1)$ and `cols[None, :]` has shape $(1,B_N)$.
- **Failure.** `[None, :]` adds a dimension; it is not host right-aligned broadcasting.

### Mask rule

- **Rule.** The mask has the same shape as the offset tile and guards load and store.
- **Formula.** $\text{mask}(m,n)=[\,p_mB_M+m\lt M\,]\land[\,p_nB_N+n\lt N\,]$.
- **Example / where.** For $M=5$, $N=7$, $B_M=2$, $B_N=4$, pid $(2,1)$ has mask `[[1,1,1,0],[0,0,0,0]]`; edge tiles have dead lanes, and the mask keeps them from touching memory.
- **Failure.** A rank-1 mask broadcasts along one axis and lets the other run out of bounds.

### Arbitrary-rank rule

- **Rule.** Build one `tl.arange` per dimension, expand each to the output rank, multiply by its stride, and add.
- **Formula.** $o=\sum_{k=0}^{r-1} i_k s_k$.
- **Example / where.** Rank 3 with $(2,3,4)$ uses strides $(12,4,1)$ and produces offsets $0,\ldots,23$; `for k in tl.static_range(0, r)` unrolls the sum at compile time.
- **Failure.** The `static_range` bound must be a `tl.constexpr`.

## Chapter map

| § | Topic | Key formula | Shape/stride rule | Kernel/demo |
|---|---|---|---|---|
| **6.1** | 1-D offsets | $o_i=\text{pid}\cdot B+i$ | $B$ lanes per program; $o_i\lt N$ | 1-D map; tile copy |
| **6.2** | 2-D offsets | $o=i s_0+j s_1=iN+j$ | row-major strides $(N,1)$ | tile copy |
| **6.3** | 3-D offsets | $o=(iD_1+j)D_2+k$ | strides $(D_1D_2,D_2,1)$ | 3-D copy |
| **6.4** | Row indexing | $o_{\text{row}}=iN+j$ | fix $i$; $N$ lanes, stride 1 | row gather; row scale |
| **6.5** | Column indexing | $o_{\text{col}}=iN+j$ | fix $j$; $M$ lanes, stride $N$ | column gather |
| **6.6** | Flattening | $f=iN+j$ | C-order; contiguous `reshape` | host flatten |
| **6.7** | Recovering row/col | $i=f//N,\ j=f\%N$ | integer div/mod by width | host unflatten; 3-D copy |
| **6.8** | Stride-based | $o=i s_0+j s_1$ | strides in elements, not bytes | stride-argument copy |
| **6.9** | Broadcasting offsets | $O=\text{rows}\,s_0+\text{cols}\,s_1$ | $(B_M,1)+(1,B_N)\to(B_M,B_N)$ | tile copy |
| **6.10** | Tiled offsets | $(p_mB_M+m)N+(p_nB_N+n)$ | grid $(\lceil M/B_M\rceil,\lceil N/B_N\rceil)$ | tile copy |
| **6.11** | Boundary masks | $[m\lt M]\land[n\lt N]$ | mask shape $(B_M,B_N)$ | tile copy |
| **6.12** | Arbitrary dimensions | $o=\sum_k i_k s_k$ | one `arange` per axis | `tl.static_range` |
| **★** | Round trip | $o=f^{-1}(f)$ | same shape and strides | all demos |

## Worked numerical example

- **Connection.** Applies the maps of 6.1–6.7 to one matrix.
- **Example.** Let $A$ be the $3\times4$ matrix below with values $0,\ldots,11$ and $N=4$.

$$ A=\begin{bmatrix}0&1&2&3\\4&5&6&7\\8&9&10&11\end{bmatrix} $$

- **Flatten.** $f(i,j)=4i+j$, so $f(1,2)=6$ and $f(2,0)=8$; its inverse maps $6\mapsto(1,2)$ and $8\mapsto(2,0)$.
- **Rows.** $o_{\text{row}}(2,j)=2\cdot4+j=8+j$, so row 2 has offsets $(8,9,10,11)$ and values $(8,9,10,11)$.
- **Columns.** $o_{\text{col}}(i,3)=4i+3$, so column 3 has offsets $(3,7,11)$ and values $(3,7,11)$.
- **Table.** Every flat index maps back as follows.

| $f$ | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| row $f//4$ | 0 | 0 | 0 | 0 | 1 | 1 | 1 | 1 | 2 | 2 | 2 | 2 |
| col $f\bmod4$ | 0 | 1 | 2 | 3 | 0 | 1 | 2 | 3 | 0 | 1 | 2 | 3 |

- **3-D case.** For $T$ of shape $(2,3,4)$ with values $0,\ldots,23$ and strides $(12,4,1)$, $o(1,2,3)=23$ and $o(0,1,2)=6$; plane $i=0$ covers $0,\ldots,11$ and plane $i=1$ covers $12,\ldots,23$.
- **Check / where this leads.** The notebook asserts every offset, table entry, and round trip; Chapter 7 lifts these maps into elementwise arithmetic.

## The canonical kernels

- **Tile copy.** The 2-D tile copy exercises 6.2, 6.9, 6.10, and 6.11 together.

```python
rows = pid_m * BM + tl.arange(0, BM)
cols = pid_n * BN + tl.arange(0, BN)
mask = (rows[:, None] < M) & (cols[None, :] < N)
offs = rows[:, None] * N + cols[None, :]
x = tl.load(x_ptr + offs, mask=mask, other=0.0)
tl.store(out_ptr + offs, x, mask=mask)
```

- **Gathers.** A row gather loads `src = idx[r]` and uses $o=\text{src}\cdot N+j$; a column gather uses $o=i\cdot N+\text{src}$.
- **3-D copy.** Recover $i=f//(D_1D_2)$, $j=(f\bmod D_1D_2)//D_2$, $k=f\bmod D_2$, then form `src = i*s0 + j*s1 + k*s2`.
- **Applied kernels.** A scalar load broadcasts across a row tile, and `tl.static_range` unrolls the rank-3 sum over $D_0$.

## Pitfalls

| Pitfall | What happens | Fix |
|---|---|---|
| Row and column swapped | $o=jN+i$ transposes silently | write $o=iN+j$; test a non-square shape |
| Missing `pid * B` | every program reads the first tile | $o=\text{pid}\cdot B+i$ |
| Off-by-one tail | the last lane accesses out of bounds | $o\lt N$, not $o\le N$; `cdiv` the grid |
| Rank-1 mask | one axis runs out of bounds | `(rows[:,None]<M) & (cols[None,:]<N)` |
| `rows + cols` | vector add, not an outer sum | expand with `[:,None]`/`[None,:]` |
| int32 overflow | offsets wrap above $2^{31}-1$ elements | compute the flat index in `int64` |
| Wrong flatten width | every recovered row is wrong | divide by $N$, the column count |
| Non-contiguous input | hard-coded $N$ reads the wrong element | pass and use `x.stride()` |
| 3-D order | $iD_0D_1+\cdots$ reads the wrong plane | $(iD_1+j)D_2+k$ |
| Byte/element stride | addresses scaled by `element_size()` | pass `x.stride()` |
| Output stride | column gather writes past $(M,K)$ when $K\ne N$ | use the output stride $K$ |

## How to run

```bash
cd part2_memory/ch06_indexing
jupyter notebook 06_complete.ipynb
```

- The notebook walks up to `00_Basic` and imports `ttools.py`.
- On CPU every numerical and host-reference cell runs, and each Triton test prints `SKIP ...`; on a GPU every kernel is compared with a PyTorch reference.

```bash
cd 00_Basic
TRITON_INTERPRET=1 PYTHONPATH=$PWD python tools/interpreter_smoke.py
```

## Study checklist

- [ ] Write $f=iN+j$ and its inverse by hand for the $3\times4$ matrix.
- [ ] Fill in the flat-to-coordinate table without looking.
- [ ] State the strides of $(2,3,4)$ and compute $o(1,2,3)$.
- [ ] Explain why `rows[:, None]` is rank 2 and `[None, :]` adds an axis.
- [ ] Draw the six-tile map for $M=5$, $N=7$, $B_M=2$, $B_N=4$ and mark dead lanes.
- [ ] Break the 2-D mask into a rank-1 mask and explain which axis escapes.
- [ ] Pass a transposed tensor through the stride-argument copy and check it.
- [ ] Promote a flat index to `int64` and explain the overflow threshold.
- [ ] Re-implement the column gather with the output stride $K$.
- [ ] Do exercises 2, 4, and 7: they expose the axis, mask, and overflow bugs.

## Exercises

- **Offset tables.** Build the $f$, $(i,j)$, and $o(i,j)$ table for a $4\times5$ matrix.
- **Row versus column traffic.** Gather rows and columns of a $6\times4$ matrix with $(3,0,3,1)$.
- **New tile map.** Repeat the table for $M=7$, $N=5$, $B_M=4$, $B_N=2$.
- **Rank-1 mask bug.** Use `(rows < M)` only and describe the first failing access.
- **Strided copy.** Pass a transposed $4\times6$ tensor with a leading batch axis to `copy3d`.
- **Column-major flatten.** Replace $f=iN+j$ with $f=jM+i$ and update the inverse.
- **int32 overflow.** Force `flat` to stay `int32` and assert the widest offset for a large shape.
- **Rank 4.** Extend `static_range_offset_kernel` with a second `tl.static_range`.
- **Softmax offsets.** Write the row-wise max and sum offset vectors.
- **Traffic.** Compare a row-wise and a 2-D tiled $1024\times1024$ copy in loads, stores, and bytes.

- Hints are in [`../../docs/SOLUTIONS.md`](../../docs/SOLUTIONS.md).
