# 11 — Tensor Index Transformations

> **Goal.** Read every shape change as a change of the index map `addr(i) = offset + Σ i_k s_k`. Flatten, reshape, squeeze, expand, transpose, permute, slice, reverse, gather, scatter, and index selection are views or gathers over that one formula.

[Open the chapter notebook](11_complete.ipynb) · [Previous: Chapter 10](../ch10_broadcasting/10_complete.ipynb) · [Next: Chapter 12](../../part5_layouts/ch12_layouts/12_complete.ipynb)

[Index](../../00_index.ipynb) · [Setup](../../docs/SETUP.md) · [Notation](../../docs/NOTATION.md) · [Glossary](../../docs/GLOSSARY.md) · [API index](../../docs/API_INDEX.md) · [Formula index](../../docs/FORMULA_INDEX.md) · [Diagrams](../../docs/DIAGRAMS.md) · [Troubleshooting](../../docs/TROUBLESHOOTING.md) · [Exercise hints](../../docs/SOLUTIONS.md)

**Estimated time:** 70–85 min · **Difficulty:** Intermediate

**Thread.**

- **Recurring idea.** A transform changes the index map, not the values.
- **Recurring decision.** Is the result a view (same buffer, new shape/stride/offset) or a copy (new contiguous buffer)?
- **Recurring question.** What are the new shape, stride, and storage offset?
- **Chain.** flatten/unflatten → reshape → squeeze/expand → transpose/permute → slice/reverse → gather/scatter/index-select → non-contiguous strides → fused kernels.
- **Payoff.** Every Triton kernel is the same load/store skeleton with a different address formula.

## Connections
- **Backward.** Chapter 6 supplies pointers, offsets, strides, masks, `tl.load`, and `tl.store`.
- **Sideways.** Chapter 10 supplies broadcasting, size-1 axes, and zero-stride views.
- **Forward.** Chapter 12 turns the stride discussion into the layout of a Triton tile.
- **Inside.** flatten/unflatten → reshape → squeeze/expand → transpose/permute → slice/reverse → gather/scatter/index-select → non-contiguous handling → fused kernels; a kernel never infers a layout, it receives a pointer, shape, and strides.

## Learning objectives

- Read a shape change as a triple `(shape, stride, offset)` and decide whether it is a view or a copy.
- Derive flatten/unflatten, squeeze/unsqueeze, transpose/permute, and slice/reverse index maps.
- State when shape and stride are preserved, when a stride is swapped, and when a stride becomes zero.
- Write gather, scatter, scatter-add, and index selection as explicit address formulas.
- Write a tiled Triton transpose, a row-gather kernel, and an atomic scatter-add kernel.
- Fuse a transform with arithmetic so the data is read and written once.

## Prerequisites

- Chapter 6: pointers, offsets, strides, masks, `tl.load`, and `tl.store`; Chapter 10: broadcasting, size-1 axes, and zero-stride views.
- Comfort with row-major indexing: `f = i*N + j`.

## Study ladder

1. **Method 01 — host reference.** One Python iteration per selected row or element; the correctness oracle.
2. **Method 02 — three Triton kernels.** A tiled transpose, a row gather, and an atomic scatter-add.
3. **Method 03 — fused and strided.** Gather-plus-affine in one pass, and a transposed or sliced view copied with explicit `(s0, s1)`.

**Study tip.** Fix one `3×4` matrix for the whole chapter:

$$
X=\begin{bmatrix}0&1&2&3\\4&5&6&7\\8&9&10&11\end{bmatrix},
\qquad \operatorname{stride}(X)=(4,1).
$$

- Write the flatten map by hand: `f(i,j)=4i+j`.
- Write the transpose map by hand: `Xᵀ[i,j]=X[j,i]=4j+i`, with shape `(4,3)` and stride `(1,4)`.
- If both lines are right, every other transform is a variation on them.

## Mathematical definition

- **Object.** A tensor is its shape `(n0,…,nd−1)`, strides `(s0,…,sd−1)`, and offset `o`.
- **Address.** `addr(i0,…,id−1) = o + Σ i_k s_k`.
- **Contiguity.** Row-major contiguous means `s_{d−1}=1`, `s_k=n_{k+1}s_{k+1}` for `0≤k<d−1`, and `o=0`.
- **View or copy.** A view keeps the buffer and changes the triple; a copy allocates a new contiguous buffer.

| Family | Index map | Shape / stride |
|---|---|---|
| Flatten | `f=iN+j` | `(M,N)→(MN,)`, stride `(1,)` |
| Unflatten | `i=f//N, j=f%N` | `(MN,)→(M,N)`, stride `(N,1)` |
| Transpose | `Y_ij=X_ji` | `(N,M)`, `(s0,s1)→(s1,s0)` |
| Permute | `Y[j0,…]=X[i0,…]`, `i_π(k)=j_k` | shape and stride follow `π` |
| Slice | `Y_ij=X[a+ip,b+jq]` | `o'=o+as0+bs1`, `(ps0,qs1)` |
| Reverse | `Y_ij=X[M−1−i,N−1−j]` | negative stride view, or a copy |
| Gather | `Y_ij=X[idx_ij,j]` | output shape = index shape |
| Scatter | `dst[idx_ij,j]=src_ij` | destination indexed by data |
| Scatter-add | `dst[idx_ij,j]+=src_ij` | duplicates accumulate |
| Index select | `Y_k=X[idx_k]` | one whole slice per index |

## Rules

1. **Shape rule.** Flatten multiplies extents; unflatten splits one. Reshape may change any extents while `∏n_k=∏m_k`. Transpose and permute reorder extents and never change the element count.
2. **Stride rule.** Transpose swaps `(s0,s1)`. Permute applies the same permutation to strides as to shapes. Squeeze drops the size-1 stride and leaves the others alone. A slice scales strides by its steps. `expand` sets the new axis's stride to zero.
3. **View-or-copy rule.** `view` is legal only when the strides allow it; `reshape` copies otherwise; `.contiguous()` always copies. Assert the stride.
4. **Flatten-order rule.** Flatten follows the *logical* row-major order, not the storage order. `X.t().reshape(-1)` is `(0,4,8,1,…)`, not `(0,1,2,…)`.
5. **Bounds rule.** Gather indices are data. Assert `0 <= idx.min()` and `idx.max() < R`; the kernel performs no check.
6. **Collision rule.** A plain scatter with duplicate indices is a race. Use `tl.atomic_add`, or a sorted segmented reduction for determinism.
7. **Negative-index rule.** PyTorch wraps `-1`; Triton pointer arithmetic does not. Normalize with `idx % R` or reject negatives before launch.

## Worked numerical example

**Thread.** Every number below is asserted by the notebook's worked-example cell.

- **Spine.** With stride `(4,1)` and offset `0`:

$$
X=\begin{bmatrix}0&1&2&3\\4&5&6&7\\8&9&10&11\end{bmatrix}.
$$

- **Flatten and unflatten.** `flat(X)=(0,1,2,3,4,5,6,7,8,9,10,11)`; the inverse sends `f=7` to `(1,3)`, `f=10` to `(2,2)`, `f=11` to `(2,3)`.
- **Reshape, not transpose.** To `(2,6)`: `[[0,1,2,3,4,5],[6,7,8,9,10,11]]`, stride `(6,1)`. To `(4,3)`: `[[0,1,2],[3,4,5],[6,7,8],[9,10,11]]`, stride `(3,1)` — not `Xᵀ`.
- **Transpose.** Shape `(4,3)` and stride `(1,4)`:

$$
X^\top=\begin{bmatrix}0&4&8\\1&5&9\\2&6&10\\3&7&11\end{bmatrix}.
$$

  Check `Y[2,1]=X[1,2]=6`; both addresses equal `2·1+1·4=1·4+2=6`.
- **Permute.** For `A` of shape `(2,3,4)` with `A[b,i,j]=12b+4i+j` and stride `(12,4,1)`, `π=(2,0,1)` gives `P[j,b,i]=A[b,i,j]`, shape `(4,2,3)`, stride `(1,12,4)`. Check `P[3,1,2]=A[1,2,3]=23` and `3·1+1·12+2·4=23`.
- **Slice and reverse.** `X[1:3,1:3]=[[5,6],[9,10]]` has offset `5` and stride `(4,1)`; `X[0:3:2,0:4:2]=[[0,2],[8,10]]` has stride `(8,2)`. Reversing both axes uses `Y[i,j]=X[2−i,3−j]`:

$$
\operatorname{flip}(X)=\begin{bmatrix}11&10&9&8\\7&6&5&4\\3&2&1&0\end{bmatrix},
\qquad Y[0,0]=X[2,3]=11.
$$

- **Gather.** With `S=[[10,11,12],[20,21,22],[30,31,32],[40,41,42]]` and `idx=(2,0,3,1)`:

| `idx` | output row |
|---|---|
| 2 | 30 31 32 |
| 0 | 10 11 12 |
| 3 | 40 41 42 |
| 1 | 20 21 22 |

  With `idx=[[0,1,3],[2,0,1]]`: `Y[0,2]=S[3,2]=42`, `Y[1,0]=S[2,0]=30`, and `Y=[[10,21,42],[30,11,22]]`.
- **Scatter-add.** With `src=[[1,2],[3,4],[5,6],[7,8],[9,10]]` and `idx=(0,2,0,1,2)`:

| source row | `idx` | adds into |
|---|---|---|
| `[1,2]` | 0 | row 0 |
| `[3,4]` | 2 | row 2 |
| `[5,6]` | 0 | row 0 |
| `[7,8]` | 1 | row 1 |
| `[9,10]` | 2 | row 2 |

  The result is `dst=[[6,8],[7,8],[12,14]]`: `dst[0]=src[0]+src[2]`, `dst[1]=src[3]`, and `dst[2]=src[1]+src[4]`.

## Chapter map

| § | Transform | Formula | Shape/stride rule | Kernel/demo |
|---|---|---|---|---|
| **11.1** | Flatten | `f=iN+j` | `(M,N)→(MN,)`, stride `(1,)` if contiguous | verification A |
| **11.2** | Unflatten | `i=f//N, j=f%N` | `(MN,)→(M,N)`, stride `(N,1)` | verification A |
| **11.3** | Reshape | same flat index `f` | `∏n_k=∏m_k`; target stride `(m1,1)` | verification A |
| **11.4** | Squeeze | drop `n_p=1` | remaining strides unchanged | verification B |
| **11.5** | Expand dims | insert `1` at `p` | new stride arbitrary; `0` after `expand` | verification B |
| **11.6** | Transpose | `Y_ij=X_ji` | `(s0,s1)→(s1,s0)` | transpose tile kernel |
| **11.7** | Permute | `Y[j0,…]=X[i0,…]` | shape and stride follow `π` | verification B |
| **11.8** | Slice | `Y_ij=X[a+ip,b+jq]` | `o'=o+as0+bs1`, `(ps0,qs1)` | verification C |
| **11.9** | Reverse | `Y_ij=X[M−1−i,N−1−j]` | negative stride view, or a copy | verification C |
| **11.10** | Gather | `Y_ij=X[idx_ij,j]` | output shape = index shape | row-gather kernel |
| **11.11** | Scatter | `dst[idx_ij,j]=src_ij` | destination indexed by data | atomic scatter-add |
| **11.12** | Index selection | `Y_k=X[idx_k]` | one whole slice per index | row-gather kernel |
| **11.13** | Non-contiguous | `addr=o+Σi_k s_k` | pass the real strides | strided ReLU kernel |
| **★** | Worked example | all of the above | `3×4` and `2×3×4` spines | asserts only |

## The canonical kernels

- **Transpose.** Load a `B_M×B_N` tile, `tile_t = tl.trans(tile)`, then store at `cols[:,None]*M + rows[None,:]`.
- **Gather.** `src_row = tl.load(idx_ptr + k)`, then `tl.load(src_ptr + src_row*D + cols, mask=cols < D)`.
- **Scatter-add.** `dst_row = tl.load(idx_ptr + k)`, then `tl.atomic_add(out_ptr + dst_row*D + cols, value, mask=cols < D)`.
- **Fused gather+affine.** Store `value*scale + bias` from the gather row; one read and one write.
- **Strided copy.** Replace `row*N+col` with `row*s0+col*s1`; the same kernel then accepts transposed and sliced views.
- **Cost.** Unfused gather-then-scale moves about `4MDb`; the fused kernel moves about `2MDb` for `M` rows of width `D` in `b`-byte elements.

## Pitfalls

| Pitfall | What happens | Fix |
|---|---|---|
| `reshape` versus `view` | `view` raises on a non-contiguous tensor; `reshape` copies silently | choose deliberately and assert `shape`/`stride` |
| Transposed view treated as contiguous | the kernel uses `N` instead of `stride(0)`; shapes match, values do not | pass `t.stride(0)`, `t.stride(1)` |
| Gather index out of bounds | undefined read or a fault, with no Python exception | assert `0 <= idx.min()` and `idx.max() < R` |
| Scatter collisions | a plain store keeps only the last writer | use `tl.atomic_add` or a sorted reduction |
| Negative indices | PyTorch wraps `-1`; Triton pointer arithmetic does not | normalize with `idx % R` or reject negatives |
| Flatten order | the flat order is logical row-major, not storage order | flatten with `reshape`, never `view`, on a view |
| Permute axes | `dims` is the new axis order, not its inverse | test one interior address, not only the shape |
| Squeeze the wrong axis | a valid shape with the wrong layout | assert both `shape` and `stride` |
| Slice endpoint | `stop` is excluded; `step < 0` raises | check the count `ceil((c-a)/p)` |
| Index dtype | a float index is truncated or rejected | cast to `torch.int32` or `torch.int64` |

## Self-check

1. What is the address of `X[2,1]` and of `Xᵀ[1,2]`? Which tensor has stride `(1,4)`?
2. Flatten with `f=iN+j`. What are `f(2,3)`, `f(1,2)`, and the coordinates of `f=9`?
3. Reshape the spine to `(4,3)` and transpose it. Why are the results different?
4. A tensor has shape `(2,1,3)` and strides `(3,3,1)`. Give the shape and stride after `squeeze(1)` and after `squeeze(0)`.
5. For `A` of shape `(2,3,4)` and stride `(12,4,1)`, give the shape and stride of `A.permute(2,0,1)`. Where is `A[1,2,3]`?
6. Why is the stride of `X[1:3,1:3]` `(4,1)` and not `(2,1)`? Why is the backward of a gather a scatter-add?

## How to run

```bash
cd part4_broadcasting/ch11_index_transforms && jupyter notebook 11_complete.ipynb
(cd ../.. && TRITON_INTERPRET=1 PYTHONPATH=$PWD python tools/interpreter_smoke.py)
```

The notebook imports `ttools.py` by walking up to `00_Basic`. On CPU every numerical, Method 01, worked-example, and autograd cell runs; each Triton test prints `SKIP …`. The interpreter exercises the transpose, gather, scatter-add, fused, and strided kernels on CPU tensors.

## Study checklist

- [ ] Write `Xᵀ` and `flatten(X)` for the `3×4` spine by hand.
- [ ] State the shape and stride of `X.t()` and of `X.reshape(4,3)`.
- [ ] Explain why `X.t().view(-1)` fails and what `.reshape(-1)` returns.
- [ ] Derive the stride of `X[1:3,1:3]` and of `X[0:3:2,0:4:2]`.
- [ ] Write the gather and scatter-add formulas from memory.
- [ ] Explain when a scatter needs atomics and when it does not.
- [ ] Read the strided ReLU kernel and identify `s0`, `s1`, and the mask.
- [ ] Break one stride in a copy and predict the wrong values before running it.

## Exercises

1. **Flatten order.** Flatten a `3×4` matrix in column-major order and verify it against `X.t().reshape(-1)`.
2. **Unflatten by hand.** For `M=4`, `N=5`, list `(i,j)` for `f=0,6,12,19`.
3. **Reshape or transpose.** Compare `X.reshape(4,3)` and `X.t()`; state which preserves the flat order.
4. **Squeeze audit.** Take shape `(2,1,3,1)`, squeeze both singleton axes, and assert the resulting shape and stride.
5. **Permute inverse.** For `π=(2,0,1)`, find `π⁻¹` and verify that permuting by `π` then `π⁻¹` is the identity.
6. **Strided slice.** Copy `X[0:3:2,1:4:2]` with a kernel that takes `start_m, step_m, start_n, step_n`.
7. **Reverse one axis.** Write a kernel for `torch.flip(x, dims=(1,))` and check that reversing twice is the identity.
8. **Gather bounds.** Add a host check for an out-of-range index, then a masked variant that drops the bad row.
9. **Deterministic scatter-add.** Replace the atomics with a sort by destination index and a segmented sum; compare bitwise across runs.
10. **Fusion traffic.** For `M=1024`, `D=768`, and `float32`, compare the bytes moved by gather-then-scale and by the fused kernel.

Hints are in [`../../docs/SOLUTIONS.md`](../../docs/SOLUTIONS.md).
