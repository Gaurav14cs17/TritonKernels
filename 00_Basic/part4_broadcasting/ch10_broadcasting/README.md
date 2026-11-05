# Chapter 10 — Broadcasting

Part 4: Broadcasting and tensor manipulation.

> **Goal.** Read broadcasting as an *index map*, not a loop and not a copy. The shape rule decides whether the map exists; the stride rule makes a size-1 axis a zero-stride view; the kernel rule indexes the short operand.

[Open the chapter notebook](10_complete.ipynb)

[Index](../../00_index.ipynb) · [← Previous](../../part3_math/ch09_activations/09_complete.ipynb) · [Next →](../ch11_index_transforms/11_complete.ipynb)

**Estimated time:** 60–75 min · **Difficulty:** Intermediate

## Connections

- **Previous.** Chapter 9 supplies elementwise tiles, activation functions, and masks.
- **Earlier.** Chapter 6 supplies pointer offsets, 2-D indexing, and boundary masks.
- **Builds on.** The shape rule pads on the left, right-aligns axes, and compares each pair once.
- **Leads to.** Chapter 11 reuses zero strides for flatten, reshape, transpose, gather, and scatter.
- **Through-line.** Scalar → vector/row/column → the four `[M,N]` pairings → 3-D and multiple axes → incompatible shapes → zero strides → no-copy fusion.

## Learning objectives

- State and apply the broadcast-shape rule: right-align, pad with `1`, require every axis to be equal or `1`.
- Read `[M,N]+[N]`, `[M,N]+[M,1]`, `[M,N]+[1,N]`, `[M,N]+scalar`, and 3-D broadcasts as address maps.
- Explain zero strides and compute the traffic saved by avoiding a copy.
- Write Triton kernels for a row bias, a column scale, and a 3-D bias.
- Name the first axis that makes an incompatible pair fail.
- Fuse a per-channel affine $y=x\,s+b$ without materializing $s$ or $b$.

## Study ladder

1. **Method 01 — host reference.** `torch.broadcast_tensors` plus a scalar loop.
2. **Method 02 — Triton kernels.** Row bias indexes by `cols`; column scale by `rows`.
3. **Method 03 — no-copy fusion.** A per-channel affine reads two short vectors.

**Study tip.** Right-align and compare one axis at a time:

- `(3,4)` with `(4,)` is `3 4` over `1 4`: `4 = 4`, then `3` vs `1`, so the result is `(3,4)`.
- `(3,4)` with `(3,1)` is `3 4` over `3 1`: `4` vs `1`, then `3 = 3`, so the result is `(3,4)`.
- `(3,4)` with `(4,3)` fails at the rightmost axis: `4` vs `3`.

## Mathematical definition

- **Padded shapes.** Let $a=(a_0,\dots,a_{d-1})$ and $b=(b_0,\dots,b_{d-1})$, with `1` inserted on the left until the ranks match.
- **Result shape.** $r_k=a_k$ when $b_k=1$; $r_k=b_k$ when $a_k=1$; $r_k=a_k$ when $a_k=b_k$; otherwise error.
- **Index map.** $\pi_k(i_k)=0$ when $a_k=1$, else $i_k$; $\tilde a_{i_0,\dots}=a_{\pi_0(i_0),\dots}$.
- **Elementwise rule.** $Y_{\mathbf i}=\tilde a_{\mathbf i}\star\tilde b_{\mathbf i}$.
- **Consequence.** A broadcast axis ignores its output index, so its stride is zero and no element is duplicated in storage.

## Worked numerical example

**Spine.**

- $X=\begin{bmatrix}1&2&3&4\\5&6&7&8\\9&10&11&12\end{bmatrix}$, $r=\begin{bmatrix}10&20&30&40\end{bmatrix}$.
- $c=\begin{bmatrix}100\\200\\300\end{bmatrix}$, $s=5$.

**Broadcast results.**

- **Row.** $r$ is padded to `[1,4]`: $X+r=\begin{bmatrix}11&22&33&44\\15&26&37&48\\19&30&41&52\end{bmatrix}$.
- **Column.** $c$ is already `[3,1]`: $X+c=\begin{bmatrix}101&102&103&104\\205&206&207&208\\309&310&311&312\end{bmatrix}$.
- **Scalar.** $s$ is padded to `[1,1]`: $X+s=\begin{bmatrix}6&7&8&9\\10&11&12&13\\14&15&16&17\end{bmatrix}$.
- **Explicit row.** $X+\begin{bmatrix}10&20&30&40\end{bmatrix}=X+r$.

**Incompatible pairs.**

- $(3,4)$ with $(2,4)$: axis $-1$ passes ($4=4$), axis $-2$ fails ($3$ vs $2$).
- $(3,4)$ with $(4,3)$: axis $-1$ fails ($4$ vs $3$).

The notebook asserts every matrix and both failures.

## Rules

**Right-align.**

- Pad the shorter shape with `1` on the **left**, then compare from the right.
- `(3,4)+(4,)` compares $(3,4)$ with $(1,4)$.

**Size-1.**

- A size-1 axis expands to the other operand's size on that axis.
- It is logical replication, not an allocation; its expanded stride is `0`.

**Scalar.**

- A scalar has shape `()`, pads to `[1,1]`, and expands on every axis.
- A true scalar kernel argument is read once; a length-1 tensor is a pointer and must be indexed as element `0`.

**Incompatible.**

- Every axis must be equal or one side must be `1`; the first violation from the right is the axis to report.
- `torch.broadcast_shapes` raises on the host; a Triton kernel cannot, because it sees only raw pointers.

**Zero-stride.**

- With $\operatorname{addr}(i,j)=\text{base}+i\,s_i+j\,s_j$:

| view | shape | stride | address |
|---|---|---|---|
| $X$ | $(3,4)$ | $(4,1)$ | $\text{base}+4i+j$ |
| row $r$ | $(3,4)$ | $(0,1)$ | $\text{base}+j$ |
| column $c$ | $(3,4)$ | $(1,0)$ | $\text{base}+i$ |
| scalar $s$ | $(3,4)$ | $(0,0)$ | $\text{base}$ |

- `expand` produces these strides for free; `.contiguous()` replaces them with a dense stride and allocates the full shape.

**Kernel.**

- A kernel receives no shape metadata and does not infer a broadcast from strides.
- Index the short operand explicitly (`bias_ptr + cols`, `col_ptr + rows`), pass a true scalar argument, or copy on the host and accept the cost.
- The chapter uses the first option; the mask guards load and store.

## Chapter map

| § | Topic | Rule | Shapes | Kernel/demo |
|---|---|---|---|---|
| **10.1** | Scalar broadcasting | every entry adds $s$ | $[M,N]+()\to[M,N]$ | scalar tile |
| **10.2** | Vector broadcasting | $[N]$ aligns right | $[N]+[N]\to[N]$ | vector add |
| **10.3** | Row broadcasting | $[1,N]$ expands rows | $[M,N]+[1,N]\to[M,N]$ | row bias |
| **10.4** | Column broadcasting | $[M,1]$ expands columns | $[M,N]+[M,1]\to[M,N]$ | column scale |
| **10.5** | `[M,N] + [N]` | pad to $[1,N]$ | $[M,N]+[N]\to[M,N]$ | row bias |
| **10.6** | `[M,N] + [M,1]` | keep trailing 1 | $[M,N]+[M,1]\to[M,N]$ | column scale |
| **10.7** | `[M,N] + [1,N]` | explicit row axis | $[M,N]+[1,N]\to[M,N]$ | row bias |
| **10.8** | `[M,N] + scalar` | pad to $[1,1]$ | $[M,N]+()\to[M,N]$ | scalar tile |
| **10.9** | 3-D broadcasting | batch and rows expand | $[B,M,N]+[N]\to[B,M,N]$ | 3-D bias |
| **10.10** | Multiple dimensions | axes expand independently | $[B,1,N]+[1,M,1]\to[B,M,N]$ | 3-D kernel |
| **10.11** | Incompatible shapes | equal or one, else error | first bad axis | host check |
| **10.12** | Zero-stride concepts | broadcast stride is 0 | $(0,1)$, $(1,0)$, $(0,0)$ | `expand` |
| **10.13** | Without copies | view, not allocation | $(2M+1)Nb$ | per-channel affine |

**Thread.** The shape rule decides whether the map exists; the stride rule says it is free; the kernel rule says to index the short operand.

## Traffic: view versus copy

- **Row bias.** For $M$ rows, $N$ columns, and $b$ bytes per element, $Q_{\text{fused}}=(2M+1)Nb$ and $Q_{\text{copy}}=(4M+1)Nb$.
- **Per-channel affine.** $Q_{\text{fused}}=(2M+2)Nb$ and $Q_{\text{copy}}=(6M+2)Nb$.
- **Numbers.** For $M=3$, $N=4$, `float32`, the row bias is $112$ bytes versus $208$ bytes; the affine is $128$ bytes versus $320$ bytes, a factor of $2.5$.
- **Reason.** Every elementwise kernel here is memory-bound, and a copy adds traffic without arithmetic.

## 3-D broadcasting

- **Rule.** For $X\in\mathbb R^{B\times M\times N}$ and $b\in\mathbb R^{N}$: $Y_{bij}=X_{bij}+b_j$.
- **Padding.** The vector is padded to `[1,1,N]` and expands across batch and rows.
- **Example.** With $b=(10,20,30)$, $X_0=\begin{bmatrix}1&2&3\\4&5&6\end{bmatrix}$, and $X_1=\begin{bmatrix}7&8&9\\10&11&12\end{bmatrix}$: $Y_0=\begin{bmatrix}11&22&33\\14&25&36\end{bmatrix}$ and $Y_1=\begin{bmatrix}17&28&39\\20&31&42\end{bmatrix}$.
- **Multiple axes.** $[B,1,N]+[1,M,1]\to[B,M,N]$; each size-1 axis expands independently.

## Pitfalls

| Pitfall | What happens | Fix |
|---|---|---|
| Left alignment | axes compare in the wrong order | right-align; pad left |
| `(M,)` versus `(N,)` | a length-$M$ vector attaches to columns | use `v[:, None]` |
| `.view()` on an expanded tensor | raises on the zero-stride axis | `reshape` only if a copy is fine |
| Writing through `expand` | one write aliases every row | allocate a fresh output |
| Large `.contiguous()` | allocates the full broadcast shape | index the short operand |
| Kernel infers a broadcast | raw pointers carry no shape | check shapes, pass offsets |
| Flat offset for a bias | `offs % N` missing, bias repeats a row | index with `cols` |
| Scalar as a length-1 pointer | kernel reads past element `0` | pass a true scalar |
| Dropping `keepdim` | `[1,N]` becomes `[N]` and attaches elsewhere | keep the size-1 axis |
| Wrong `other` | a dead lane contributes `0` to a product | use the neutral element |

## Self-check

1. Right-align `(5,1,7)` and `(3,7)`; write the result shape.
2. Why does `(3,4)+(3,)` fail while `(3,4)+(4,)` succeeds?
3. What are the strides of the row and column `expand` views for `(4,6)`?
4. Which term disappears in the address formula for a row broadcast?
5. What is the traffic ratio for a fused row bias at $M=64$, $N=512$?
6. What does a write through an expanded view do to the other rows?
7. In the row-bias kernel, which lanes does the bias load touch?
8. Extend the 3-D kernel to `[B,M,N]+[M,1]`; which pointer index changes?

## How to run

```bash
cd part4_broadcasting/ch10_broadcasting
jupyter notebook 10_complete.ipynb
```

The notebook imports `ttools.py` by walking up to `00_Basic`. On CPU every numerical and host-side cell runs; Triton tests print `SKIP ...`. On a GPU each kernel is compared with a PyTorch reference. For a smoke test: `TRITON_INTERPRET=1 PYTHONPATH=$PWD python tools/interpreter_smoke.py`.

## Study checklist

- [ ] Write the padded shapes and axis comparisons for exercise 1.
- [ ] Explain why `[N]` aligns last and why rows need `[M,1]`.
- [ ] Derive the strides of `r.expand(3,4)` and `c.expand(3,4)`.
- [ ] Break the row mask and explain which access fails.
- [ ] Compute fused and copied traffic and check the ratio.
- [ ] Explain what a write through `expand` does to the base vector.
- [ ] Run the 3-D kernel with non-multiple `M` or `N`.
- [ ] Do exercises 4, 5, and 9 (strides, traffic, backward).

## Exercises

The notebook ends with ten: axis tables, first-failure detection, vector versus column, `as_strided`, large-shape traffic, ragged tiles, 3-D columns, per-channel affine copies, the backward sum over broadcast axes, and a no-copy wrapper assertion. Hints are in [`../../docs/SOLUTIONS.md`](../../docs/SOLUTIONS.md).

## Summary

- Broadcasting is a shape rule plus an index map, not a loop or a copy.
- Right-align, pad with `1`, and require every axis to be equal or `1`.
- A size-1 axis expands to the other size and has stride `0`.
- `[N]` attaches last; a column needs `[M,1]`; a scalar expands everywhere.
- 3-D and multi-axis broadcasts apply the same rule per axis.
- An incompatible pair fails at the first violating axis from the right.
- `expand` is free; `contiguous()` is a copy; fused kernels index the short operand.
