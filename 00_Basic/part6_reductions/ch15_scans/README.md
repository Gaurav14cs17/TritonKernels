# Chapter 15 — Scan Operations

Part 6: Reductions, scans, and combined recipes.

> **Goal.** Read a scan as a reduction that keeps every prefix. A reduction
> collapses an axis to one value; a scan keeps one value per position. The
> operator must be associative, the output shape must equal the input shape, and
> the exclusive form must start from the identity.

[Open the chapter notebook](15_complete.ipynb) · [Shared helpers](../../ttools.py) · [Setup](../../docs/SETUP.md) · [Notation](../../docs/NOTATION.md) · [Glossary](../../docs/GLOSSARY.md) · [API index](../../docs/API_INDEX.md) · [Formula index](../../docs/FORMULA_INDEX.md) · [Diagrams](../../docs/DIAGRAMS.md) · [Troubleshooting](../../docs/TROUBLESHOOTING.md) · [Exercise hints](../../docs/SOLUTIONS.md)

[Index](../../00_index.ipynb) · [← Previous](../ch14_reductions/14_complete.ipynb) · [Next →](../ch16_combining_ops/16_complete.ipynb)

**Estimated time:** 70–85 min · **Difficulty:** Intermediate

## Learning objectives

- Read a scan as a reduction that keeps every prefix, and state the difference in output shape.
- Choose inclusive or exclusive, and convert one into the other with a shift and the identity.
- Write `tl.cumsum` and `tl.associative_scan` kernels; state the associativity requirement.
- Implement Hillis–Steele doubling by hand and count its work and depth.
- Apply a scan to position ids, segmented sums, compaction, and a first-order recurrence.
- Explain why a scan is not a reduction with a loop around it.

## Prerequisites

- Chapter 14: `tl.sum`, `tl.max`, axis reductions, and the reduction shape rule.
- Chapter 5: masked loads and stores, pointer offsets, and `other=`.
- Comfort with $\sum_{j=0}^{i}$ notation and base-2 logarithms.

## Thread

- **Recurring idea:** a scan is a reduction that keeps every prefix; the operator is associative and the output shape never changes.
- **Chain:** Chapter 14 reduction → 15.1 prefix sum → 15.2 cumulative sum → 15.3 `tl.associative_scan` → 15.4 inclusive → 15.5 exclusive → 15.6 parallel scan → 15.7 reduction vs scan → 15.8 applications.
- **Shape thread:** a scan preserves every axis; a reduction removes the reduced axis.
- **Offset thread:** the exclusive scan is the offset generator behind positions, compaction, and segments.

## Connections

- **Backward:** Chapter 14 supplies `tl.sum`, axis reductions, and the shape rule that a scan inverts by keeping every prefix.
- **Forward:** Chapter 16 composes reductions and scans into softmax, log-sum-exp, LayerNorm statistics, and online softmax.
- **Within Part I:** 15.1 defines the prefix sum; 15.2 names its row-wise form; 15.3 generalizes the operator; 15.4 and 15.5 fix the two alignments; 15.6 schedules the tree; 15.7 separates scan from reduction; 15.8 applies all of it.
- **Applied:** Part IV turns 15.8 into segmented scans, stream compaction, and a first-order recurrence.

## Study ladder

1. **Method 01 — naive loop.** A Python accumulator walks left to right. It is the correctness oracle, has $O(N)$ depth, and is far too serial for a GPU.
2. **Method 02 — Triton kernels.** `tl.cumsum`, `tl.associative_scan`, the exclusive shift, and a manual Hillis–Steele kernel that exposes the tree.
3. **Method 03 — applied scans.** Segmented position ids and sums, stream compaction, and an affine scan for a first-order recurrence.

**Study tip.**

- Fix the spine $x=(3,1,4,1,5)$.
- Inclusive keeps the current element: $(3,4,8,9,14)$.
- Exclusive drops it and starts at $0$: $(0,3,4,8,9)$.
- The last inclusive value is the reduction $14$; the last exclusive value is $14-5=9$.

## Mathematical definition

- **Inclusive.** $y_i=x_0\oplus x_1\oplus\cdots\oplus x_i$ for $i\in\{0,\dots,N-1\}$.
- **Exclusive.** $e_i=x_0\oplus\cdots\oplus x_{i-1}$ and $e_0=e$.
- **Identity.** For addition $e=0$; for multiplication $e=1$; for maximum $e=-\infty$.
- **Associativity.** $(a\oplus b)\oplus c=a\oplus(b\oplus c)$ is the license to re-parenthesize into a tree.
- **Counterexample.** On the spine, subtraction gives $(((3-1)-4)-1)-5=-8$, while $((3-1)-4)-(1-5)=2$.

## Rules

### Shape rule

- A scan preserves the input shape: $\operatorname{shape}(y)=\operatorname{shape}(x)$.
- A reduction removes the reduced axis: `tl.sum(A, axis=0)` on $(R,C)$ returns $(C,)$.
- A scan output that lost a dimension is a reduction; a reduction that kept every prefix is a scan.

### Associativity rule

- `tl.associative_scan(x, axis, combine_fn)` requires an associative `@triton.jit` combine function.
- For $k$ state tensors it takes $2k$ scalar arguments and returns $k$ values.
- Sum, product, min, max, gcd, bitwise and/or, affine pairs, and `(head, value)` pairs qualify; subtraction, division, and a running mean without a carried count do not.

### Identity rule

| Operator | Identity `other` |
|---|---|
| sum | `0.0` |
| product | `1.0` |
| maximum | `-inf` |
| minimum | `+inf` |
| affine recurrence | `(1.0, 0.0)` |

The exclusive shift `x_ptr + offs - 1` with `mask=(offs > 0) & mask` and `other=e` puts the identity in slot zero and leaves memory untouched.

### Axis rule

- The scan runs along exactly one axis.
- For row-wise sums of $(R,C)$ use `axis=1`; for column-wise use `axis=0`.
- Both return $(R,C)$, so a wrong axis is shape-correct and fails only a value check.

### Offset rule

- An exclusive scan produces offsets, not totals.
- For a keep mask $k$, $\operatorname{pos}_i=\sum_{j\lt i}k_j$ is the destination of item $i$ in stream compaction.
- The scan still has length $N$; only the output shrinks to $K=\sum_j k_j$.

## Worked example (chapter spine)

| $i$ | 0 | 1 | 2 | 3 | 4 |
|---|---:|---:|---:|---:|---:|
| $x_i$ | 3 | 1 | 4 | 1 | 5 |
| inclusive $y_i$ | 3 | 4 | 8 | 9 | 14 |
| exclusive $e_i$ | 0 | 3 | 4 | 8 | 9 |

- **Shift.** The exclusive scan is the inclusive scan of $(0,3,1,4,1)$.
- **Subtract form.** For addition, $e_i=y_i-x_i$ gives $(0,3,4,8,9)$.
- **Hillis–Steele.** Doubling the distance each round:

| round | $d$ | array |
|---|---:|---|
| 0 | — | 3 1 4 1 5 |
| 1 | 1 | 3 4 5 5 6 |
| 2 | 2 | 3 4 8 9 11 |
| 3 | 4 | 3 4 8 9 14 |

- **Count.** After $\lceil\log_2 5\rceil=3$ rounds the scan is complete; the active combines are $4,3,1$, so the tree uses $8$ adds where a loop uses $4$.
- **Warning.** Subtraction is not associative; the left fold is $-8$ and the balanced fold is $2$.

## Chapter map

| § | Topic | Formula | Shape/associativity rule | Kernel/demo |
|---|---|---|---|---|
| **15.1** | Prefix sum | $y_i=\sum_{j=0}^{i}x_j$ | $\operatorname{shape}(y)=\operatorname{shape}(x)$ | `inclusive_scan_kernel` |
| **15.2** | Cumulative sum | $C_{r,c}=\sum_{k=0}^{c}A_{r,k}$ | axis preserved: $(R,C)\to(R,C)$ | `scan2d_kernel` (`AXIS=1`) |
| **15.3** | `tl.associative_scan()` | $y_i=x_0\oplus\cdots\oplus x_i$ | $\oplus$ associative; $k$ states in, $k$ out | `assoc_scan_kernel` |
| **15.4** | Inclusive scan | $y_i=x_0\oplus\cdots\oplus x_i$ | $y_0=x_0$; shape preserved | `inclusive_scan_kernel` |
| **15.5** | Exclusive scan | $y_i=x_0\oplus\cdots\oplus x_{i-1}$ | $y_0=e$; $e_i=y_{i-1}$ | `exclusive_scan_kernel` |
| **15.6** | Parallel scan | $y_i^{(d)}=y_i^{(d/2)}\oplus y_{i-d}^{(d/2)}$ | depth $\lceil\log_2 N\rceil$ | `hillis_steele_kernel` |
| **15.7** | Reduction vs scan | $r=\bigoplus_j x_j$ vs $y_i=\bigoplus_{j\le i}x_j$ | reduction removes the axis, scan keeps it | `reduce_scan_kernel` |
| **15.8** | Applications of scan | positions, segments, recurrence | same shape; tuple states | `segmented_scan_kernel`, `compact_kernel`, `recurrence_kernel` |

**Applied examples.**

- Segmented scan gives sums $(1,3,6,4,9,15)$ and positions $(0,1,2,0,1,2)$.
- Compaction of `1 0 1 0 1 1` gives $(1,3,5,6)$.
- The recurrence $y_i=a_i y_{i-1}+b_i$ with $a=(1,2,3,4,5)$ and $b=\mathbf 1$ gives $(1,3,10,41,206)$.

## Work and depth

- **Sequential loop.** $W_{\text{seq}}=N-1$ and $D_{\text{seq}}=N-1$.
- **Hillis–Steele.** $D_{\text{HS}}=\lceil\log_2 N\rceil$ and $W_{\text{HS}}=\sum_{k=0}^{m-1}(N-2^k)=mN-(2^m-1)$, with $m=\lceil\log_2 N\rceil$; counting all lanes gives $W_{\text{HS}}\le mN$.
- **Blelloch.** $W_{\text{Blelloch}}=2(N-1)$ and $D_{\text{Blelloch}}=2\lceil\log_2 N\rceil$.

| $N$ | loop $W/D$ | Hillis–Steele $W/D$ | Blelloch $W/D$ |
|---:|---:|---:|---:|
| 5 | 4 / 4 | 8 / 3 | 8 / 6 |
| 8 | 7 / 7 | 17 / 3 | 14 / 6 |
| 1024 | 1023 / 1023 | 9217 / 10 | 2046 / 20 |

- **Parallelism.** It is $W/D$: Hillis–Steele exposes about $N$ independent operations, Blelloch about $N/2$; a machine with $P$ lanes cannot beat $S(P)\le\min(P,W/D)$.
- **Memory.** A scan reads and writes each element once, $Q\approx2Nb$ bytes; at large $N$ it is bandwidth-bound.

## Pitfalls

| Pitfall | Symptom | Fix |
|---|---|---|
| inclusive vs exclusive | offsets are one element late or early | write $e_i=y_{i-1}$ explicitly |
| wrong axis | row-wise scan becomes column-wise | pass `axis=1`; assert a known entry |
| non-associative combine | tree and loop disagree | prove associativity before the scan |
| float order | last-bit difference from `torch.cumsum` | `assert_close(atol=1e-5)`, not `equal` |
| masked lanes | a padded lane wins a max or pollutes a sum | `other=0.0`, `1.0`, `-inf`, `+inf` |
| identity element | exclusive slot zero is garbage | shift with `other=e` |
| overflow | `int32`/`float16` prefix wraps or saturates | accumulate in `int64`/`float32` |
| block boundary | each block restarts at zero | scan block totals and add carries |
| scan for a total | needless work and memory | use `tl.sum` when only the last value is used |
| compaction order | equal keys appear permuted | keep the exclusive offsets in input order |

## How to run

```bash
cd part6_reductions/ch15_scans
jupyter notebook 15_complete.ipynb
```

- **CPU:** every numerical cell and the three CPU tests run; the nine Triton tests print `SKIP <name>: ...` and the notebook continues.
- **GPU:** every kernel is compared with its PyTorch reference.
- **Interpreter:**

```bash
cd 00_Basic
TRITON_INTERPRET=1 PYTHONPATH=$PWD python tools/interpreter_smoke.py
```

`tl.cumsum` and `tl.associative_scan` both run under the interpreter in Triton 3.7. The manual Hillis–Steele kernel is the fallback path and the pedagogical view of the same tree.

## Study checklist

- [ ] Write the inclusive and exclusive scans for $(3,1,4,1,5)$ by hand.
- [ ] State the output shape of a scan and of a reduction on $(R,C)$.
- [ ] Explain why `inclusive - x` works for sum but not for max.
- [ ] Run Hillis–Steele for three rounds on the spine and count the adds.
- [ ] Break the exclusive shift mask and explain which lane is wrong.
- [ ] Change the 2-D kernel from `AXIS=1` to `AXIS=0` and find the first differing entry.
- [ ] Write a chunked global scan for $N$ larger than one block.
- [ ] Derive the affine combine from $T_i(y)=a_i y+b_i$.

**Where this leads.** Chapter 16 composes reductions and scans into stable softmax, log-sum-exp, LayerNorm statistics, and online softmax.

## Exercises

1. **Exclusive max.** Write an exclusive running maximum with the shift-and-scan route, and explain why `inclusive - x` fails.
2. **Chunked global scan.** Split $N=1024$ into blocks of 128, scan each block, scan the 8 totals, and add the carries.
3. **Suffix scan.** Use `reverse=True` to compute suffix sums and check that `suffix[0]` is the total.
4. **Product scan.** Implement a cumulative product with identity `1.0` and positive inputs.
5. **Running argmax.** Carry `(value, index)` through a tuple scan; break ties by the smaller index.
6. **Compaction stability.** Run compaction on duplicate keys and show that the output follows the input order.
7. **Nonzero initial state.** Modify the recurrence kernel for $y_{-1}=5$; the answer is `B_i + A_i * 5`.
8. **Work count.** For $N=1000$, compute both work/depth pairs and the parallelism of each.
9. **Float reassociation.** Find three `float32` values whose left-to-right sum differs from a balanced sum.
10. **Segment mean.** Extend the segmented scan to carry `(sum, count)` and produce per-segment means.
