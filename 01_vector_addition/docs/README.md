# Docs — Vector Addition (deep dive)

Expands the folder README with memory-hierarchy intuition, masking lemmas, and exercises. Visuals live in [`images/`](images/) and [`visual-guide.md`](visual-guide.md).

---

## 1. Why not “one program per element”?

GPUs run **warps** of 32 threads. Triton abstracts that: you program **blocks of elements** and the compiler maps them onto warps.

For vector add, a good mental model is:

$$
\begin{aligned}
&\text{DRAM} \xrightarrow{\text{coalesced load}} \text{registers} \\
&\qquad\qquad\qquad\qquad \downarrow \;\; x_i + y_i \\
&\text{DRAM} \xleftarrow{\text{coalesced store}} \text{registers}
\end{aligned}
$$

No shared memory is needed: there is no reuse across elements. Naive grid size $N$ still hurts launch overhead and occupancy bookkeeping — blocking amortizes that.

---

## 2. Masking formula

Let $B = 2^k$ (powers of two map cleanly to Triton shapes). For program `pid`:

$$
o_j = \texttt{pid}\cdot B + j, \quad j\in\{0,\ldots,B-1\},
$$

$$
\text{mask}_j = \mathbf{1}[o_j < N].
$$

When $N \bmod B = 0$, every mask bit is 1. When not, the last program has trailing zeros in the mask — those lanes **must not** touch memory.

**Teaching walk** ($N=5$, $B=4$):

| pid | $o_j$ | mask | valid indices |
|-----|---------|------|---------------|
| 0 | $0,1,2,3$ | $1,1,1,1$ | $0..3$ |
| 1 | $4,5,6,7$ | $1,0,0,0$ | $4$ only |

**Lemma (partition).** $\bigcup_p T_p = \{0,\ldots,N-1\}$ with $T_p = \{pB+j : j<B\}\cap\{0,\ldots,N-1\}$ pairwise disjoint.

---

## 3. Roofline (qualitative)

Peak DRAM bandwidth $B_{\text{peak}}$ (e.g. ~2 TB/s on high-end GPUs) caps performance:

$$
t_{\min} \approx \frac{3\,N\,b}{B_{\text{peak}}}.
$$

If your measured $t$ is close to $t_{\min}$, the Triton kernel is near hardware limit — matching PyTorch is expected for this op.

GB/s from a measured runtime $t$:

$$
\mathrm{GB/s} = \frac{3\,N\,b}{t\cdot 10^{9}}.
$$

Official bench sizes: $N = 2^{12},\ldots,2^{27}$ (log $x$-axis).

---

## 4. Visual checklist before coding

```
[ ] Choose B (often 1024)
[ ] grid = cdiv(N, B)
[ ] Load with mask
[ ] Add in registers
[ ] Store with same mask
[ ] allclose vs torch (include N % B ≠ 0)
[ ] Bench GB/s vs N (log scale)
```

---

## 5. Exercises

1. **Dtype.** Switch to `float16`. Does GB/s change? (Hint: $b$ halves → less traffic.)
2. **Fused SAXPY.** Implement $z = a\cdot x + y$ for scalar $a$. Intensity: still ~1 FLOP / 3 loads? Count carefully.
3. **2D grid.** Launch with `(grid_x, grid_y)` and map `pid` → linear index yourself.
4. **Remainder stress.** For $N=98432$, list how many full tiles and what the last mask looks like for $B=1024$.

---

## See also

- Visual guide: [`visual-guide.md`](visual-guide.md)
- Notebook: [`../01_vector_addition.ipynb`](../01_vector_addition.ipynb)
- Official: [01-vector-add](https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html)
