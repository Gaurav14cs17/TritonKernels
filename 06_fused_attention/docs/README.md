# Docs — Fused Attention (deep dive)

Expands the folder README with complexity, a clearer online-softmax proof sketch, causal masking, and exercises. Visuals live in [`images/`](images/) and [`visual-guide.md`](visual-guide.md).

---

## 1. Complexity

| Method | Memory (activations) | Time (approx.) |
|--------|----------------------|----------------|
| Materialize $S$ | $O(N^{2})$ | $O(N^{2}d)$ |
| Flash / fused | $O(Nd)$ | $O(N^{2}d)$ (same FLOPs, better IO) |

Speedups come from **IO**: fewer HBM round-trips, not fewer FLOPs.

---

## 2. Online softmax — update equations

Fix one query row (or a $B_M$-row tile). Scores arrive in blocks $s^{(1)},s^{(2)},\ldots$ with corresponding value blocks $V^{(t)}$.

**State.** Running max $m$, sum-of-exps $\ell$, accumulator $\mathrm{Acc}$ (same row width as $V$).

**Initialization.**

$$
m^{(0)} = -\infty,\qquad \ell^{(0)} = 0,\qquad \mathrm{Acc}^{(0)} = 0.
$$

**Update at block $t$** (row-wise; $\max_j$ / $\sum_j$ over the block’s key axis):

$$
\begin{aligned}
m^{(t)}
&= \max\bigl(m^{(t-1)},\, \max_j s^{(t)}_{j}\bigr), \\[4pt]
\ell^{(t)}
&= e^{m^{(t-1)}-m^{(t)}}\,\ell^{(t-1)}
  + \sum_j e^{s^{(t)}_{j}-m^{(t)}}, \\[4pt]
\mathrm{Acc}^{(t)}
&= e^{m^{(t-1)}-m^{(t)}}\,\mathrm{Acc}^{(t-1)}
  + \sum_j e^{s^{(t)}_{j}-m^{(t)}}\, V^{(t)}_{j,:}.
\end{aligned}
$$

**Finalize.**

$$
o = \frac{\mathrm{Acc}^{(T)}}{\ell^{(T)}}.
$$

**Correctness idea.** Softmax only depends on $\max$ and $\sum e^{s-\max}$. When a new block raises the max, the factor $e^{m_{\mathrm{old}}-m_{\mathrm{new}}}$ renormalizes previous mass and the weighted $V$ sum to the new max — identical to softmax over the concatenation of all blocks (up to FP error). This is the same shift-invariance used in stable softmax.

---

## 3. Causal attention

$$
S_{ij} =
\begin{cases}
q_i^{\top}k_j / \sqrt{d} & j \le i, \\
-\infty & j > i.
\end{cases}
$$

In tiling, skip or mask $K$-tiles entirely beyond the diagonal when possible.

---

## 4. Tile grid intuition

With block sizes $B_M,B_N$ along sequence length:

$$
\#\{Q\text{-tiles}\} = \lceil N/B_M\rceil,
\qquad
\#\{K/V\text{-tiles per }Q\text{-tile}\} = \lceil N/B_N\rceil.
$$

Scratch per program is dominated by $B_M\times B_N$ scores plus $B_M\times d$ accumulators — independent of $N^{2}$.

---

## 5. Exercises

1. Implement PyTorch reference `softmax(QKᵀ/√d) @ V` and match a tiled loop in pure PyTorch (no Triton) to internalize online softmax — compare to `blocked_attention_pseudo`.
2. Add causal masking to the tiled loop; verify against a masked full-$S$ baseline.
3. Log $(m,\ell)$ after each block on a tiny $N=8$ example and check the final $o$ against naive attention.
4. Read the official Triton 06 notebook for the production-grade kernel.

---

## See also

- Visual guide: [`visual-guide.md`](visual-guide.md)
- Notebook: [`../06_fused_attention.ipynb`](../06_fused_attention.ipynb)
- Official: [06-fused-attention](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html)
