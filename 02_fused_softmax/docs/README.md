# Docs — Fused Softmax (deep dive)

Expands the folder README with traffic algebra, padding rules, occupancy, and exercises. Visuals live in [`images/`](images/) and [`visual-guide.md`](visual-guide.md).

---

## 1. Softmax as a probability map

For logits $x \in \mathbb{R}^{N}$,

$$
p_j = \frac{e^{x_j}}{\sum_k e^{x_k}} \in (0,1), \qquad \sum_j p_j = 1.
$$

**Shift invariance.** $\operatorname{softmax}(x+c)=\operatorname{softmax}(x)$ for any scalar $c$. Choosing $c=\max_j x_j$ keeps exponents $\le 1$ and avoids overflow.

In transformers, $x$ is often a row of attention scores $QK^\top / \sqrt{d}$. The same fusion idea appears again in FlashAttention (tutorial 06).

---

## 2. Exact traffic model (naive)

Let $M$ rows, $N$ cols, element size $b$ bytes. Naive multi-kernel traffic (approx.):

$$
T_{\text{naive}} \approx (8MN + 4M)\,b \quad\text{(bytes)},
$$

fused:

$$
T_{\text{fused}} \approx 2MN\,b.
$$

Ratio:

$$
\frac{T_{\text{naive}}}{T_{\text{fused}}} \approx 4 + \frac{2}{N} \xrightarrow{N\to\infty} 4.
$$

**Teaching walk.** For $M=32$, $N=64$, $b=4$: fused $\approx 2\cdot 32\cdot 64\cdot 4 = 16\,\mathrm{KiB}$ of traffic vs $\sim 4\times$ that for naive (ignoring tiny $M$-vectors).

---

## 3. Power-of-two padding

Triton block shapes are typically powers of two. For width $N$,

$$
B = 2^{\lceil\log_2 N\rceil}.
$$

Load:

$$
\text{row}_j =
\begin{cases}
X_{i,j} & j < N \\
-\infty & N \le j < B
\end{cases}
$$

so $\max$ ignores padding and $\exp(-\infty)=0$ contributes nothing to the sum.

**Example.** $N=5$ → $B=8$; lanes $5,6,7$ are $-\infty$.

---

## 4. Persistent row schedule

With $P=\texttt{num\_programs}$ programs,

$$
\text{rows}(\mathrm{pid}) = \{\mathrm{pid},\;\mathrm{pid}+P,\;\mathrm{pid}+2P,\;\ldots\}\cap\{0,\ldots,M-1\}.
$$

Each program reuses the same compiled block body across many rows — fewer launches than “one program per row” when $M$ is large.

---

## 5. Occupancy sketch

After compiling, register count $R$ and shared memory $S$ per program bound how many programs fit per SM:

$$
\text{occupancy} \le \min\left(\left\lfloor\frac{\text{regs/SM}}{R\cdot\text{warps}}\right\rfloor,
\left\lfloor\frac{\text{smem/SM}}{S}\right\rfloor\right).
$$

`num_programs ≈ #SM × occupancy`, capped by $M$.

---

## 6. Exercises

1. Prove $\operatorname{softmax}(x+c)=\operatorname{softmax}(x)$.
2. Implement column-wise softmax (operate on $X^\top$ or change strides).
3. Compare `tl.exp` vs a more accurate path if available — plot max abs error vs `torch.softmax`.
4. For $N=1000$, compute $B=\operatorname{next\_power\_of\_2}(N)$ and count how many lanes are $-\infty$ padding.

---

## See also

- Visual guide: [`visual-guide.md`](visual-guide.md)
- Notebook: [`../02_fused_softmax.ipynb`](../02_fused_softmax.ipynb)
- Official: [02-fused-softmax](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html)
