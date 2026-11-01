# Docs — Low-Memory Dropout (deep dive)

Expands the folder README with expectation proofs, Philox intuition, and exercises. Visuals live in [`images/`](images/) and [`visual-guide.md`](visual-guide.md).

---

## 1. Expectation identity

With keep-probability $q=1-p$ and $M_i\sim\operatorname{Bernoulli}(q)$,

$$
\mathbb{E}[y_i \mid x_i]
= q\cdot\frac{x_i}{q} + (1-q)\cdot 0
= x_i.
$$

Variance increases ($\operatorname{Var}(y_i\mid x_i)= \frac{p}{1-p}\,x_i^{2}$ for inverted dropout); that is intentional regularization.

---

## 2. Philox (intuition)

Philox is a **counter-based** PRNG: output depends only on $(\text{seed}, \text{offset})$, not on previous draws. Perfect for GPUs where each lane generates independently:

$$
u_i = \operatorname{tl.rand}(\text{seed}, i).
$$

Reference: Salmon et al., “Parallel Random Numbers: As Easy as 1, 2, 3” (2011).

**Reproducibility lemma.** Fix seed $s$. The map $i\mapsto m_i=\mathbf{1}[\operatorname{Philox}(s,i)>p]$ is a deterministic function of $(s,p,i)$. Forward and backward (or recomputed forward under checkpointing) that share $s$ see the same mask without storing it.

---

## 3. Checkpointing benefit

Activation checkpointing recomputes forward. With a stored mask, you must also save/restore RNG state carefully. With a **seed**, recomputation is deterministic by construction — only $O(1)$ extra state beside activations you already keep.

---

## 4. Memory traffic sketch

Let $b$ be bytes per element. Classic apply pass roughly reads $x$ and $m$, writes $y$: $\sim 3Nb$. Fused seed path reads $x$, writes $y$: $\sim 2Nb$, and never allocates the mask buffer (peak memory win under activation checkpointing is often larger than the traffic win).

---

## 5. Exercises

1. Prove $\mathbb{E}[y]=x$ for inverted dropout; derive $\operatorname{Var}(y_i\mid x_i)$.
2. Extend to a matrix with **one seed per row** (offset = `row * N + col`).
3. Measure peak memory: seeded vs materializing `torch.rand_like(x) > p` for large $N$.
4. Show that changing only the seed changes the keep-set, while $p=0$ always yields $y=x$.

---

## See also

- Visual guide: [`visual-guide.md`](visual-guide.md)
- Notebook: [`../04_low_memory_dropout.ipynb`](../04_low_memory_dropout.ipynb)
- Official: [04-low-memory-dropout](https://triton-lang.org/main/getting-started/tutorials/04-low-memory-dropout.html)
