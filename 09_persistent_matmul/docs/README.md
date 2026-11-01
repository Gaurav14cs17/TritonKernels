# Docs — Persistent Matmul (deep dive)

Expands the folder README with launch-cost models, tile maps, and exercises. Visuals live in [`images/`](images/) and [`visual-guide.md`](visual-guide.md). Builds on [`../../03_matrix_multiplication/docs/README.md`](../../03_matrix_multiplication/docs/README.md).

---

## 1. Launch overhead model

If launch cost is $L$ seconds and compute is $t_{\text{comp}}$,

$$
t_{\text{classic}} \approx L + t_{\text{comp}}, \qquad
t_{\text{persist}} \approx L' + t_{\text{comp}}
$$

with $L' \ll L$ when many tiles share one launch (and waves are fuller). For huge tiles, $t_{\text{comp}}\gg L$ and persistence helps less.

---

## 2. Same $C$, different schedule

**Corollary.** Any covering of the tile grid by programs that each compute the correct block product yields the same $C$. Persistence changes **who** runs which tile and **when**, not the bilinear map $(A,B)\mapsto AB$.

Static stride with $P$ programs:

$$
\bigcup_{p=0}^{P-1}\{p+kP:k\ge 0,\,p+kP<T\}=\{0,\ldots,T-1\}.
$$

Atomic queue: programs race on a global counter; still a partition of $\{0,\ldots,T-1\}$ if each claimed id is unique and stops at $T$.

---

## 3. Tile → $(p_m, p_n)$

Row-major (classic):

$$
p_m = \left\lfloor \frac{\text{tile\_id}}{G_N} \right\rfloor, \quad
p_n = \text{tile\_id} \bmod G_N.
$$

Combine with **grouped ordering** from tutorial 03 for L2 locality — only the map $\mathrm{tile\_id}\mapsto(p_m,p_n)$ changes; the $K$-loop / `tl.dot` body stays identical.

---

## 4. Choosing $P$

Rule of thumb: $P \sim (\#\text{SMs})\times(\text{desired occupancy})$, capped by $T$. Too small $P$ underuses the GPU; $P=T$ degenerates to classic.

---

## 5. Pitfalls (study)

- Changing $P$ without recomputing coverage ($P>T$ idle; $P=1$ serializes tiles).
- Expecting magic speedups when each tile already dominates runtime.
- Editing the schedule but accidentally changing the tile product (should still match $AB$).

---

## 6. Exercises

1. Instrument number of tiles per program (histogram) for stride vs a simulated atomic queue.
2. Compare classic vs persistent on small $M=N=K=256$ vs large $4096$.
3. Remap `tile_id` with tutorial 03’s grouped formulas; verify `allclose` unchanged.
4. Read the official 09 tutorial kernel end-to-end (work queue + optional TMA / pipelining notes).

---

## See also

- Notebook: [`../09_persistent_matmul.ipynb`](../09_persistent_matmul.ipynb)
- [`../../03_matrix_multiplication/README.md`](../../03_matrix_multiplication/README.md)
- Group GEMM queues: [`../../08_group_gemm/README.md`](../../08_group_gemm/README.md)
- Official: [09-persistent-matmul](https://triton-lang.org/main/getting-started/tutorials/09-persistent-matmul.html)
