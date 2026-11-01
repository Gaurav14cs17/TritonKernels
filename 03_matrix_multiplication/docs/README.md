# Docs — Matrix Multiplication (deep dive)

Expands the folder README with grouped-scheduling algebra, pointer forms, and exercises. Visuals live in [`images/`](images/) and [`visual-guide.md`](visual-guide.md).

---

## 1. Why not “one program per $C_{mn}$”?

Correct (bijection $\mathrm{pid}\mapsto(m,n)$), but:

- Grid size $MN$ is huge.
- No reuse of $A$-rows / $B$-cols across neighbors.
- Column loads of $B$ are strided in row-major layout.

Blocking gives each program a $B_M\times B_N$ tile and amortizes loads via `tl.dot` on SRAM-friendly tiles.

---

## 2. Tile grid and row-major pid

$$
G_M=\left\lceil\frac{M}{B_M}\right\rceil,\quad
G_N=\left\lceil\frac{N}{B_N}\right\rceil,\quad
\mathrm{pid}\in\{0,\ldots,G_M G_N-1\}.
$$

Row-major:

$$
p_m=\left\lfloor\mathrm{pid}/G_N\right\rfloor,\qquad
p_n=\mathrm{pid}\bmod G_N.
$$

Teaching walk ($M=6,N=8,K=3$, $B_M=3$, $B_N=4$):

| pid | $(p_m,p_n)$ | Writes |
|-----|---------------|--------|
| 0 | (0,0) | $C[0:3,0:4]=A[0:3,:]\,B[:,0:4]$ |
| 1 | (0,1) | $C[0:3,4:8]$ |
| 2 | (1,0) | $C[3:6,0:4]$ |
| 3 | (1,1) | $C[3:6,4:8]$ |

---

## 3. $K$-panels

For large $K$, never materialize the full strip. With $B_K$:

$$
C_{\mathrm{tile}}=\sum_{t=0}^{\lceil K/B_K\rceil-1} A_t B_t,
\quad
A_t\in\mathbb{R}^{B_M\times B_K},\;
B_t\in\mathbb{R}^{B_K\times B_N},
$$

accumulate in FP32, cast on store. Example: $6\times 8@8\times 4$, $B_M=3,B_N=2,B_K=4$ → two `tl.dot` steps per tile.

Pointers advance by $B_K\cdot\texttt{stride}_{ak}$ / $B_K\cdot\texttt{stride}_{bk}$ each iteration; masks handle $K\bmod B_K\neq 0$.

---

## 4. Grouped pid swizzle

Let $G=\texttt{GROUP\_SIZE\_M}$. Inside a group of height $g_m\le G$, walk tiles **column-major**:

$$
\begin{aligned}
\mathrm{num\_pid\_in\_group} &= G\cdot G_N,\\
\mathrm{group\_id} &= \lfloor\mathrm{pid}/(G\,G_N)\rfloor,\\
\mathrm{first\_pid\_m} &= \mathrm{group\_id}\cdot G,\\
g_m &= \min(G_M-\mathrm{first\_pid\_m},\,G),\\
w &= \mathrm{pid}\bmod(G\,G_N),\\
p_m &= \mathrm{first\_pid\_m}+(w\bmod g_m),\\
p_n &= \lfloor w/g_m\rfloor.
\end{aligned}
$$

Example $G=2$, $G_N=2$: pid order on the $2\times 2$ tile grid is

$$
\begin{bmatrix}0&2\\1&3\end{bmatrix}
\quad\text{instead of row-major}\quad
\begin{bmatrix}0&1\\2&3\end{bmatrix}.
$$

**Theorem (same $C$).** The map $\mathrm{pid}\mapsto(p_m,p_n)$ is still a bijection onto the tile grid; only schedule changes. Kernel body after computing $(p_m,p_n)$ is identical to blocked.

**Cache intuition (notes).** Writing 9 tiles with group size 3: row-major ≈ $9$ A + $81$ B panel loads vs grouped ≈ $27$ A + $27$ B — fewer unique $B$ reloads when vertical neighbors share a column panel.

---

## 5. Autotune knobs

Common configs: `BLOCK_SIZE_M/N/K`, `GROUP_SIZE_M` (often 8), `num_warps`, `num_stages`. Prefer FP16 inputs + FP32 accumulator for throughput demos vs cuBLAS.

---

## 6. Exercises

1. Reproduce the note’s pid visual matrix for $G_M=G_N=4$, $G=2$ and print the $4\times 4$ pid grid.
2. Count theoretical A/B panel loads for row-major vs grouped on an $8\times 8$ tile grid, $G=4$.
3. Change only the pid map in `blocked_matmul_kernel` to the grouped formulas and verify `allclose` vs torch.
4. Bench TFLOPS for $M=N=K\in\{512,1024,2048\}$ with and without grouping.

---

## See also

- Notebook: [`../03_matrix_multiplication.ipynb`](../03_matrix_multiplication.ipynb)
- Official: [03-matrix-multiplication](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html)
