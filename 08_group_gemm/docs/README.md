# Docs — Group GEMM (deep dive)

Expands the folder README with scheduling algebra, MoE intuition, and exercises. Visuals live in [`images/`](images/) and [`visual-guide.md`](visual-guide.md). Builds on [`../../03_matrix_multiplication/docs/README.md`](../../03_matrix_multiplication/docs/README.md).

---

## 1. Independence of problems

For each $g$,

$$
C^{(g)}_{mn} = \sum_{k=0}^{K_g-1} A^{(g)}_{mk}\, B^{(g)}_{kn}.
$$

**Proposition.** There is no coupling between groups: the map

$$
\bigl((A^{(g)},B^{(g)})\bigr)_{g=1}^{G}
\;\mapsto\;
\bigl(C^{(g)}\bigr)_{g=1}^{G}
$$

is the Cartesian product of the individual bilinear maps. Scheduling (one grid vs $G$ grids) does not change exact arithmetic $C^{(g)}$.

---

## 2. Why not `torch.bmm`?

`torch.bmm` needs identical shapes per batch item. MoE experts often have different token counts $M_g$ → group GEMM or padded batch + masks (wasteful FLOPs and memory).

Total useful FLOPs remain $\sum_g 2 M_g N_g K_g$; padding to $\max_g M_g$ etc. charges for empty work.

---

## 3. Launch strategies

1. **Loop on host** — simple (this folder’s baseline); launch overhead $\times G$.
2. **Single kernel** — programs read `(A_ptr[g], B_ptr[g], M_g, …)` from device arrays.
3. **Persistent worker** — steal next unfinished group / tile from a queue (related to tutorial 09).

Pointer-table sketch (host → device):

$$
\texttt{A\_ptrs}[g],\;
\texttt{B\_ptrs}[g],\;
\texttt{C\_ptrs}[g],\;
M[g],\; N[g],\; K[g].
$$

Each program still runs the tutorial-03 tile body once it knows which $(g, p_m, p_n)$ it owns.

---

## 4. Relation to tutorial 03 “grouped” ordering

| Term | Meaning |
|------|---------|
| Grouped pid (03) | L2-friendly swizzle of tiles **inside one** $C=AB$ |
| Group GEMM (08) | Many independent $C_g=A_g B_g$, possibly different shapes |

Same `tl.dot` / K-panel math; different **problem packing**.

---

## 5. Pitfalls (study)

- Saying “grouped” without saying **03 L2 swizzle** vs **08 multi-GEMM**.
- Treating the sequential Triton loop in this folder as the finished one-launch Group GEMM.
- Padding all experts to $\max M_g$ and forgetting you still pay for empty FLOPs.

---

## 6. Exercises

1. Time sequential launches vs one fused grouped kernel (official tutorial).
2. Simulate MoE: random $M_g$, fixed $K,N$; plot utilization vs padding to $\max M_g$.
3. Add FP16 autotune configs per size bucket (small / medium / large $M_g N_g K_g$).
4. Sketch a persistent queue over groups (preview of 09) and argue when it beats a static grid.

---

## See also

- Notebook: [`../08_group_gemm.ipynb`](../08_group_gemm.ipynb)
- Matmul primer: [`../../03_matrix_multiplication/README.md`](../../03_matrix_multiplication/README.md)
- Official: [08-group-gemm](https://triton-lang.org/main/getting-started/tutorials/08-group-gemm.html)
