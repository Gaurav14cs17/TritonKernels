# Docs — Block-Scaled Matmul (deep dive)

Expands the folder README with quantization error, MX intuition, and exercises. Visuals live in [`images/`](images/) and [`visual-guide.md`](visual-guide.md). Builds on [`../../03_matrix_multiplication/docs/README.md`](../../03_matrix_multiplication/docs/README.md).

---

## 1. Scalar homogeneity

For $s_A,s_B\in\mathbb{R}$,

$$
(s_A A)(s_B B) = s_A s_B\, (AB).
$$

All three notebook paths (`naive_block_scaled_matmul_pseudo`, `block_scaled_matmul_ref`, `block_scaled_matmul_triton`) realize this identity; FP differences come only from rounding order.

---

## 2. Quantization error (toy model)

If true value $a = s\cdot q + e$ with quantization error $e$, products accumulate error in $C$. Block scaling reduces $|e|$ by choosing $s$ to fit the block’s dynamic range:

$$
s = \frac{\max_{(i,j)\in\mathcal{B}} |A_{ij}|}{\max|\text{representable FP8}|}.
$$

Per-tensor $s$ is forced by the global max (outliers hurt); per-block $s$ adapts locally.

---

## 3. Block contribution in the $K$-loop

With scales $s^{A}_{p_m,t}$ and $s^{B}_{t,p_n}$ on $K$-panel $t$,

$$
C_{p_m,p_n}
=
\sum_t
\bigl(s^{A}_{p_m,t} A_{p_m,t}\bigr)
\bigl(s^{B}_{t,p_n} B_{t,p_n}\bigr)
=
\sum_t
\bigl(s^{A}_{p_m,t}\, s^{B}_{t,p_n}\bigr)\,
\bigl(A_{p_m,t} B_{t,p_n}\bigr)
$$

in exact arithmetic when scales are scalars per panel. Hardware may apply scales inside the MMA instead of forming explicit FP16 tiles.

---

## 4. Relation to MX formats

Microscaling (MX) formats pair low-precision payloads with shared exponents / scales over small blocks (e.g. 32 elements). Tutorial 10 shows how Triton expresses those layouts in kernels — metadata tensors indexed beside the usual $A,B$ pointers from tutorial 03.

---

## 5. Pitfalls (study)

- Treating scalar $s_A,s_B$ demos as full MX/FP8 layout support.
- Forgetting scales must be indexed **with** $K$-panels in the blocked loop.
- Claiming bit-exact match to FP32 $AB$ after FP8 quant — measure relative error instead.

---

## 6. Exercises

1. Simulate FP8 cast + per-block scale in PyTorch; plot $\|AB - A_{\text{q}}B_{\text{q}}\|_F / \|AB\|_F$.
2. Compare per-tensor vs per-block scales on a matrix with one large outlier.
3. Prove the scalar identity algebraically; then break it deliberately with per-row scales that do **not** factor as $s_A s_B$.
4. Study the official 10 tutorial for hardware-specific FP8 MMA APIs.

---

## See also

- Notebook: [`../10_block_scaled_matmul.ipynb`](../10_block_scaled_matmul.ipynb)
- [`../../03_matrix_multiplication/README.md`](../../03_matrix_multiplication/README.md)
- Official: [10-block-scaled-matmul](https://triton-lang.org/main/getting-started/tutorials/10-block-scaled-matmul.html)
