# Docs — Libdevice (deep dive)

Expands the folder README with GELU variants, accuracy/throughput notes, and exercises. Visuals live in [`images/`](images/) and [`visual-guide.md`](visual-guide.md).

---

## 1. Exact GELU and the CDF

Recall $\Phi(x)=\tfrac12\bigl(1+\operatorname{erf}(x/\sqrt{2})\bigr)$. Then

$$
\operatorname{GELU}(x)=x\,\Phi(x)=\tfrac12\,x\bigl(1+\operatorname{erf}(x/\sqrt{2})\bigr).
$$

In Triton, `libdevice.erf` is the CUDA math-library binding — same mathematical map as `math.erf` / `torch.erf`, GPU-tuned implementation inside an elementwise program.

---

## 2. tanh GELU approximation

Another common form (Hendrycks & Gimpel / BERT-era approx):

$$
\operatorname{GELU}_{\tanh}(x)=\tfrac{1}{2}x\left(1+\tanh\!\left(\sqrt{\tfrac{2}{\pi}}\bigl(x+0.044715\,x^{3}\bigr)\right)\right).
$$

Compare abs error vs `torch.nn.functional.gelu` (exact erf path) for both implementations. The tanh form is pure arithmetic (`tl.tanh` or a poly) — no libdevice call.

---

## 3. Accuracy vs throughput

| Regime | Prefer | Why |
|--------|--------|-----|
| Huge $N$, memory-bound | Either; libdevice if you care about ULPs | Extra FLOPs hide behind DRAM |
| Medium $N$, compute-sensitive | Poly / tanh | Fewer expensive special-function calls |
| Need bit-close to `F.gelu` | `libdevice.erf` | Matches the erf definition |

Libdevice calls may cost more **latency per element** than a fused polynomial. For huge tensors that are memory-bound, accuracy often wins “for free”; for tiny tensors, launch and call overhead matter more.

---

## 4. Kernel shape (elementwise)

Same SPMD pattern as tutorial 01:

$$
\text{offsets}(p)=p\cdot B + \{0,\ldots,B-1\},\qquad
m_j = (\text{offsets}_j < N).
$$

Load → cast FP32 → `libdevice.erf` → scale → store. Grid $\lceil N/B\rceil$.

---

## 5. Pitfalls (study)

- `F.gelu(..., approximate='tanh')` vs exact erf — match the reference you claim.
- Tiny-$N$ microbenchmarks are launch-noise; use large $N$ for GB/s conclusions.
- Mixing dtypes without an FP32 math path can hide accuracy bugs.

---

## 6. Exercises

1. Plot $\lvert\operatorname{GELU}_{\text{erf}}-\operatorname{GELU}_{\tanh}\rvert$ on $[-5,5]$.
2. Implement $\operatorname{silu}(x)=x\cdot\sigma(x)$ with `tl.sigmoid` or libdevice if needed; check vs `F.silu`.
3. Bench GB/s: libdevice GELU vs pure-`tl` tanh approx for $N\in\{2^{14},2^{20},2^{24}\}$.
4. Swap `libdevice.erf` for a 3-term Taylor of erf near 0 and measure where the approximation breaks.

---

## See also

- Notebook: [`../07_libdevice.ipynb`](../07_libdevice.ipynb)
- Official: [07-libdevice](https://triton-lang.org/main/getting-started/tutorials/07-libdevice.html)
- Extern-functions slug: [07-extern-functions](https://triton-lang.org/main/getting-started/tutorials/07-extern-functions.html)
