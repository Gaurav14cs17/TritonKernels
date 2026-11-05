# 16 — Combining Basic Operations

> **Goal.** Read a normalization as a small pipeline of reductions, then build
> it one program per row: sum of squares, mean and variance, maximum and
> normalization, stable softmax, log-sum-exp, online softmax, Welford variance,
> and LayerNorm statistics.

Notebook: [`16_complete.ipynb`](16_complete.ipynb) · Helpers:
[`../../ttools.py`](../../ttools.py) · [Setup](../../docs/SETUP.md) ·
[Notation](../../docs/NOTATION.md) · [Glossary](../../docs/GLOSSARY.md) ·
[API index](../../docs/API_INDEX.md) · [Formula index](../../docs/FORMULA_INDEX.md) ·
[Diagrams](../../docs/DIAGRAMS.md) · [Troubleshooting](../../docs/TROUBLESHOOTING.md) ·
[Exercise hints](../../docs/SOLUTIONS.md)

[Index](../../00_index.ipynb) · [← Previous](../../part6_reductions/ch15_scans/15_complete.ipynb) · [Next →](../../part7_construction/ch17_control_flow/17_complete.ipynb)

**Estimated time:** 90–120 min · **Difficulty:** Intermediate–Advanced

**Thread.**

- **Recurring idea:** every recipe is a reduction followed by a normalization; every stability fix is either a shift or a better merge.
- **Topic chain:** sum of squares → mean/variance → max/normalize → stable softmax → log-sum-exp → online softmax → Welford variance → LayerNorm.
- **Shape thread:** one row per program; reduce the last axis; return one scalar or one row per row.
- **Stability thread:** subtract a maximum before `exp`; keep epsilon inside `rsqrt`; merge statistics instead of subtracting large squares.

## Learning objectives

- Combine two or more reductions into a recipe and state the order of the reductions.
- Derive the stable-softmax shift and the online max/denominator update.
- Derive the Welford mean and $M_2$ merge formulas.
- Write one-program-per-row Triton kernels for all four recipes.
- Apply the **shape rule** (reduce the last axis) and the **stability rule** (shift before `exp`, epsilon inside `rsqrt`).
- Choose between two-pass and online schedules by counting bytes.

## Study ladder

1. **Method 01 — scalar reference.** One row and one element at a time, using the formulas as written. It is the correctness oracle.
2. **Method 02 — row-wise Triton kernels.** One program per row for sum-of-squares/variance, stable softmax, log-sum-exp, and LayerNorm stats.
3. **Method 03 — online and fused.** Online softmax carries $(m,s)$; Welford carries $(n,\mu,M_2)$; a fused LayerNorm writes the normalized row once.

**Study tip.**

- Take $x=(-1,0,1,2)$ and compute the mean, variance, softmax, log-sum-exp, and LayerNorm by hand.
- Redo the softmax for $x+1000$: the result is identical.
- Redo it without the shift: the naive exponentials overflow.

## Mathematical definition

- **Row:** $x=(x_0,\ldots,x_{N-1})\in\mathbb R^N$.
- **Sum of squares:** $s=\sum_jx_j^2$ and $\|x\|_2=\sqrt{s}$.
- **Mean and variance:** $\mu=\frac1N\sum_jx_j$ and $\sigma^2=\frac1N\sum_j(x_j-\mu)^2$.
- **Maximum and normalization:** $m=\max_jx_j$ and $y_j=x_j/m$ for $m\ne0$.
- **Stable softmax:** $p_j=\frac{e^{x_j}}{\sum_ke^{x_k}}=\frac{e^{x_j-m}}{\sum_ke^{x_k-m}}$ for any finite shift, with $m=\max_kx_k$.
- **Why the shift works:** every exponent is $\le0$, the largest term is $1$, and the denominator is in $[1,N]$.
- **LogSumExp:** $\operatorname{LSE}(x)=m+\ln\sum_je^{x_j-m}$ and $\log p_j=x_j-\operatorname{LSE}(x)$.
- **Online softmax:** $m'=\max(m,m_B)$ and $s'=s\,e^{m-m'}+s_B\,e^{m_B-m'}$.
- **Welford merge:** $\delta=\mu_B-\mu$, $n'=n+n_B$, $\mu'=\mu+\delta\frac{n_B}{n'}$, and $M_2'=M_2+M_{2,B}+\delta^2\frac{n\,n_B}{n'}$.
- **LayerNorm:** $\hat x_j=(x_j-\mu)\operatorname{rsqrt}(\sigma^2+\epsilon)$ and $y_j=\gamma_j\hat x_j+\beta_j$; input $[M,N]$, stats $[M]$, parameters $[N]$.

## Connections

- **16.1 → 16.2:** the sum of squares is the first term of the one-pass variance identity.
- **16.2 → 16.3:** the row statistics become the scale for max normalization.
- **16.3 → 16.4:** the maximum becomes the softmax shift.
- **16.4 → 16.5:** the shifted denominator becomes the log-sum-exp argument.
- **16.5 → 16.6:** the same shifted sum becomes loop-carried online state.
- **16.2 + 16.6 → 16.7:** Welford applies the chunk-merge idea to mean and variance.
- **16.2 + 16.7 → 16.8:** LayerNorm consumes the mean and the stable variance.
- **16.8 → fused kernel:** the shape and epsilon rules become the fused LayerNorm contract.

## Worked example

- **Spine:** $x=(-1,0,1,2)$, $N=4$, $\epsilon=10^{-5}$.
- **Sum of squares:** $s=1+0+1+4=6$, $\|x\|_2=\sqrt6=2.449490$.
- **Mean:** $\mu=2/4=0.5$ and $x-\mu=(-1.5,-0.5,0.5,1.5)$.
- **Variance:** $\sum_j(x_j-\mu)^2=5$, so $\sigma^2=5/4=1.25$ and $\sigma=1.118034$.
- **One-pass check:** $6/4-0.5^2=1.25$.
- **Stable softmax:** $m=2$, $e^{x-m}=(0.049787,0.135335,0.367879,1)$, $\sum_je^{x_j-m}=1.553001$, and $p=(0.032059,0.087144,0.236883,0.643914)$.
- **Log-sum-exp:** $\operatorname{LSE}(x)=2+\ln(1.553001)=2.440190$; check $e^{2.440190}=11.475218$.
- **LayerNorm:** $\operatorname{rsqrt}(1.25+10^{-5})=0.894424$ and $\hat x=(-1.341635,-0.447212,0.447212,1.341635)$.
- **Shift:** $x+1000=(999,1000,1001,1002)$ and $m'=1002$.

| $j$ | $x_j$ | $x_j+1000$ | $(x_j+1000)-m'$ | $e^{(x_j+1000)-m'}$ |
|---:|---:|---:|---:|---:|
| 0 | $-1$ | $999$ | $-3$ | $0.049787$ |
| 1 | $0$ | $1000$ | $-2$ | $0.135335$ |
| 2 | $1$ | $1001$ | $-1$ | $0.367879$ |
| 3 | $2$ | $1002$ | $0$ | $1$ |

- **Shift invariance:** $e^{(x_j+1000)-1002}=e^{x_j-2}$, so the denominator and softmax are unchanged.
- **Overflow without the shift:** $e^{999}=e^{1000}=e^{1001}=e^{1002}=\infty$ in `float32`, because `exp` overflows above $88.7228$; the naive ratio is $\infty/\infty=\mathrm{nan}$.
- **Overflow avoided:** with the shift, the largest exponent is $e^0=1$ and the denominator is finite.

## Rules

### Stability rule

- Subtract the maximum before every `exp`; keep the running maximum when the row is chunked.
- The shift is exact; it only changes the reference point.
- `float32` overflows above $88.7228$; `float16` above $11.0899$.

### Axis/shape rule

- A reducing recipe maps $[M,N]\to[M]$; a normalizing recipe maps $[M,N]\to[M,N]$.
- A row tile is 1-D, so its reduction axis is `axis=0`.
- Reducing `axis=1` on an $[M,N]$ tensor gives column statistics, not per-token LayerNorm statistics.

### Online rule

- Compute $m_{\text{new}}=\max(m,m_B)$ before rescaling the old denominator.
- Update $s\leftarrow s\,e^{m-m_{\text{new}}}+s_B\,e^{m_B-m_{\text{new}}}$, then assign $m\leftarrow m_{\text{new}}$.
- Initialize $m=-\infty$, not $0$, so an all-negative row uses the right reference.
- Skip a fully masked chunk ($m_B=-\infty$) before taking $e^{m_B-m_{\text{new}}}$.

### Epsilon rule

- Use $\operatorname{rsqrt}(\sigma^2+\epsilon)$, not $\operatorname{rsqrt}(\sigma^2)+\epsilon$.
- At $\sigma^2=10^{-8}$ and $\epsilon=10^{-5}$, the correct factor is $316.070$; the wrong placement gives $10000.00001$ ($31.6\times$).
- Use the biased variance (divide by $N$) for LayerNorm.
- Do not add epsilon to a softmax denominator unless an empty row is part of the contract.

## Chapter map

| § | Recipe | Formula | Shape/stability rule | Kernel/demo |
|---|---|---|---|---|
| **16.1** | Sum of squares | $s=\sum_jx_j^2$ | one scalar per row; `float32` | `sumsq_var` |
| **16.2** | Mean and variance | $\mu,\ \sigma^2$ centered | last axis; avoid $\overline{x^2}-\mu^2$ | `sumsq_var` |
| **16.3** | Maximum and normalization | $y_j=x_j/\max_kx_k$ | guard $m=0$; `-inf` for masked max | `softmax` |
| **16.4** | Stable softmax | $p_j=e^{x_j-m}/\sum_ke^{x_k-m}$ | $m=\max_kx_k$; exponents $\le0$ | `softmax` |
| **16.5** | LogSumExp | $m+\ln\sum_je^{x_j-m}$ | one scalar per row; shift first | `lse` |
| **16.6** | Online softmax | $m',s'$ update | compute $m'$ before rescaling $s$ | `online_softmax` |
| **16.7** | Welford variance | $(n,\mu,M_2)$ merge | keep $\delta^2nn_B/n'$ | `welford_var` |
| **16.8** | LayerNorm statistics | $(x-\mu)\operatorname{rsqrt}(\sigma^2+\epsilon)$ | stats $[M]$, $\gamma,\beta$ $[N]$ | `ln_stats`, `fused_layernorm` |
| **★** | Fused LayerNorm | $y=\gamma\hat x+\beta$ | one read, one write | `fused_layernorm` |

## How to run

- From the chapter folder:

```bash
cd part6_reductions/ch16_combining_ops
jupyter notebook 16_complete.ipynb
```

- **CPU:** the worked example, the eight Method 01 references, the online and Welford chunk arithmetic, and the stability cell run; kernel tests print `SKIP ...`.
- **Interpreter smoke test without a GPU:**

```bash
cd 00_Basic
TRITON_INTERPRET=1 PYTHONPATH=$PWD python tools/interpreter_smoke.py
```

## Study checklist

- [ ] Compute all five spine quantities for $x=(-1,0,1,2)$ by hand.
- [ ] Prove the softmax shift identity for an arbitrary finite $m$.
- [ ] Derive the online update and state why $m_{\text{new}}$ comes first.
- [ ] Derive the Welford merge and identify the cross term.
- [ ] Explain why $m=-\infty$ is the correct initial maximum.
- [ ] Compare $\operatorname{rsqrt}(\sigma^2+\epsilon)$ with $\operatorname{rsqrt}(\sigma^2)+\epsilon$ at $\sigma^2=10^{-8}$.
- [ ] Count the MiB for three-pass and online softmax at $M=1024,N=4096$.
- [ ] Break the tail mask in one kernel and explain the failure.

## Pitfalls

| Pitfall | What goes wrong | Fix |
|---|---|---|
| Missing max shift | `exp` overflows; `inf/inf = nan` | subtract the row/running max |
| Epsilon outside `rsqrt` | $31.6\times$ scale error at $\sigma^2=10^{-8}$ | `tl.rsqrt(var + eps)` |
| Variance cancellation | $\overline{x^2}-\mu^2$ gives $0$ instead of $1.25$ | centered two-pass or Welford |
| Wrong axis | column stats instead of per-row stats | one program per row; `axis=0` |
| Masked lanes | zeros bias mean/variance | `other=0` for sums, `-inf` for max |
| Online update order | old chunk is dropped | compute $m'$, rescale $s$, then assign $m$ |
| Welford $M_2$ | within-chunk variance only | include the cross term |
| Zero denominator | empty/all-`-inf` row gives `nan` | define the empty-row contract |

## Exercises

1. Repeat the worked example on $x=(-2,-1,0,1)$.
2. Prove the softmax shift identity and explain $m=\max_kx_k$.
3. Merge three Welford chunks in two orders; compare rounding.
4. Replace the two-pass variance with $\overline{x^2}-\mu^2$ at mean $10^4$ and measure the `float32` error.
5. Implement `log_softmax(x) = x - lse(x)` and check it against PyTorch.
6. Change `BLOCK_N` to the next smaller power of two and explain the failure.
7. Add a temperature $T$; state which parts of the kernel change.
8. Fuse a scale $s$ into the softmax output and count the saved traffic.
9. Find the smallest $\epsilon$ that keeps a constant row finite in `float32`.
10. For $M=2048,N=8192$, count the traffic of both schedules.

**Where this leads.** The exercises revisit each contract; the notebook's Part IV turns them into streaming and fused kernels, and Chapter 17 adds data-dependent control flow to the same row-wise recipes.

Hints: [`../../docs/SOLUTIONS.md`](../../docs/SOLUTIONS.md).
