# Docs — Layer Normalization (deep dive)

Expands the folder README with RMSNorm, backward sketch, block-size rules, and exercises. Visuals live in [`images/`](images/) and [`visual-guide.md`](visual-guide.md).

---

## 1. Why fuse?

Each separate PyTorch op on a row of length $N$ tends to read/write $\Theta(N)$ from HBM. Layer-norm needs at least mean, variance, and an affine map — fusion collapses that into **one** load of $x$ and **one** store of $y$ (plus small $\mu,r$ buffers).

For $X\in\mathbb{R}^{M\times N}$, launch $M$ programs (one per row). Work per program is $O(N)$; total FLOPs stay $O(MN)$, but **bytes moved** drop.

---

## 2. RMSNorm (cousin)

Many LLMs use **RMSNorm** (no mean subtraction):

$$
\operatorname{RMS}(x)=\sqrt{\frac{1}{N}\sum_j x_j^{2}+\varepsilon}, \qquad
y_j = w_j\,\frac{x_j}{\operatorname{RMS}(x)}.
$$

Fewer reductions → slightly cheaper; same fusion ideas apply (one program per row, masked block, FP32 stats).

---

## 3. Backward (sketch)

Given $\partial L/\partial y$, gradients w.r.t. $x,w,b$ involve the saved $\mu$ and $r=\bigl(\sigma^{2}+\varepsilon\bigr)^{-1/2}$. With $\hat{x}=r(x-\mu\mathbf{1})$,

$$
\frac{\partial L}{\partial w_j} = \sum_{\text{rows}} \frac{\partial L}{\partial y_j}\,\hat{x}_j,
\qquad
\frac{\partial L}{\partial b_j} = \sum_{\text{rows}} \frac{\partial L}{\partial y_j},
$$

while $\partial L/\partial x$ mixes the centered residuals of $\partial L/\partial y \odot w$ — again reductions over $N$. Fuse those when writing a training kernel.

---

## 4. Block size

$$
B = \operatorname{next\_power\_of\_2}(N), \quad \text{mask } j < N.
$$

Teaching sizes: $N\in\{8,16,255,1024\}$. Odd $N$ (e.g. 255) specifically exercises the mask path.

If $N$ exceeds SRAM budget, use multi-block reductions (advanced).

---

## 5. Numerics identity

$$
\sigma^{2}
= \frac{1}{N}\sum_j (x_j-\mu)^{2}
= \Bigl(\frac{1}{N}\sum_j x_j^{2}\Bigr) - \mu^{2}.
$$

Centered form is often stabler in practice; either way, accumulate in FP32.

---

## 6. Exercises

1. Derive $\partial y_j / \partial x_i$ for layer-norm (or look up and re-derive $\mu,\sigma$ terms).
2. Implement RMSNorm as a one-file Triton kernel mirroring `_layer_norm_fwd_kernel`.
3. Bench vs `torch.nn.functional.layer_norm` for $N\in\{512,2048,8192\}$.
4. Verify centering: add a constant to a row and check $\hat{x}$ is unchanged up to tolerance.

---

## See also

- Visual guide: [`visual-guide.md`](visual-guide.md)
- Notebook: [`../05_layer_normalization.ipynb`](../05_layer_normalization.ipynb)
- Official: [05-layer-norm](https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html)
