# Docs — Embedding Forward and Backward (deep dive)

Expands the folder README with proofs, the atomic argument, bf16 details, the
cost model, and exercises. Visuals live in [`images/`](images/) and
[`visual-guide.md`](visual-guide.md).

---

## 1. One-hot equivalence

Let $P\in\{0,1\}^{M\times V}$ with $P_{m,j}=1$ iff $j=I_m$. Then

$$
(PW)_{m,d}
=\sum_{j=0}^{V-1}P_{m,j}W_{j,d}
=W_{I_m,d}
=(W[I])_{m,d}.
$$

So the gather and the dense product are the same function. The dense form is
useful for the derivative; the gather is what the kernel executes (it never
materializes the $M\times V$ matrix).

---

## 2. Backward by the adjoint (trace) trick

Write the loss differential as

$$
dL=\langle G,dX\rangle,
\qquad
G=\frac{\partial L}{\partial X}.
$$

Since $X=PW$ and $P$ is constant with respect to the parameters,

$$
dX=P\,dW.
$$

Substitute:

$$
dL=\langle G,P\,dW\rangle
=\operatorname{tr}\!\left(G^\top P\,dW\right)
=\operatorname{tr}\!\left((P^\top G)^\top dW\right)
=\langle P^\top G,dW\rangle.
$$

Therefore

$$
\boxed{\,dW=P^\top G\,}
$$

and componentwise

$$
dW_{j,:}=\sum_{m=0}^{M-1}P_{m,j}G_{m,:}
=\sum_{m:\,I_m=j}G_{m,:}.
$$

This is exactly the scatter-add the kernel performs with atomics.

---

## 3. Why `tl.atomic_add` is required

The parallel schedule partitions tokens, not vocabulary rows. If token $m$ and
token $m'$ share an id $j$, two different programs write to `grad_weight[j]`.
A load-add-store sequence is not atomic:

```
program A: tmp = dW[j]; tmp += G[m];  dW[j] = tmp
program B: tmp = dW[j]; tmp += G[m']; dW[j] = tmp
```

Interleaving can lose one update. `tl.atomic_add` performs the read-modify-write
as one atomic operation per element, so both contributions survive.

**Consequences**

- Correct sum, but **nondeterministic order** → tiny run-to-run differences.
- Contention: if many tokens share one row, atomics serialize on that row.
  Real embedding layers often use a row-wise reduction or a sorted segmented
  scan to avoid this.
- The interpreter and some older devices do not support bf16 atomics; the
  wrapper promotes to fp32.

---

## 4. bf16 accumulation

bf16 has 8 exponent bits and only 7 mantissa bits. Accumulating many gradient
rows in bf16 loses precision quickly, and hardware bf16 atomics are slow or
missing. The notes therefore accumulate in fp32:

```python
if dtype == torch.bfloat16:
    grad_out = grad_out.float()          # promote
    grad_weight = torch.zeros(...)       # fp32 accumulator
    kernel(...)                          # atomic_add in fp32
    return grad_weight.to(torch.bfloat16)
```

fp16 atomics are supported on modern GPUs, so the wrapper keeps fp16 direct;
the same promote-to-fp32 trick is a safe alternative.

---

## 5. Tiling and masks

For tile $(\texttt{pid}_m,\texttt{pid}_n)$:

$$
\begin{aligned}
r_i&=\texttt{pid}_mB_M+i, & c_j&=\texttt{pid}_nB_N+j,\\
m^{row}_i&=\mathbf 1[r_i<M], & m^{col}_j&=\mathbf 1[c_j<D],\\
m_{ij}&=m^{row}_i\wedge m^{col}_j.
\end{aligned}
$$

The id load uses `mask=mask_row`, `other=0`. Masked rows then gather row 0, but
the gather **load** and the output **store** use the combined mask, so no
out-of-bounds address is dereferenced and no invalid value is written.

**Coverage lemma.** The tiles
$T_{pq}=\{pB_M+i:0\le i<B_M\}\times\{qB_N+j:0\le j<B_N\}$ cover
$\{0,\dots,M-1\}\times\{0,\dots,D-1\}$ when $p$ ranges over
$\lceil M/B_M\rceil$ and $q$ over $\lceil D/B_N\rceil$; the masks remove only
points outside the rectangle. Hence every output element is written exactly
once.

For the backward the same coverage holds, but each element is an **atomic add**
into $dW$, so repeated ids are summed correctly.

---

## 6. Cost and roofline

Let $b$ be bytes per element. The forward reads $MDb$ bytes of table rows and
writes $MDb$ bytes; the backward touches the full table ($VDb$) and reads the
gradient ($MDb$). Roughly,

$$
t_{\text{fwd}}\approx\frac{2MDb}{B_{\text{peak}}},
\qquad
t_{\text{bwd}}\approx\frac{(V+M)Db}{B_{\text{peak}}}.
$$

For $V=50{,}000$, $D=1024$, $M=2048$, fp16:

| Quantity | Value |
|---|---|
| Forward output | $2048\cdot1024\cdot2=4.19$ MB |
| Forward traffic (read+write) | $8.39$ MB |
| Backward table | $50{,}000\cdot1024\cdot2=102.4$ MB |
| Backward traffic (+gradient) | $106.6$ MB |

The backward is memory-bound on the table, not on the tokens. This is why
production embedding gradients are often sparse or sharded.

---

## 7. Position embedding fusion

The fused kernel computes

$$
H_{m,d}=W_{I_m,d}+\mathcal P_{m,d}
$$

with one load from the table, one load from the position table, one add, and one
store. The intermediate $X$ never reaches HBM, saving one $MDb$ write and one
$MDb$ read.

If $\mathcal P$ has shape $(1,D)$ and is broadcast over $M$ tokens, the forward
is still elementwise, but the backward must **sum** the broadcast axis:

$$
d\mathcal P_{0,d}=\sum_{m=0}^{M-1}G_{m,d}.
$$

The notebook implements the same-shape case; the broadcast case is an exercise.

---

## 8. Exercises (with hints)

1. **One-hot.** For $V=5,D=3,M=4$, $I=[2,0,1,2]$, write $P$ and verify
   $X=PW$ and $dW=P^\top G$. *Hint:* $P$ has one 1 per row.
2. **Collision.** Choose $I=[2,2,2,2]$ and $G=\mathbf 1$. Compute $dW_2$.
   *Hint:* $4$ rows of ones.
3. **Race.** Sketch an interleaving of two non-atomic updates that loses a row.
4. **Masks.** Set $M=5$, $D=7$, $B_M=B_N=4$. List the four tiles and their masks.
5. **bf16.** Why promote to fp32 before the atomic? What changes in the output
   dtype?
6. **Broadcast position.** Modify the fused kernel for $\mathcal P\in\mathbb R^{1\times D}$;
   what does the backward return?
7. **Determinism.** Replace atomics with a two-pass sorted reduction. What is
   the extra memory and when is it worth it?
