# Chapter 20 — Fusion

Part 7: Kernel construction and optimization.

**Goal.**
- Read a composition as a dataflow, not as a list of operators.
- Keep intermediates in registers and remove HBM round trips.
- Use the byte model for the ceiling and register pressure for the real limit.

Notebook: [`20_complete.ipynb`](20_complete.ipynb) · Helpers:
[`../../ttools.py`](../../ttools.py) · Apparatus:
[Setup](../../docs/SETUP.md) · [Notation](../../docs/NOTATION.md) ·
[Glossary](../../docs/GLOSSARY.md) · [API index](../../docs/API_INDEX.md) ·
[Formula index](../../docs/FORMULA_INDEX.md) · [Diagrams](../../docs/DIAGRAMS.md) ·
[Troubleshooting](../../docs/TROUBLESHOOTING.md) ·
[Exercise hints](../../docs/SOLUTIONS.md)

[Index](../../00_index.ipynb) · [← Previous](../ch19_tiling/19_complete.ipynb) · [Next →](../../part8_advanced/ch21_atomics/21_complete.ipynb)

**Estimated time:** 75–90 min · **Difficulty:** Advanced

## Connections

- **From Chapter 19.** Tiling, block size, warps per program, and register blocking supply the tile that fusion keeps live.
- **From Chapter 13.** Coalescing, bandwidth, and register/shared-memory trade-offs explain why HBM trips dominate.
- **Section chain.** §20.1–§20.2 affine → §20.3–§20.4 bias/residual → §20.5 chains → §20.6 reduction → §20.7–§20.8 GEMM epilogue → §20.9 traffic rule → §20.10 register limit.
- **To Parts II–IV.** Method 01 measures the three-pass baseline; Method 02 builds fused elementwise and reduction kernels; Method 03 fuses the GEMM epilogue.
- **To Chapter 21.** Atomics and synchronization take over when a fused update must be shared across programs.

## Learning objectives

- State $Q_{\text{fused}}=(I+O)Nb$ and compare it with $Q_{\text{sep}}=2kNb$ for $k$ passes.
- Read a composition as register dataflow: load once, compose, store once.
- Fuse bias, activation, residual, and multi-op pointwise chains.
- Fuse a reduction with its normalization and a bias-plus-activation epilogue into a GEMM accumulator.
- Decide when *not* to fuse from register pressure, occupancy, recomputation, code size, and numerical differences.

## Prerequisites

- Chapters 19 and 13: tiling, warps, coalescing, register/shared-memory trade-offs.
- Chapter 7: elementwise arithmetic; Chapter 8: masks and `tl.where`.

## The ladder

1. **Method 01 — three separate kernels.** Eager PyTorch runs `t = x * a`, `u = t + b`, and `y = relu(u)` as three launches with two full-size intermediates in HBM. This is the correctness oracle.
2. **Method 02 — one fused elementwise kernel.** A masked load, the whole composition in registers, and one masked store: affine + ReLU, residual + activation, a five-op chain, and sum-of-squares + normalization.
3. **Method 03 — a fused GEMM epilogue.** `tl.dot` accumulates in fp32 registers; bias and activation run on the accumulator before the only store.

**Study tip.**
- For $N=1024$ fp32 ($b=4$), count full-size streams.
- Three passes move $6Nb=24{,}576$ bytes.
- One fused pass moves $2Nb=8{,}192$ bytes.
- The fusion saves $4Nb=16{,}384$ bytes, a factor of three.

## The traffic rule

- **Notation.** $N$ elements, $b$ bytes per element, $I$ full-size input streams, $O$ full-size outputs.
- **Kernel traffic.** $Q=(I+O)Nb$.
- **Separate chain.** $k$ unary passes move $Q_{\text{sep}}=2kNb$.
- **Fused chain.** $Q_{\text{fused}}=2Nb$, so $Q_{\text{sep}}/Q_{\text{fused}}=k$.
- **Counting convention.** Count a stream where the kernel actually reads it; scalars are registers and a length-$N$ bias row is small next to an $M\times N$ intermediate, so counting a scalar as $Nb$ over-predicts the speedup.

## Worked numerical example

Take the chapter spine

$$
x=(1,2,3,4),\qquad a=2,\qquad b=-1,\qquad y=ax+b=2x-1=(1,3,5,7).
$$

- **Activation.** Every entry is positive, so $\operatorname{ReLU}(y)=(1,3,5,7)$; the activation is inactive on this spine.
- **Three passes.** $t_1=ax$, $t_2=t_1+b$, and $y=\operatorname{ReLU}(t_2)$; each pass reads one full vector and writes one full vector.
- **Separate traffic.** $Q_{\text{sep}}=2Nb+2Nb+2Nb=6Nb=24{,}576$ bytes for $N=1024$ fp32.
- **Fused traffic.** Load $x$ once, compute $ax+b$ and the ReLU in registers, store $y$ once: $Q_{\text{fused}}=Nb+Nb=2Nb=8{,}192$ bytes.

| Path | Loads | Stores | Bytes at $N=1024$, $b=4$ | Relative |
|---|---|---:|---:|---:|
| three separate kernels | $3Nb$ | $3Nb$ | $24{,}576$ | $3.00\times$ |
| one fused kernel | $Nb$ | $Nb$ | $8{,}192$ | $1.00\times$ |
| saved | $2Nb$ | $2Nb$ | $16{,}384$ | $3.00\times$ less |

The notebook verifies $y=(1,3,5,7)$ and every byte count with assertions.

## Rules

1. **One load, one store.** A pointwise composition should read each input once and write each output once; a full-size temporary means the kernel is not fused.
2. **Count full-size streams.** Use $Q=(I+O)Nb$; scalars are registers, broadcast rows are small, and an $M\times N$ intermediate is the expensive object.
3. **Keep the accumulator in fp32.** Add the bias and apply the activation *before* the cast to fp16 or bf16.
4. **Reduce in registers.** Load the row once, compute `tl.sum`/`tl.max`, and broadcast the scalar back; do not launch a second kernel to divide.
5. **Use the right neutral element.** `other=0.0` for a sum, `-inf` for a max, `+inf` for a min; a masked maximum with `other=0.0` is wrong when every real value is negative.
6. **Do not save an intermediate still needed.** An in-place update can change a value read by a later lane or pass; use distinct pointers.
7. **Fuse only while the live state fits.** A spilled tile adds $2Nb$ of local-memory traffic and lowers occupancy; split at a natural boundary.
8. **Match the reference order when reproducibility matters.** Fused FMA or reassociation changes the last bits; compare with a tolerance or preserve the order.

## Chapter map

| § | Fusion | Formula | Bytes saved | Kernel/demo |
|---|---|---|---|---|
| **20.1** | Add + multiply | $Y=(X+b)s$ | $2Nb$ | `fused_affine_relu` |
| **20.2** | Multiply + add | $Y=Xa+b$ | $2Nb$ | `fused_affine_relu` |
| **20.3** | Bias + activation | $Y=\operatorname{act}(X+B)$ | $2Nb$ | Method 01 / 02 |
| **20.4** | Residual + activation | $Y=\operatorname{act}(X+R)$ | $2Nb$ | `fused_residual_relu` |
| **20.5** | Multiple elementwise ops | $Y=f_k(\cdots f_1(X))$ | $2(k-1)Nb$ | `fused_chain` |
| **20.6** | Reduction + normalization | $Y=X/\sum_j X_j$ | $Nb$ | `fused_rms_norm` |
| **20.7** | GEMM + bias | $C=AB+\mathbf 1b^\top$ | $2MNb$ | `gemm_epilogue` |
| **20.8** | GEMM + activation | $C=\operatorname{act}(AB+\mathbf 1b^\top)$ | $2MNb$ | `gemm_epilogue` |
| **20.9** | Fusion vs separate kernels | $Q_{\text{sep}}=2kNb$ | $2(k-1)Nb$ | traffic model |
| **20.10** | Register-pressure trade-offs | $r\approx LE/T$ | spill adds $2Nb$ | register model |

## When not to fuse

- **Connection.** Qualifies the §20.9 byte decision with the §20.10 register limit.
- **Rule.** Fusion is a traffic optimization, not a universal speedup; when extra live state costs more than the saved bytes, split.
- **Spill model.** With $s$ spilled tiles out of $L$ live tiles, $Q_{\text{fuse}}=2Nb$ when $L\le L_{\max}$, otherwise $2Nb+2sNb$; $Q_{\text{sep}}=2kNb$.
- **Crossover.** Fusion wins while $s\lt k-1$; for a two-pass chain one spill is a tie, and for $k=3$ one spill still wins while two spills lose.

| Situation | Why fusion can lose | What to do |
|---|---|---|
| Register spills | Local-memory traffic and lower occupancy | Smaller `BLOCK`, split, recompute |
| Low occupancy | Fewer blocks per SM hide less latency | More warps or a smaller tile |
| Recomputation | A backward pass may need a saved intermediate | Save it, or split the forward |
| Code size / numerics | Bloat, or last-bit changes from FMA | Split at a boundary; match the order |
| Aliasing / in-place | A later read sees an overwritten value | Separate output or prove the lifetime |

## How to run

```bash
cd part7_construction/ch20_fusion
jupyter notebook 20_complete.ipynb
```

On CPU every numerical cell, the Method 01 reference, the worked example, and the register/traffic asserts run. Each GPU test prints `SKIP <name>: ...`.

Interpreter smoke test without a GPU:

```bash
cd 00_Basic
TRITON_INTERPRET=1 PYTHONPATH=$PWD python tools/interpreter_smoke.py
```

The interpreter runs the four Method 02 kernels and the GEMM epilogue on CPU tensors; `tl.dot` works for the 16×16 test tile.

## Study checklist

- [ ] Write $Q_{\text{sep}}$ and $Q_{\text{fused}}$ for the spine at $N=1024$.
- [ ] Explain why a scalar affine costs $4Nb\to2Nb$ but a bias add costs $5Nb\to3Nb$.
- [ ] State what the sum-of-squares normalization keeps in registers.
- [ ] Draw the GEMM epilogue and identify the matrix that never reaches HBM.
- [ ] Compute $r=LE/T$ for $E=1024$, $T=128$, $L=32$ and decide whether it spills.
- [ ] Name two cases where saving an intermediate beats recomputing it.
- [ ] Break a bias broadcast and the reduction's `other` value; explain each bug.
- [ ] Measure fused versus separate affine+ReLU and compare with the byte model.

**Where this leads.**
- **Chapter 21.** Atomics, synchronization, and race conditions for shared updates.
- **Chapter 22.** Full `tl.dot` GEMMs, tensor cores, and epilogue optimization.
- **Chapter 23.** Fused transformer blocks that chain attention, normalization, and MLP epilogues.
- **Chapter 24.** Profiling and roofline analysis to confirm whether the kernel is bandwidth- or occupancy-limited.

## Exercises

1. For $N=2048$ fp32, derive the bytes of a four-pass chain and one fused pass; verify the ratio with assertions.
2. Write a 2-D bias+ReLU kernel with `bias[None, :]` and `bias[:, None]`; construct a silent-broadcast bug.
3. Modify `fused_residual_relu` to write in place over `x`; find an input where the result changes.
4. Split the five-op chain into two kernels at the square root and compare bytes.
5. Write a 2-D row normalization; test a non-square matrix and a masked row.
6. Extend the fused reduction kernel to a stable softmax with one load/store pair.
7. Add GELU to `gemm_epilogue_kernel`; explain why the interpreter may need a fallback.
8. For $B_M=B_N=64$ and $T=128$, compute accumulator registers per thread; how many more $64\times64$ fp32 tiles fit under 255 registers?
9. Using $Q_{\text{fuse}}(L)$, find the smallest $s$ that makes a three-pass chain prefer the separate kernels.
10. Benchmark fused and separate affine+ReLU on a GPU; compare with the predicted $3\times$ byte ratio and explain the gap.
