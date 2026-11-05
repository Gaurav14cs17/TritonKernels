# 13 — Memory Optimization

> **Goal.** Read a kernel as a traffic problem before reading it as an
> instruction stream. Coalescing, alignment, vector width, cache reuse,
> registers, shared memory, and fusion all change the bytes moved or the lines
> touched. Every number is checkable on paper; the code cells assert it.

Notebook: [`13_complete.ipynb`](13_complete.ipynb) · Helpers:
[`../../ttools.py`](../../ttools.py) · Apparatus:
[Setup](../../docs/SETUP.md) · [Notation](../../docs/NOTATION.md) ·
[Glossary](../../docs/GLOSSARY.md) · [API index](../../docs/API_INDEX.md) ·
[Formula index](../../docs/FORMULA_INDEX.md) · [Diagrams](../../docs/DIAGRAMS.md) ·
[Troubleshooting](../../docs/TROUBLESHOOTING.md) ·
[Exercise hints](../../docs/SOLUTIONS.md)

[Index](../../00_index.ipynb) · [← Previous](../ch12_layouts/12_complete.ipynb) · [Next →](../../part6_reductions/ch14_reductions/14_complete.ipynb)

**Estimated time:** 80–95 min · **Difficulty:** Intermediate

**Connections.**

- **From Chapter 12.** The logical layout and lane map are fixed; this chapter
  asks how many lines that map touches.
- **Within Chapter 13.** 13.1 coalescing → 13.2 stride → 13.3 alignment → 13.4
  loads → 13.5 stores → 13.6 bandwidth → 13.7 cache → 13.8 registers → 13.9
  shared memory → 13.10 access patterns → 13.11 redundant loads → 13.12
  redundant stores.
- **To Chapter 14.** Reductions reuse the same traffic model: fewer bytes and
  fewer lines still win.

## Learning objectives

- Count the $128$-byte lines touched by a warp for any stride, and explain why
  coalescing turns 32 lane addresses into one or four transactions.
- Compute bytes moved, arithmetic intensity, and roofline time for a 1-D kernel.
- Align a base and a tile, emit $16$-byte vector loads/stores, and relate
  elements per thread to vector width.
- Treat cache, registers, and shared memory as explicit traffic/occupancy budgets.
- Remove redundant loads and stores by fusing an affine chain, and prove the
  traffic reduction with arithmetic.

## Prerequisites

- Chapter 12: logical layouts, elements per thread, `num_warps`.
- Chapter 6: pointers, offsets, and masks.
- Chapter 7: why elementwise kernels are memory-bound.

---

## Study ladder

1. **Method 01 — naive strided reference.** A Python loop copies a row-major
   matrix in row order, then in column order. The values agree; the strides do
   not. It is the correctness oracle and the address-map model.
2. **Method 02 — coalesced and strided Triton kernels.** A 1-D scale keeps the
   lane index unit-stride. A second kernel compacts `x[::4]` into a contiguous
   output, so the load is strided and the store is coalesced.
3. **Method 03 — fused and vectorized.** One kernel computes
   $y=\alpha x+\beta$ with one load and one store. A second variant uses
   `BLOCK=1024`, `num_warps=8`, four `float32` per thread, and a $16$-byte
   vector. A 2-D row-broadcast kernel demonstrates register blocking.

**Study tip.** Fix $N=1024$ `float32`, $b=4$ B, $B=256$, and write

$$
Q_r=Nb=4096\ \text{B},\quad Q_w=Nb=4096\ \text{B},\quad
Q=8192\ \text{B}=8\ \text{KiB},\quad
N_{\text{line}}=\frac{Nb}{128}=32
$$

per direction. Every section changes one of those numbers: coalescing changes
lines, vectorization changes instructions, fusion changes $Q$.

---

## Mathematics in one page

- **Lines.** A warp with element stride $s$ spans $32sb$ bytes and touches
  $T_{\text{line}}(s)=\lceil 32sb/L\rceil$ lines, with $L=128$ B. For
  `float32`, $s=1,4,32$ gives $1,4,32$ lines; a tile spans $Bsb$ bytes, or
  $Bsb/L$ lines.
- **Vector width.** $w=128$ bits $=16$ bytes, so $v=w/b=4$ for `float32` and
  $v=8$ for `float16`/`bfloat16`. A contiguous aligned tile needs $B/v$ loads
  and $B/v$ stores; a $16$-byte access needs $a\bmod16=0$.
- **Cache.** A $128$-byte line has four $32$-byte sectors. A working set is
  reused when $W\le C_{\text{L2}}$ and the reuse distance is short; a
  $1024\times1024$ `float32` matrix is $4$ MiB and $32{,}768$ lines.
- **Roofline.** $Q=Q_{\text{read}}+Q_{\text{write}}$,
  $\mathrm{BW}=Q/t$, and
  $t\ge\max(N/P_{\text{peak}},Q/\mathrm{BW})$, with
  $\mathrm{AI}=\text{FLOPs}/Q$. For $\alpha x+\beta$, $2$ FLOPs and $8$ bytes
  per element give $\mathrm{AI}=0.25$ FLOP/B.
- **Registers.** With $e=B/(32W)$ elements per thread and $k$ live tiles, the
  data-register estimate is $ke$.
- **Shared memory.** A triple-buffered GEMM tile stages
  $M_{\text{smem}}=s(B_MB_K+B_KB_N)b$ bytes; for $64^3$, `float16`, and $s=3$
  that is $48$ KiB.

---

## Worked numerical example

- **Traffic.** For $N=1024$ `float32` and a $256$-element tile,
  $Q_r=4096$ B, $Q_w=4096$ B, and $Q=8192$ B.
- **Lines.** At $32$ elements per line, each direction touches $32$ lines and
  the copy touches $64$; a tile is $1024$ B, or $8$ lines, so four tiles cover
  the $32$ lines of one direction.
- **Threads and vectors.** The map follows $e=B/T$:

| threads $T$ | warps $W$ | elements/thread $e$ | per-thread access |
|---:|---:|---:|---|
| 32 | 1 | 8 | two 16-byte vectors |
| 64 | 2 | 4 | one 16-byte vector |
| 128 | 4 | 2 | one 8-byte vector |
| 256 | 8 | 1 | one 4-byte scalar |

- Four elements per thread means $T=64$ (two warps), not $T=128$: a $4$-warp
  program covers the same $256$-element tile with two elements per thread. The
  vector width is $128/32=4$, so the tile needs $256/4=64$ transactions.
- **Stride-4.** A stride-4 tile touches $0,16,\ldots,4080$ B, so its span is
  $256\cdot4\cdot4=4096$ B and $N_{\text{line}}=4096/128=32$, against $8$
  contiguous lines, a $4\times$ penalty. Per warp that is $4$ lines; stride $32$
  gives $32$ lines.
- **Bandwidth.** At $t=4.0\ \mu s$, the tiny copy achieves
  $\mathrm{BW}=8192/(4.0\times10^{-6})=2.05$ GB/s, which is launch overhead: the
  $3$ TB/s roof for $8$ KiB is $2.73$ ns. The notebook therefore uses $N=2^{26}$,
  where $Q=512$ MiB and $t=(512\ \text{MiB})/(3.0\ \text{TB/s})=0.179$ ms.

## Rules

| # | Rule | Statement | Check |
|---|---|---|---|
| 13.1 | Coalesce | lane $\ell$ touches $a_0+\ell b$ | one line per `float32` warp |
| 13.2 | Avoid strides | lane $\ell$ touches $a_0+\ell sb$, $s\gt 1$ | $s\times$ line overfetch |
| 13.3 | Align | $a\bmod w=0$ for a $w$-byte access | base and tile start $\bmod16=0$ |
| 13.4 | Vector-load | $v=128/b$ elements per 16-B load | $B/v$ loads per tile |
| 13.5 | Vector-store | $B/v$ stores per tile | store offsets match load offsets |
| 13.6 | Bound bandwidth | $t\ge Q/\mathrm{BW}$ | $Q/t$ is achieved bandwidth |
| 13.7 | Reuse cache | $W\le C_{\text{L2}}$ | 4 MiB fits a large L2 |
| 13.8 | Watch registers | $e=B/(32W)$ per live tile | spills add local-memory traffic |
| 13.9 | Budget shared memory | $s(B_MB_K+B_KB_N)b$ | 48 KiB for $64^3$ fp16, 3 stages |
| 13.10 | Map lanes | $\Delta a=b$ across lanes | row 1 line, column 32 lines |
| 13.11 | Load once | $2kNb\to2Nb$ | fuse a $k$-op chain |
| 13.12 | Store once | $4Nb\to2Nb$ | no intermediate HBM |

---

## Chapter map

| § | Topic | Rule/formula | Traffic implication | Kernel/demo |
|---|---|---|---|---|
| **13.1** | Coalesced access | $T_{\text{line}}=\lceil 32b/128\rceil$ | one line per warp | `scale_coalesced` |
| **13.2** | Uncoalesced access | $T_{\text{line}}=\lceil 32sb/128\rceil$ | up to $s\times$ overfetch | `scale_strided` |
| **13.3** | Memory alignment | $a\bmod w=0$ | misalignment splits vectors | `fused_affine_vec` |
| **13.4** | Vectorized loads | $v=128/b$ | $B/v$ 16-B loads | `fused_affine_vec` |
| **13.5** | Vectorized stores | $B/v$ 16-B stores | same lines, fewer instructions | `fused_affine_vec` |
| **13.6** | Memory bandwidth | $\mathrm{BW}=Q/t$ | roofline $t\ge Q/\mathrm{BW}$ | performance model |
| **13.7** | Cache behavior | $W\le C_{\text{L2}}$ | reuse cuts HBM bytes | `row_broadcast` |
| **13.8** | Register usage | $e=B/(32W)$ | spills add traffic | `fused_affine_vec` |
| **13.9** | Shared memory | $s(B_MB_K+B_KB_N)b$ | staging is an occupancy budget | resource arithmetic |
| **13.10** | Access patterns | $\Delta a=b$ across lanes | row 1 line, column 32 lines | `row_broadcast` |
| **13.11** | Avoid redundant loads | $2kNb\to2Nb$ | one read per byte | `fused_affine` |
| **13.12** | Avoid redundant stores | $4Nb\to2Nb$ | no intermediate HBM | `fused_affine` |
| **★** | Capstone | all of the above | $2\times$ fusion bound | fused + vectorized |

---

## How to run

```bash
cd part5_layouts/ch13_memory_optimization
jupyter notebook 13_complete.ipynb
```

CPU: every arithmetic, reference, resource, and roofline cell runs. GPU tests
print `SKIP <name>: ...` and the notebook continues.

```bash
cd 00_Basic
TRITON_INTERPRET=1 PYTHONPATH=$PWD python tools/interpreter_smoke.py
```

## Study checklist

- [ ] Draw the lane-address map for a contiguous tile and a stride-4 tile.
- [ ] Count bytes, lines, and vector transactions for $N=1024$, $B=256$.
- [ ] Explain why four elements per thread needs 64 threads for a 256 tile.
- [ ] Compute the roofline time for $Q=512$ MiB at $3$ TB/s.
- [ ] State the alignment condition for a 16-byte access.
- [ ] Estimate registers for two live tiles and say when spilling begins.
- [ ] Compute the shared-memory budget for a $64^3$ `float16` tile.
- [ ] Show that fusion halves the affine-chain traffic; run CPU and interpreter.

**Where this leads.** The exercises extend the line-count, vectorization,
roofline, register, and fusion arguments; Chapter 14 reuses the model.

## Exercises

1. **Line count.** For $N=4096$ `float32` and stride $8$, compute the
   contiguous and strided line counts; verify by counting distinct
   `addr // 128` values for one warp.
2. **Tail.** Run the coalesced kernel with $N=1025$, `BLOCK=256`. Which tile
   loses vectorization, and how many scalar accesses remain?
3. **Alignment.** Pass `x[1:]` to the coalesced kernel and print
   `data_ptr() % 16`. Does the compiler still emit a vector load?
4. **Stride sweep.** Count or time $s=1,2,4,8,16,32$ for $2^{20}$ `float32`;
   compare with $32sb/128$ and explain the plateaus.
5. **Roofline.** For $N=2^{24}$, $b=4$, compute $Q$, $\mathrm{AI}$, and the
   expected time at $2$ TB/s for a copy, a fused affine, and a two-pass affine.
6. **Fusion chain.** Fuse $y=3x^2+2x+1$ and compare the load/store count with a
   three-kernel version.
7. **Registers and shared memory.** Keep three live tiles, compile two `BLOCK`
   values, inspect spills, then compute the staged bytes for `float16`,
   $B_M=B_N=128$, $B_K=32$, $s=2,3,4$ and the blocks-per-100-KiB budget.
8. **Column kernel.** Read a $32\times32$ `float32` column and write it
   contiguously; compute the line count before and after the transpose.

Hints are in [`../../docs/SOLUTIONS.md`](../../docs/SOLUTIONS.md).

[Index](../../00_index.ipynb) · [← Previous](../ch12_layouts/12_complete.ipynb) · [Next →](../../part6_reductions/ch14_reductions/14_complete.ipynb)
