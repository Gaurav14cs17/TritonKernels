# Chapter 19 — Tiling and Vectorization

Part 7: Kernel construction and optimization.

> **Thesis.** Tiling chooses the unit of work; vectorization makes that unit's
> traffic wide and aligned. A tile is correct when its mask covers every live
> cell. A tile is *good* when it fills its warps, fits the register budget, and
> moves 16 bytes per lane.

[Open the chapter notebook](19_complete.ipynb)

[Index](../../00_index.ipynb) · [← Previous](../ch18_kernel_construction/18_complete.ipynb) · [Next →](../ch20_fusion/20_complete.ipynb)

**Estimated time:** 75–95 min · **Difficulty:** Advanced

## Thread

- **Recurring idea:** a tile is a rectangle of work with three properties — a mask, a thread complement, and a memory-transaction width. Every section changes one of them.
- **Chain:** 1-D tile → 2-D tile → 3-D tile → block size → tile size → warps per program → elements per thread → vectorization → register blocking → shared-memory reuse → double buffering → software pipelining.

## Connections

- **Builds on:** Chapter 12 (layouts, warps, elements per thread), Chapter 13 (coalescing, alignment, vector loads), Chapter 18 (1-D/2-D/3-D launch grids).
- **Feeds into:** Chapter 20 (fusion) and every tiled kernel in Part 8.
- **Ladder:** Method 01 (scalar + tile-map simulation) → Method 02 (1-D scale, 2-D copy, vector kernel) → Method 03 (register blocking, broadcast reuse, pipeline).

## Goal

- Read a shape, a tile, and `num_warps` and predict the grid, tile volume, elements per thread, and vector width.
- Write the masked 1-D/2-D kernel, the vector kernel, and the two-stage pipeline.

## Learning objectives

- Compute the grid, $V=\prod_iB_i$, and $e=V/(32W)$.
- Draw a 1-D/2-D/3-D tile map and mark the masked tail.
- Distinguish a block size (one extent) from a tile size (the product).
- Choose a vector width from the dtype and explain the 16-byte transaction.
- Explain register blocking, shared-memory reuse, double buffering, and pipelining.
- Diagnose tail waste, idle lanes, register spill, and unaligned loads.

## Ladder

| Method | What it is | What it proves |
|---|---|---|
| **01** | Scalar scale + Python tile-map simulation | the definition and coverage |
| **02** | Triton 1-D scale, 2-D copy, 4-elements-per-thread kernel | masked tiles and vector loads |
| **03** | Register-blocked rows, broadcast reuse, two-stage pipeline | reuse and overlap |

## Rules

### Grid rule

- **Connection.** First rule; fixes the number of programs for every later section.
- **Rule.** One program owns one tile, and the grid is the outer product of the per-axis tile counts.
- **Formula.** $P=\left\lceil\frac{M}{B_M}\right\rceil\left\lceil\frac{N}{B_N}\right\rceil$ and $P_{3D}=B\,P$.
- **Where.** A 1-D problem drops one axis.
- **Failure.** Forgetting the batch axis, or using the wrong extent on one axis.

### Tile-volume rule

- **Connection.** Multiplies the block sizes from the grid rule into the unit that consumes registers or shared memory.
- **Rule.** The tile is the product of its extents; its byte count is the product times the element size.
- **Formula.** $V=\prod_i B_i$ and $\text{bytes}=V\,b$.
- **Example.** A $2\times4$ fp32 tile is 8 elements / 32 B; a $64\times64$ fp16 tile is 4096 elements / 8 KiB.
- **Where.** Decides whether the program fits in registers or shared memory.
- **Failure.** Too large spills; too small cannot fill a warp.

### Warp rule

- **Connection.** Divides the tile volume among the threads that execute it.
- **Rule.** One warp is 32 lanes, so a program with `num_warps=W` has $32W$ threads.
- **Formula.** $e=\frac{V}{32W}$.
- **Example.** $V=256$ and $W=2$ give 64 threads and $e=4$.
- **Where.** $e$ is the register-pressure and vectorization knob.
- **Failure.** $e\lt 1$ leaves lanes idle; very large $e$ spills.

### Vector rule

- **Connection.** Turns the per-thread element count into memory transactions.
- **Rule.** A lane can issue at most 128 bits, so the vector width is a dtype property.
- **Formula.** $v=\frac{128}{8b}=\frac{16}{b}$, with $vb\le16$ bytes per lane.
- **Example.** fp32: $v=4$; fp16: $v=8$; both give 16 B per lane.
- **Where.** The main lever on memory-bound kernels.
- **Failure.** An unaligned base or a stride between lanes splits the vector.

### Reuse rule

- **Connection.** Reuses loaded values across the outputs a thread owns.
- **Rule.** A thread with an $r_m\times r_n$ output tile reuses each loaded $A$ value $r_n$ times and each $B$ value $r_m$ times.
- **Formula.** $\text{accumulators/thread}=r_mr_n$.
- **Example.** A $4\times4$ block holds 16 accumulators.
- **Where.** The per-thread analogue of a larger tile.
- **Failure.** Too many accumulators spill; too few give low reuse.

### Pipeline rule

- **Connection.** Overlaps the next load with the current compute.
- **Rule.** A pipelined loop has a prologue, a steady state, and an epilogue.
- **Formula.** $T_{\text{serial}}=2T$ and $T_{\text{pipeline}}=T+s-1$.
- **Example.** $s=2$, $T=4$: 8 vs 5 steps, a $1.6\times$ schedule model.
- **Where.** Independent loop iterations over tiles.
- **Failure.** A stage mismatch or a loop-carried dependency removes the overlap.

### Mask rule

- **Connection.** Guards every tile computed by the previous rules.
- **Rule.** Mask every axis and combine the masks; a masked load with an unmasked store is still a bug.
- **Formula.** `mask = (rows[:, None] < M) & (cols[None, :] < N)`.
- **Where.** Every boundary tile and every tail program.
- **Failure.** One-axis masking reads or writes out of range on the other axis.

## Worked numerical example

- **A. The $5\times7$ matrix with $2\times4$ tiles.**
  - $M=5$, $N=7$, $B_M=2$, $B_N=4$, and $P=\lceil5/2\rceil\lceil7/4\rceil=3\cdot2=6$.
  - The six tiles have live counts $(8,6,8,6,4,3)$, summing to $35$.
  - Padded footprint $6\cdot8=48$, so tail waste $=1-35/48=13/48\approx27.1\%$.

- **B. Elements per thread.** The tile volume is $V=8$, so $e=8/(32W)$.

  | $W$ | threads | $e$ for $V=8$ | verdict |
  |---:|---:|---:|---|
  | 1 | 32 | 0.25 | 3 of 4 lanes idle |
  | 2 | 64 | 0.125 | 7 of 8 lanes idle |
  | 4 | 128 | 0.0625 | 15 of 16 lanes idle |

  - Two warps are **64** threads on NVIDIA, not 128.
  - The "128 threads, one element per thread" case needs a 128-element tile and 4 warps: $8\times16$ gives $128/(4\cdot32)=1$.
  - Equivalently, $8\times8=64$ with 2 warps gives $64/64=1$; either way the $2\times4$ tile is too small.

- **C. The 1-D fp32 tile.** For $B=256$, $b=4$, $e=4$: threads $=256/4=64=2$ warps, $v=16/4=4$ fp32 per lane.
  - Each lane moves 16 B; a warp moves $32\cdot16=512$ B.
  - The grid is $\lceil N/256\rceil$, which is 4 for $N=1024$.

- **D. Double-buffer timeline.** For $T=4$ panels and two stages:

  ```text
  serial:      L0 C0 L1 C1 L2 C2 L3 C3                8 units
  pipelined:   L0 | C0/L1 | C1/L2 | C2/L3 | C3        5 units
  ```

  - Serial $2T=8$; pipelined $T+1=5$; model speedup $8/5=1.6$.

## Chapter map

| § | Topic | Formula | Implication | Kernel/demo |
|---|---|---|---|---|
| **19.1** | 1-D tiling | $P=\lceil N/B\rceil$ | mask the tail | `scale1d_kernel` |
| **19.2** | 2-D tiling | $P=\lceil M/B_M\rceil\lceil N/B_N\rceil$ | two grid axes | `copy2d_kernel` |
| **19.3** | 3-D tiling | $P=B\lceil M/B_M\rceil\lceil N/B_N\rceil$ | batch is separate | tile-map arithmetic |
| **19.4** | Block size | $P=\lceil N/B\rceil$ | first knob to sweep | block-size table |
| **19.5** | Tile size | $V=\prod_i B_i$ | fit registers/SMEM | worked arithmetic |
| **19.6** | Warps/program | $32W$ | changes per-thread work | `row_affine_kernel` |
| **19.7** | Elements/thread | $e=V/(32W)$ | keep $e$ integer in $[1,8]$ | worked table |
| **19.8** | Vectorization | $v=16/b$ | contiguous and aligned | `vec4_scale_kernel` |
| **19.9** | Register blocking | $r_mr_n$ accumulators | per-thread reuse | `row_affine_kernel` |
| **19.10** | Shared-memory reuse | $s(B_MB_K+B_KB_N)b$ | `tl.dot` stages panels | `row_broadcast_kernel` |
| **19.11** | Double buffering | $T+1$ vs $2T$ | overlap load/compute | `pipeline_kernel` |
| **19.12** | Software pipelining | prologue/steady/epilogue | `num_stages` | `pipeline_kernel` |
| **★** | Worked example | grid $(3,2)$, $V=8$, $e=0.25$ | asserts for every count | assert cell |

## Pitfalls

| Pitfall | What happens | Fix |
|---|---|---|
| Tile too small | $e\lt 1$; lanes idle | make $V\ge32W$ |
| Register spill | local-memory traffic | lower $e$ or $r_mr_n$ |
| Unaligned vector load | request splits | align; use `tl.multiple_of` |
| Stage mismatch | buffers exceed trip count/SMEM | keep `num_stages <= panels` |

## How to run

```bash
cd 00_Basic/part7_construction/ch19_tiling
jupyter notebook 19_complete.ipynb
```

- **CPU:** every numeric cell and assert runs; Triton tests print `SKIP ...`.
- **GPU:** each kernel is compared with a PyTorch reference.
- **Interpreter:**

```bash
cd 00_Basic
TRITON_INTERPRET=1 PYTHONPATH=$PWD python tools/interpreter_smoke.py
```

## Study checklist

- [ ] Recompute the grid, $V$, and $e$ for the $5\times7$ example.
- [ ] Write the six tile coordinates and live counts.
- [ ] Explain why 2 warps is 64 threads and why 8 elements waste lanes.
- [ ] Derive $v=16/b$ for fp32 and fp16.
- [ ] Distinguish a block size from a tile size.
- [ ] Relate register blocking to a larger tile.
- [ ] Draw the two-stage timeline for $T=4$ and count 5 steps.
- [ ] Run the interpreter and confirm every `check` passes.

## Where this leads

- **Next:** Chapter 20 fuses the tiles into single kernels; the same grid, mask, and vector rules decide whether fusion saves traffic or adds register pressure.
- **Later:** the tiled GEMM and attention chapters reuse the register-blocking and pipeline patterns directly.

## Exercises

1. **Tile map.** For $M=7$, $N=11$, $B_M=3$, $B_N=4$, list tiles, live cells, and tail waste; compare with `tile_map` and `tail_waste`.
2. **Warp arithmetic.** Tabulate $e$ for $V\in\{64,128,256,512\}$ and $W\in\{1,2,4,8\}$; mark $e\lt 1$ and $e\gt 16$.
3. **1-D sweep.** Run `scale1d` for `BLOCK` in $\{64,128,256,512\}$ and `num_warps` in $\{1,2,4\}$; keep $e\ge1$ and explain the pattern.
4. **2-D tail.** Change the $37\times53$ copy to $37\times49$ with `BLOCK_N=16`; which column block has the most masked lanes?
5. **Vector width.** Rewrite `vec4_scale_kernel` for fp16 at 16 B/lane; give the new $e$ and `BLOCK` for 2 warps.
6. **Register block.** Extend `row_affine_kernel` to 8 rows; compute $e$ for `BLOCK_N=64` and 2 warps, then check.
7. **Shared memory.** For $B_M=B_N=B_K=64$, fp16, give bytes per stage and the number of stages that fit in 48 KiB.
8. **Pipeline depth.** Run `NSTEPS=1,2,4,8`; which are degenerate, and why?
9. **Alignment.** Offset a 1-D tensor by one element; is the result still correct, and what did you lose?
10. **Roofline.** For a $4096\times4096$ fp32 copy, compute bytes and minimum time at 1 TB/s; explain why tiling alone cannot beat it.

## Summary

- **Tiling** chooses the unit; **vectorization** widens its traffic.
- **Correctness** is the mask; **quality** is $e$ in $[1,8]$, $e\ge v$, and a tile that fits the budget.
- **The numbers** are counts and schedules, not measurements.
