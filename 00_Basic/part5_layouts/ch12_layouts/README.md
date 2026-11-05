# Chapter 12 — Triton Layout Fundamentals

Part 5: Layouts and data movement.

> **Goal.** Separate the tensor you *think* you are writing from the tensor the hardware actually holds. A Triton tensor has a **logical shape** and a compiler-chosen **physical layout**: which lane owns each element, how many elements each thread keeps in registers, and when values must move between lanes. The arithmetic is small; the consequences for speed are not.

**Thread.**

- **Recurring idea.** Every Triton tensor is two objects at once: a logical shape $(M,N)$, and a compiler-chosen map from each logical element to a lane and a register slot.
- **Chain of topics.** Shape → layout → distribution → elements/thread → warps → `tl.arange` → `tl.trans()` → layout conversion → distributed, blocked, and slice layouts → compatibility → performance.
- **Driving question.** For each tile: who owns each element, how many elements does each owner hold, and what does that map cost in memory transactions?

Notebook: [`12_complete.ipynb`](12_complete.ipynb) · Shared helpers: [`../../ttools.py`](../../ttools.py) · Book apparatus: [Setup](../../docs/SETUP.md) · [Notation](../../docs/NOTATION.md) · [Glossary](../../docs/GLOSSARY.md) · [API index](../../docs/API_INDEX.md) · [Formula index](../../docs/FORMULA_INDEX.md) · [Diagrams](../../docs/DIAGRAMS.md) · [Troubleshooting](../../docs/TROUBLESHOOTING.md) · [Exercise hints](../../docs/SOLUTIONS.md)

[Index](../../00_index.ipynb) · [← Previous](../../part4_broadcasting/ch11_index_transforms/11_complete.ipynb) · [Next →](../ch13_memory_optimization/13_complete.ipynb)

**Estimated time:** 65–80 min · **Difficulty:** Intermediate

## Learning objectives

- Separate a tensor's logical shape from the compiler-chosen physical layout.
- Compute elements/thread, threads/program, and warps/program, and use those counts to predict register pressure and lane utilization.
- Write the canonical blocked map for a 1-D and a 2-D tile, and check a mapping table by hand.
- State what `tl.arange` produces and what `tl.trans` changes — and what it does not change.
- Recognize when a layout conversion is required and estimate its traffic.
- Predict coalescing efficiency from a thread-to-address map.

## Prerequisites

- Chapter 11: reshape, transpose, permute, and slicing as logical operations.
- Chapter 2: programs, grids, and `tl.program_id`.
- Chapter 3: `tl.arange` and tensor shapes.
- No prior GPU layout or shared-memory background is assumed.

## Connections

- **Builds on.** Chapter 11's logical transforms, Chapter 2's program/grid model, and Chapter 3's index vectors.
- **Internal chain.** Shape → layout → distribution → elements/thread → warps → `tl.arange` → `tl.trans` → conversion → distributed/blocked/slice → compatibility → performance.
- **Leads to.** Chapter 13 (memory optimization: coalescing, alignment, vectorization, and measured bandwidth).
- **Teaching boundary.** The concrete layout is chosen by the compiler; this chapter uses one explicit warp-blocked, lane-interleaved model so every table can be checked by hand.

## The mathematical model

- **Logical shape.** For a matrix tile, $\operatorname{shape}(X)=(M,N)$ and $\operatorname{numel}(X)=MN$. With row-major order, $k=iN+j$, $i=\lfloor k/N\rfloor$, and $j=k\bmod N$. Shape fixes the grid, masks, and pointer arithmetic; it does not fix which thread owns which element.
- **Layout.** A layout is an injective map $\mathcal L:(i,j)\mapsto(t,r)$ with $t\in\{0,\ldots,32W-1\}$ and $r\in\{0,\ldots,e_t-1\}$. When the tile is smaller than the thread count, some threads hold nothing; when it is larger, a thread receives several register slots.
- **Counts.** With tile volume $T$ and $W$ warps per program, $\bar e=T/(32W)$ and the padded tile is $T'=32W\lceil T/(32W)\rceil$. Threads/program is $32W$. If a kernel keeps $R$ live tensors, register demand is roughly $R\,e_t$ values per thread before temporaries.
- **Cost model.** Effective traffic gives the lower bound $t\ge Q_{\text{eff}}/\mathrm{BW}$; efficiency is $\eta=Q_{\text{useful}}/Q_{\text{eff}}$. A shared-memory conversion moves at least $2Tb$ bytes per tile.

## Worked numerical example

### An 8-element 1-D tile on one warp

- **Setup.** $T=8$, $W=1$: 32 lanes available, 8 live elements.
- **Map.**

| $k$ | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| thread $t$ | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
| warp | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| lane | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
| register | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

- **Counts.** Average $\bar e=8/32=0.25$; one value on 8 active lanes; utilization $8/32=25\%$.

### A 4×4 tile over two warps

- **Setup.** Flatten row-major, $k=4i+j$, with $T=16$ and $W=2$; the warp chunk is $C=\lceil16/2\rceil=8$.
- **Map.**

| $(i,j)$ | $k$ | thread | warp | lane | register |
|---|---:|---:|---:|---:|---:|
| (0,0) | 0 | 0 | 0 | 0 | 0 |
| (0,3) | 3 | 3 | 0 | 3 | 0 |
| (1,0) | 4 | 4 | 0 | 4 | 0 |
| (1,3) | 7 | 7 | 0 | 7 | 0 |
| (2,0) | 8 | 32 | 1 | 0 | 0 |
| (2,3) | 11 | 35 | 1 | 3 | 0 |
| (3,0) | 12 | 36 | 1 | 4 | 0 |
| (3,3) | 15 | 39 | 1 | 7 | 0 |

- **Counts.** Average $\bar e=16/64=0.25$; one value on 16 active lanes; utilization $16/64=25\%$.
- **Reading.** Warp 0 owns flat indices 0–7, warp 1 owns 8–15, and each active lane holds one value in register 0.

### Where a second register appears

- **8×8 tile.** $T=64$ on one warp gives $e_t=2$: indices 0–31 sit on lanes 0–31 in register 0, and indices 32–63 sit on the same lanes in register 1.
- **64×64 tile.** $T=4096$; 4 warps gives $T/(32\cdot4)=32$ values per thread, and 8 warps gives 16.
- **Check.** The notebook asserts every number in these tables.

## Rules

- **Shape rule.** Shape is logical metadata; strides are physical storage. A transposed view changes strides, not values.
- **Layout rule.** A layout is a map $(i,j)\mapsto(t,r)$. The concrete map is chosen by the compiler; the tables use one explicit teaching model.
- **Count rule.** Elements/thread $=T/(32W)$; threads/program $=32W$. A tile smaller than the thread count leaves lanes idle; a tile much larger raises register pressure.
- **Arange rule.** `tl.arange(0, BLOCK)` builds consecutive integers of a compile-time, power-of-two length. It is an index vector, not a data vector.
- **Transpose rule.** `tl.trans(x)` reverses logical axes: $y_{ji}=x_{ij}$. It does not move global memory by itself; the store decides the destination.
- **Conversion rule.** When producer and consumer layouts differ, the compiler inserts a conversion. A shared-memory conversion moves at least $2Tb$ bytes per tile: one write and one read.
- **Compatibility rule.** Elementwise combination needs broadcast-compatible shapes *and* reconcilable layouts. Same shape is necessary, not sufficient.
- **Performance rule.** $t\ge Q_{\text{eff}}/\mathrm{BW}$. A stride-$s$ warp read touches about $\min(32,\max(1,s))$ 128-byte transactions for 32 useful fp32 words, so efficiency is about $1/s$ until the 32-transaction ceiling.

## Chapter map

- **How to read the table.** Each row names the formula or rule, the layout implication, and the runnable demo.

| § | Topic | Formula/rule | Layout implication | Kernel/demo |
|---|---|---|---|---|
| **12.1** | Logical tensor shape | $\operatorname{shape}=(M,N)$, $k=iN+j$ | shape fixes grid and mask, not owner | 2-D tile copy |
| **12.2** | Layout concepts | $\mathcal L:(i,j)\mapsto(t,r)$ | owner and memory adjacency | host map |
| **12.3** | Element distribution | $w=\lfloor k/C\rfloor,\ \ell=k-wC$ | adjacent lanes → coalescing | 8-element table |
| **12.4** | Elements per thread | $\bar e=T/(32W)$ | register pressure / idle lanes | count table |
| **12.5** | Threads per warp | 32 on NVIDIA | scheduling and transaction unit | `num_warps` sweep |
| **12.6** | Warps per program | $32W$ threads/program | latency hiding vs work per thread | `num_warps` sweep |
| **12.7** | `tl.arange` layouts | $a_k=\text{start}+k$ | index-vector layout | arange kernel |
| **12.8** | `tl.trans()` | $y_{ji}=x_{ij}$ | swaps axes, may convert | trans kernel |
| **12.9** | Layout conversion | $Q_{\text{conv}}\ge2Tb$ | shared-memory round trip | tiled transpose |
| **12.10** | Distributed tensors | $X=\bigcup_p X_p$ | per-program ownership | tiled transpose |
| **12.11** | Blocked layouts | $C_d=s_dt_dw_d$ | regular per-thread sub-blocks | 16×16 blocked table |
| **12.12** | Slice layouts | $Y_{ij}=X_{i+s_m,j+s_n}$ | non-contiguous sub-window | offset tile |
| **12.13** | Layout compatibility | same shape + common layout | conversions at elementwise joins | trans + add |
| **12.14** | Layout performance | $t\ge Q_{\text{eff}}/\mathrm{BW}$ | $1/s$ coalescing efficiency | traffic model |

## How to run

- **Notebook.** `cd part5_layouts/ch12_layouts && jupyter notebook 12_complete.ipynb`
- **CPU.** Every host-side cell and table check runs; Triton tests print `SKIP <name>: ...`.
- **Interpreter.** `cd 00_Basic && TRITON_INTERPRET=1 PYTHONPATH=$PWD python tools/interpreter_smoke.py`
- **Imports.** The notebook walks up to `00_Basic` for `ttools.py`; setup prints only the active device.
- **Guarding.** GPU tests are guarded by `ttools.HAS_GPU`; the interpreter harness sets it to `True` after importing `ttools`.

## Study checklist

- [ ] Draw the 8-element tile twice: once as 2 threads × 4 registers, once as one warp with 8 active lanes.
- [ ] Write the 4×4 table for $(i,j)=(2,3)$ without looking; then check the notebook's assert.
- [ ] Compute elements/thread for 8×8, 16×16, and 32×32 tiles with 1, 2, and 4 warps.
- [ ] State what `tl.trans` changes and what it leaves alone.
- [ ] Explain why `tl.trans(x) + y` can be slower than `x + y`.
- [ ] Compute the traffic ratio for a transpose with one uncoalesced side.
- [ ] Break one table entry and explain which assert fails.

## Pitfalls

- **Assuming a fixed thread mapping.** The compiler chooses the concrete layout; test values, not owners.
- **Confusing `tl.trans` with `torch.transpose`.** One permutes a distributed tile; the other changes strides.
- **Forgetting the conversion cost.** A transpose or a reduction may stage a tile through shared memory.
- **Confusing warps with threads.** `threads = 32 * num_warps` on NVIDIA.
- **Too many elements per thread.** Registers spill; raise `num_warps` or shrink the tile. Too few leaves lanes idle.
- **Assuming same shape means same layout.** A conversion can appear at the elementwise join.

## Exercises

1. Draw the 8-element tile as a 2-warp map with 4 elements per thread. Write the thread/lane/register table and compute utilization.
2. Compute elements/thread for 8×8, 16×16, and 32×32 tiles with 1, 2, and 4 warps. Which combination is most likely to spill, and why?
3. Extend the host map to 1024 elements and 4 warps. Assert the inverse for every live index and report the number of active threads.
4. Modify the `tl.arange` kernel to start at 100 and length 16. Verify the values and explain what the layout does not change.
5. Build a 2-D tile with rows $[2,3]$ and columns $[4,5,6]$. Write the flat offsets by hand, then check them with asserts.
6. Transpose a $5\times7$ tile using an $8\times8$ block. Write both masks and explain why the store mask is not the transpose of the load mask.
7. Apply `tl.trans` twice in one kernel. Show logically that the result is the input, then explain why it may still cost two conversions.
8. Compute the traffic for a $2048\times2048$ fp32 transpose with one uncoalesced side. Compare with the $4096\times4096$ case.
9. Design a copy kernel whose warp reads with stride 4. Predict transactions and efficiency, then propose a tiled alternative.
10. Give a kernel where two tensors have the same shape but different layouts, and explain where the compiler inserts the conversion.

## Where this leads

- **Exercises.** The ten exercises test the map, the counts, the conversion cost, and the coalescing model.
- **Next chapter.** Chapter 13 turns the efficiency table into alignment, vectorization, and measured bandwidth.
- **Hints.** See [`../../docs/SOLUTIONS.md`](../../docs/SOLUTIONS.md).
