# Chapter 18 — GPU Kernel Construction

Part 7: Kernel construction and optimization.

> **Goal.** Choose the work assignment before writing the kernel body. A Triton
> program is a block of lanes; the grid is the shape of the ownership relation.
> This chapter derives vectors, rows, columns, tiles, 3-D batches, grid-stride
> workers, and multi-launch pipelines, then checks every map with an `assert`.

Notebook: [`18_complete.ipynb`](18_complete.ipynb) · Helpers: [`ttools.py`](../../ttools.py) · Apparatus:
[Setup](../../docs/SETUP.md) · [Notation](../../docs/NOTATION.md) · [Glossary](../../docs/GLOSSARY.md) ·
[API index](../../docs/API_INDEX.md) · [Formula index](../../docs/FORMULA_INDEX.md) · [Diagrams](../../docs/DIAGRAMS.md) ·
[Troubleshooting](../../docs/TROUBLESHOOTING.md) · [Solutions](../../docs/SOLUTIONS.md)

[Index](../../00_index.ipynb) · [← Previous](../ch17_control_flow/17_complete.ipynb) · [Next →](../ch19_tiling/19_complete.ipynb)

**Estimated time:** 80–100 min · **Difficulty:** Intermediate–Advanced

## Thread

- **Recurring idea.** A kernel is a work-assignment map: the grid is the shape of the ownership relation, and every program owns a disjoint rectangle of the output.
- **Chain.** Vector → row → column → tile → 1-D/2-D/3-D grids → grid-stride → boundaries → multiple launches.
- **Check.** Every mapping is backed by an `assert`, and every tail is masked on both the load and the store.

## Learning objectives

- Choose a mapping — vector, row, column, tile, or persistent worker — before writing the body.
- Derive 1-D, 2-D, and 3-D grids with $P_k=\lceil \text{size}_k/B_k\rceil$.
- Write the program-to-element map for a $5\times7$ matrix and a $3\times4$ vector, then mask every non-divisible axis on both load and store.
- Convert a fixed grid to a grid-stride loop and choose between fusion and two launches.

**Prerequisites:** Chapter 2 (program ids, launch tuples), Chapter 6 (2-D offsets), Chapter 7 (masked loads and `other=`).

## Study ladder

1. **Method 01 — host simulation.** Python walks each ownership relation and writes a CPU tensor; it is the correctness oracle.
2. **Method 02 — one kernel per mapping.** Row/column scale, tiled copy/transpose, and a 3-D batch copy.
3. **Method 03 — persistent + multi-launch.** Grid-stride plus a two-launch pipeline compared with its fused form.

**Study tip.**

- Fix $A\in\mathbb R^{5\times7}$ with $A_{ij}=7i+j+1$ and $v\in\mathbb R^{3\times4}$ with $v_{ij}=4i+j+1$.
- Write four maps by hand: per row (5 programs), per $3\times4$ tile (grid $2\times2$), per element (35 programs), and grid-stride with 2 programs.
- List the rows, columns, and masks; every hand value is asserted in the notebook.

## Connections

- **Backward.** Chapter 2 supplies `tl.program_id` and launch tuples; Chapter 6 supplies 2-D offsets and row-major addresses; Chapter 7 supplies masked loads and `other=`.
- **Within the chapter.** 18.1 vector → 18.2 row → 18.3 column → 18.4 tile → 18.5–18.7 grid rank → 18.8 grid-stride → 18.9 boundaries → 18.10 multiple launches.
- **Method chain.** Part II simulates the maps, Part III transcribes them into Triton, and Part IV makes the grid persistent and pipelines two launches.
- **Forward.** Chapter 19 turns the tile map into a full tiling strategy; Chapter 20 fuses multi-launch pipelines into one kernel.

## The mapping and the grid

- **Vector.** Program $p$ owns $V_p=\{pB+j:0\le j\lt B\}\cap[0,N)$ and $P=\lceil N/B\rceil$.
- **Row.** Program $p$ owns $R_p=\{(p,j):0\le j\lt N\}$ and $P=M$.
- **Column.** Program $p$ owns $C_p=\{(i,p):0\le i\lt M\}$ and $P=N$.
- **Tile.** Program $(p_m,p_n)$ owns $[p_mB_M,(p_m+1)B_M)\times[p_nB_N,(p_n+1)B_N)$ with $P_m=\lceil M/B_M\rceil$, $P_n=\lceil N/B_N\rceil$.
- **3-D.** Grid $(B,P_m,P_n)$ maps $(b,p_m,p_n)$ to batch $b$, tile $T_{p_m,p_n}$ with offset $bMN+iN+j$.
- **Launch-grid rule.** $P_k=\lceil \text{size}_k/B_k\rceil$ on every axis: 1-D `(P,)`, 2-D `(P_m,P_n)`, 3-D `(B,P_m,P_n)`.
- **Grid-stride rule.** With $P$ fixed programs, program $p$ visits starts $pB+kPB$ until $N$: `range(pid*BLOCK, n, num_programs*BLOCK)`.

## Worked example: a $5\times7$ matrix and a $3\times4$ vector

- **Setup.**

$$
A=\begin{bmatrix}
1&2&3&4&5&6&7\\
8&9&10&11&12&13&14\\
\vdots&&&\vdots&&&\vdots\\
29&30&31&32&33&34&35
\end{bmatrix},
\qquad
v=\begin{bmatrix}1&2&3&4\\5&6&7&8\\9&10&11&12\end{bmatrix}.
$$

- **Per row.** $P=M=5$; program $p$ owns row $p$. With $B_N=8$, every program has one masked lane ($j=7$).
- **Per $3\times4$ tile.** Grid $(\lceil5/3\rceil,\lceil7/4\rceil)=(2,2)$.

| $(p_m,p_n)$ | rows | cols | masked cells |
|---|---|---:|---:|
| (0,0) | 0,1,2 | 0,1,2,3 | 0 |
| (0,1) | 0,1,2 | 4,5,6,7 | 3 |
| (1,0) | 3,4,5 | 0,1,2,3 | 4 |
| (1,1) | 3,4,5 | 4,5,6,7 | 6 |

- **Tile totals.** $0+3+4+6=13=48-35$; program $(1,1)$ masks row 5 and column 7.
- **Per element.** $P=MN=35$; program $q$ owns $(\lfloor q/7\rfloor,q\bmod7)$, with no mask.
- **Grid-stride with two programs.** Flatten $A$ to $N=35$; $B=4$, $P=2$, stride $PB=8$.

| pid | starts | row/col of each start | tail mask |
|---:|---|---|---|
| 0 | 0,8,16,24,32 | $(0,0),(1,1),(2,2),(3,3),(4,4)$ | last tile `T,T,T,F` |
| 1 | 4,12,20,28 | $(0,4),(1,5),(2,6),(4,0)$ | none |

- **Tail.** Program 0's last tile covers flat $32,33,34,35$; flat 35 would be $(5,0)$ and is masked.
- **The $3\times4$ vector with $2\times2$ tiles.** Rows $0,1$ are full and rows $2,3$ carry the mask ($16-12=4$ slots); per column, program 3 owns $(4,8,12)$.

## Rules

1. **Decide the mapping first:** coordinate, rows, columns, then mask.
2. **Use the ceiling on every axis.** `P = cdiv(size, block)`; `size // block` drops the tail.
3. **Mask every non-divisible axis twice.** Guard both `tl.load` and `tl.store`.
4. **Build 2-D masks with explicit axes:** `row_mask[:, None] & col_mask[None, :]`; two `(B,)` masks make `(B,B)`.
5. **Keep one axis convention.** `program_id(0)` is the first grid axis.
6. **Pad the Triton block, not the logical tile.** `tl.arange` needs a power of two, so a $3\times4$ tile runs on a $4\times4$ block with a `lane < 3` mask.
7. **Choose `other` from the next operation:** `0.0` add/sum, `-inf` max, `+inf` min.
8. **Grid-stride: start `pB`, step `PB`, stop at `N`.**
9. **Fuse elementwise stages; relaunch for a barrier or layout change.**

## Chapter map

| § | Mapping | Grid formula | Boundary rule | Kernel/demo |
|---|---|---|---|---|
| **18.1** | one program per vector | $P=\lceil N/B\rceil$ | `offs < N` | worked-example assert |
| **18.2** | one program per row | $P=M$ | `cols < N` | `row_scale_kernel` |
| **18.3** | one program per column | $P=N$ | `rows < M` | `column_scale_kernel` |
| **18.4** | one program per tile | $(\lceil M/B_M\rceil,\lceil N/B_N\rceil)$ | row mask $\land$ col mask | `tile_copy_kernel` |
| **18.5** | 1-D launch grid | $(P,)$ | one axis | flat grid-stride baseline |
| **18.6** | 2-D launch grid | $(P_m,P_n)$ | two axes | `tile_copy_kernel` |
| **18.7** | 3-D launch grid | $(B,P_m,P_n)$ | batch + two axes | `batch_copy_kernel` |
| **18.8** | grid-stride | $P$ fixed, step $PB$ | per-iteration mask | `grid_stride_kernel` |
| **18.9** | boundary handling | any of the above | all axes | every kernel |
| **18.10** | multiple launches | two grids in sequence | each stage own rule | `two_launch_scale_transpose` |

## Methods in the notebook

- **Method 01.** Six maps (`map_vector`, `map_rows`, `map_cols`, `map_tiles`, `map_grid_stride`, `map_batch`) plus six CPU implementations; the host skips masked lanes with `if`, the kernel masks memory.
- **Method 02.** Row/column scale, tiled copy/transpose (2-D grid, `TRANSPOSE` flag, load $iN+j$, store $jM+i$), and a 3-D batch copy (stride $MN$).
- **Method 03.** A grid-stride kernel fixes $P$ and loops with step `tl.num_programs(0) * BLOCK`; tests cover $N=35,B=4,P=2$ (one masked lane), $N=37,B=8,P=3$ (masked tail), and $N=3,P=8$ (idle programs). The pipeline computes $Z=(s\odot X)^\top$ in two launches and one fused kernel: $Q_{\text{two}}=4MNb$ versus $Q_{\text{fused}}=2MNb$.

## Choosing a mapping

- **Forces.** Occupancy, tail waste, coalescing, and launch count.
- **Formula.** Waste $=(P_mP_nB_MB_N-MN)/(P_mP_nB_MB_N)$.

| Mapping | Grid | Coalescing on row-major | Tail waste | Strength |
|---|---|---|---|---|
| per vector | $\lceil N/B\rceil$ | perfect | $O(B)$ | elementwise |
| per row | $M$ | perfect along the row | $O(MB_N)$ if $B_N\gt N$ | row reductions |
| per column | $N$ | poor (stride $N$) | $O(NB_M)$ | column reductions |
| per tile | $P_mP_n$ | perfect inside a tile row | $P_mP_nB_MB_N-MN$ | 2-D reuse |
| 3-D | $BP_mP_n$ | same as tile per batch | same as tile | batched work |
| grid-stride | $P$ fixed | same as the underlying tile | at most one tile | persistence |

- **Rule of thumb.** Start with one program per tile, or per row for row-wise reductions. Use grid-stride for a huge or dynamic grid, and keep two launches only for a global barrier or layout change.

## How to run

```bash
cd part7_construction/ch18_kernel_construction
jupyter notebook 18_complete.ipynb
# from 00_Basic:
TRITON_INTERPRET=1 PYTHONPATH=$PWD python tools/interpreter_smoke.py
```

- **CPU.** The worked example and Method 01 run; kernel tests print `SKIP <name>`.
- **Interpreter.** The notebook finds `ttools.py` by walking up to `00_Basic`.

## Study checklist

- [ ] Write the four program-to-element tables for $A$ ($5\times7$) by hand.
- [ ] State the launch-grid rule for 1-D, 2-D, and 3-D grids.
- [ ] Explain why the grid-stride stride is $PB$.
- [ ] Break the column mask and say which access fails first.
- [ ] Change a row block to `BLOCK_N < N` and describe the wrong output.
- [ ] Derive the transposed store offset $jM+i$ from $iN+j$.
- [ ] Compute two-launch versus fused traffic for $1024\times1024$ `float16`.

## Pitfalls

- **Grid too small.** `N // B` drops the tail; use `cdiv`.
- **Wrong axis order.** Swapping `p_m`/`p_n` is out-of-bounds when $M\ne N$.
- **Tail not masked.** Mask both the load and the store.
- **Too many programs.** A grid bigger than the tile count schedules idle work.
- **Grid-stride off-by-one.** Start `pB`, step `PB`, loop `< N`.
- **Block versus tile.** `tl.arange` needs a power-of-two block.
- **Fusion.** Two launches add traffic plus latency.

**Where this leads.** Chapter 19 turns the tile map into a full tiling strategy, and Chapter 20 fuses the multi-launch pipelines into one kernel.

## Exercises

1. Table for $A$ per column with $B_M=4$; which lanes are masked?
2. Program count and waste for $B_M\times B_N\in\{1\times7,5\times1,5\times7,3\times4\}$.
3. Grid and offset of program $(3,1,1)$ for $B=4$, $M=5$, $N=7$, $B_M=3$, $B_N=4$.
4. Starts and masks for $N=35$, $B=4$, $P=3$; how many tiles?
5. Delete the column mask; which access fails first on the $5\times7$ test?
6. `row_scale` with `BLOCK_N=4` on $A$: wrong output and fix it without changing `BLOCK_N`.
7. Derive $jM+i$ from $iN+j$; what if the store uses $jN+i$?
8. Two-launch versus fused traffic for $1024\times1024$ `float16`; when does launch overhead dominate?
9. $P=132$, $N=10^7$, $B=1024$: tiles per program and the last mask?
10. Scores $(B,H,S,S)$: choose a 3-D grid and name the axes.
