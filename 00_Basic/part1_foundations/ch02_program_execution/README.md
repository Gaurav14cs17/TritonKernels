# Chapter 2 — Triton Program Execution

Part 1: Foundations: what Triton is, how programs run, and how vectors are built.

**Thread.** One launch, one tile, one mask.

- The host chooses a grid, and each program receives one scalar coordinate per grid axis.
- `tl.program_id` turns that coordinate into a tile; `tl.arange` turns the tile into lanes.
- A mask removes dead lanes, and `tl.num_programs` lets one program walk several tiles.
- The same moves compose into 1-D, 2-D, and 3-D work.

**Connections.**

- **Builds on Chapter 1:** host vs device code, `@triton.jit`, and `kernel[grid](...)`.
- **Uses:** integer ceiling, div/mod, and row-major matrix indexing.
- **Feeds Chapter 3:** the tile coordinates here become the vectors built by `tl.arange`, `tl.full`, and `tl.zeros`.
- **Leads to:** tiling, grid-stride kernels, and every later chapter that launches more than one program.

[Open the chapter notebook](02_complete.ipynb) · [Index](../../00_index.ipynb) · [← Previous](../ch01_introduction/01_complete.ipynb) · [Next →](../ch03_vector_construction/03_complete.ipynb)

**Estimated time:** 65–85 min · **Difficulty:** Beginner–Intermediate

## Learning objectives
- Read `tl.program_id(axis)` as a scalar grid coordinate in `$[0,P_\text{axis})$`, not a thread id.
- Derive 1-D, 2-D, and 3-D launch grids from `$\lceil N/B\rceil$` and the product rule.
- Map a program id to a tile and to matrix coordinates, including the inverse map.
- Distinguish a Triton program (one CTA) from a CUDA thread and use `num_warps`.
- Write masked 1-D and 2-D kernels that cover every element exactly once.
- Write a grid-stride loop with `tl.num_programs(0)` and prove its coverage.

## Prerequisites
- Chapter 1: host vs device code, `@triton.jit`, and launch syntax.
- Integer division and ceiling; basic matrix indexing.

## Study ladder
- **Concept:** grid coordinates and a CPU tile map.
- **Numeric demo:** the pid/offsets/mask table for `$N=10$`, `$B=4$`.
- **Micro-kernel:** 1-D vector add and 2-D transpose.
- **Applied use:** a grid-stride kernel where one program owns several tiles.

**Study tip.**

- Fix `$N=10$` and `BLOCK=4`.
- Write `pid`, `offs = pid*4 + [0,1,2,3]`, and `mask = offs < 10` by hand.
- Program 2 has offsets `$8,9,10,11$`, mask `$T,T,F,F$`, and owns only `$8,9$`.
- If your table matches the worked example, every kernel is the same eight lines with a different pointer expression.

## Mathematical core
- **Rule.** A 1-D grid `(P,)` gives one scalar `$p$` per program; a 2-D grid `(P_m,P_n)` gives `$(p_m,p_n)$`; a 3-D grid gives `$(p_m,p_n,p_b)$`.
- **Formula.** For a vector of length `$N$` and block `$B$`, `$P=\lceil N/B\rceil=\lfloor(N-1)/B\rfloor+1=(N+B-1)//B$`.
- **Formula.** For a matrix and a batch, `$P_m=\lceil M/B_M\rceil$`, `$P_n=\lceil N/B_N\rceil$`, and `$P=P_mP_nP_b$`; for example, `$N=10$`, `$B=4$` gives `$P=3$`, and `$M=5$`, `$N=7$`, `$B_M=2$`, `$B_N=3$` gives `$(3,3)$`.
- **Triton.** Use `tl.program_id(axis)` for the coordinate and `tl.num_programs(axis)` for the extent; the host chooses the grid, and the kernel turns each coordinate into a tile.
- **Failure.** Over-covering is safe only when the tail is masked; under-covering silently drops work.

### Grid shapes
- **Rule.** Axis 0 = row tile, axis 1 = column tile, axis 2 = batch; any permutation is legal, but host and kernel must agree.
- **Formula.** Addresses are `$pB+j$` for 1-D, `$rN+c$` for 2-D, and `$bMN+rN+c$` for 3-D.

### `tl.num_programs` and the CTA model
- **Rule.** `tl.num_programs(axis)` is the launch extent read inside the kernel; it is a scalar constant, unlike `tl.program_id(axis)`.
- **Formula.** Grid-stride step `$P_0B$`; tile label `$t=p_mP_n+p_n$`.
- **Example.** `$N=10$`, `$B=4$`, `$G=2$` gives stride `$8$`: program 0 starts at `$0,8$`, program 1 at `$4$`.
- **Rule.** One program is one CUDA CTA (block); with `num_warps=W`, there are `$32W$` threads per CTA and about `$\text{tile elements}/(32W)$` elements per thread. For `$B=256$`, `$W=4$`, that is 128 threads and about two elements per thread.
- **Failure.** Equating `BLOCK` with thread count, or sizing the grid by CUDA threads.

## Rules

### Grid formulas

| Situation | Rule | Example |
|---|---|---|
| 1-D, length `$N$`, block `$B$` | `$P=\lceil N/B\rceil$` | `$N=10,B=4\Rightarrow P=3$` |
| 2-D, `$M\times N$`, tile `$B_M\times B_N$` | `$(\lceil M/B_M\rceil,\lceil N/B_N\rceil)$` | `$M=5,N=7,B_M=2,B_N=3\Rightarrow(3,3)$` |
| 3-D, batch `$B$` | add `$P_b=B$` | `$B=2\Rightarrow(3,3,2)$` |
| zero-size tensor | launch `(0,)` or return early | `$N=0\Rightarrow P=0$` |

### pid ranges

| Axis | Range | Meaning |
|---|---|---|
| 0 | `$0\le p\lt P_0$` | row tile, or flat tile |
| 1 | `$0\le p\lt P_1$` | column tile |
| 2 | `$0\le p\lt P_2$` | batch / head |

### Tile mapping

| Mapping | Index for program `$p$`, lane `$j$` | Use |
|---|---|---|
| block-contiguous | `$i=pB+j$` | 1-D elementwise |
| grid-stride | `$i=(p+kP)B+j$` | persistent / small grid |
| row-per-program | `$i=pN+j$` | row-wise normalization |
| 2-D tile | `$i=(p_mB_M+i_r)N+(p_nB_N+j_c)$` | matrix tiles |

- **Formula.** Inverse row-major map: `$p_m=\lfloor t/G_n\rfloor$`, `$p_n=t\bmod G_n$`, and `$t=p_mG_n+p_n$`; `$t$` is a software label, not the hardware's block-scheduling order.

### Coverage and padding
- **Formula.** `$[0,N)=\bigsqcup_p I_p$` with `$I_p=[pB,(p+1)B)\cap[0,N)$`.
- **Rule.** Reads may overlap safely; writes need exactly one owner or `tl.atomic_add`.
- **Triton.** There is no inter-program synchronization; a second launch is the portable barrier.
- **Rule.** `tl.arange(0,k)` requires a power-of-two `$k$`; a width-3 pencil tile is padded to 4 lanes and masked by `cols < N`.
- **Example.** The worked example uses `$B_N=3$` and a `$(3,3)$` grid; the transpose kernel uses `$B_N=4$` and a `$(3,2)$` grid.

## Worked numerical example

### 1-D: `$N=10$`, `$B=4$`
- **Rule.** `$P=\lceil10/4\rceil=3$`.

| `pid` | `offs` | `mask` | elements owned |
|---:|---|---|---|
| 0 | 0,1,2,3 | T T T T | 0,1,2,3 |
| 1 | 4,5,6,7 | T T T T | 4,5,6,7 |
| 2 | 8,9,10,11 | T T F F | 8,9 |

- **Vector.** The same ownership picture, with `·` for a dead lane:

$$
x=(0,1,2,3,\;4,5,6,7,\;8,9,\cdot,\cdot),
\qquad
\underbrace{0\,1\,2\,3}_{p=0}\;
\underbrace{4\,5\,6\,7}_{p=1}\;
\underbrace{8\,9}_{p=2}.
$$

- **Ownership.** The last program has two dead lanes; its load and store are masked, so the output has exactly 10 elements.

### 2-D: `$M=5$`, `$N=7$`, `$B_M=2$`, `$B_N=3$`
- **Rule.** `$P_m=\lceil5/2\rceil=3$`, `$P_n=\lceil7/3\rceil=3$`, grid `$(3,3)$`: 9 programs.

- **Tiles.** Rows are `$p_m\cdot2,\ldots,p_m\cdot2+1$`, columns `$p_n\cdot3,\ldots,p_n\cdot3+2$`, clipped at `$5$` and `$7$`; counts are `$[[6,6,2],[6,6,2],[3,3,1]]$`.

- **Matrix.** Entry `$T_{rc}=p_mG_n+p_n$` is the tile that owns cell `$(r,c)$`; `$C_{p_mp_n}$` counts live cells:

$$
T=
\begin{bmatrix}
0&0&0&1&1&1&2\\
0&0&0&1&1&1&2\\
3&3&3&4&4&4&5\\
3&3&3&4&4&4&5\\
6&6&6&7&7&7&8
\end{bmatrix},
\qquad
C=
\begin{bmatrix}
6&6&2\\
6&6&2\\
3&3&1
\end{bmatrix}.
$$

- **Counts.** The counts sum to `$35=5\cdot7$`; the notebook asserts `$T$`, `$C$`, and the sum rather than printing them.

## Chapter map

| § | Topic | Key formula | Shape/grid rule | Kernel/demo |
|---|---|---|---|---|
| **2.1** | `tl.program_id(0)` | `$i=pB+j$` | grid `(P,)`, `$p\in[0,P)$` | pid range demo |
| **2.2** | `tl.program_id(1)` | `$(r,c)=(p_mB_M+i,\ p_nB_N+j)$` | grid `(P_m,P_n)` | 2-D coordinate table |
| **2.3** | `tl.program_id(2)` | `$\text{addr}=bMN+rN+c$` | grid `(P_m,P_n,P_b)` | 3-D coordinate table |
| **2.4** | `tl.num_programs(0)` | `$\text{step}=P_0B$` | grid extent, not work size | grid-stride loop |
| **2.5** | 1-D program grid | `$P=\lceil N/B\rceil$` | one tile per program | vector add |
| **2.6** | 2-D program grid | `$P=(P_m,P_n)$` | two masks: rows and columns | transpose |
| **2.7** | 3-D program grid | `$P=(P_m,P_n,P_b)$` | batch on the slow axis | batched tile table |
| **2.8** | Program IDs and matrix coordinates | `$p_m=\lfloor t/G_n\rfloor,\ p_n=t\bmod G_n$` | label ↔ tile | inverse map demo |
| **2.9** | Programs vs CUDA threads and blocks | `$\text{threads/CTA}=32W$` | `num_warps` ≠ `BLOCK` | thread-count table |
| **2.10** | Launch-grid calculation | `$P=\lceil N/B\rceil=(N+B-1)//B$` | multiply per-axis counts | cdiv table |
| **2.11** | Multiple programs on one tensor | `$[0,N)=\bigsqcup_p I_p$` | reads may overlap, writes may not | ownership sets |
| **2.12** | Program-to-element mapping | four mappings above | last axis is the lane axis | mapping comparison |

## Canonical kernel
- **Rule.** One masked tile per program; the 2-D variant replaces `offs` with `rows[:, None] * N + cols[None, :]` and the mask with `(rows[:, None] < M) & (cols[None, :] < N)`.

```python
pid = tl.program_id(0)
offs = pid * BLOCK + tl.arange(0, BLOCK)
mask = offs < n_elements
tl.store(out_ptr + offs, tl.load(x_ptr + offs, mask=mask) + tl.load(y_ptr + offs, mask=mask), mask=mask)
```

## Pitfalls

| Pitfall | What happens | Fix |
|---|---|---|
| off-by-one tail | `offs <= N` includes one extra lane | zero-based: `offs < N` |
| `P = N // B` | tail elements never launch | `P = cdiv(N, B)` |
| swapped pid axes | transposed tiles, wrong/OOB addresses | keep `(Pm, Pn)` order |
| row mask only | last column tile crosses into the next row | mask rows **and** columns |
| 1-D vs 2-D flattening | `pm*BM + pn*BN + j` is not `r*N + c` | `rows[:, None]*N + cols[None, :]` |
| `pid` treated as a thread | grid sized by elements, huge launch | one program = one CTA; pad width-3 `tl.arange` to 4 |

## How to run

```bash
cd part1_foundations/ch02_program_execution
jupyter notebook 02_complete.ipynb
# CPU-only: numerical cells run, Triton tests print SKIP
TRITON_INTERPRET=1 PYTHONPATH=../../ python ../../tools/interpreter_smoke.py
```

- **CPU.** Every numerical cell and Method 01 test runs; Triton tests print `SKIP <name>`.
- **Interpreter and timing.** The harness prepends `ttools.HAS_GPU=True`; `ttools.bench` returns `nan` without CUDA, so this chapter reports no timings.

## Study checklist
- [ ] Write the pid/offsets/mask table for `$N=10$`, `BLOCK=4` without looking.
- [ ] Compute the grid and tile map for `$M=5$`, `$N=7$`, `$B_M=2$`, `$B_N=3$`.
- [ ] Convert `$t=7$` to `$(p_m,p_n)$` for a `$3\times3$` tile grid and back.
- [ ] Explain why a Triton program is a CTA, not a CUDA thread; give one safe overlapping read and one unsafe overlapping write.
- [ ] Derive the grid-stride step and prove every tile is visited once.
- [ ] Explain why `tl.arange(0,3)` fails and how to pad a width-3 tile.

## Exercises
- **Table and grid drills.** Do `$N=13$`, `BLOCK=5`; `$M=9$`, `$N=11$`, `$B_M=4$`, `$B_N=3$`; and `$B=4$`, `$M=8$`, `$N=8$`, `$B_M=4$`, `$B_N=4$`.
- **Inverse and axes.** Convert `$t=13$` for `$G_m=4$`, `$G_n=5$`; then swap `program_id(0)` and `program_id(1)` and find the first out-of-bounds access.
- **Grid and mask bugs.** Launch `N // BLOCK` programs; remove the column mask from the `$B_N=4$` transpose.
- **Grid-stride and threads.** Run `programs=1` and `programs=3` for `$N=10$`, `BLOCK=4`; then size `BLOCK=128`, `num_warps=8`.
- **Row-per-program.** Write the kernel and explain why the grid is exactly `$M$`.

Hints are in [`../../docs/SOLUTIONS.md`](../../docs/SOLUTIONS.md).

**Where this leads.** [Chapter 3 — Triton Vector Construction](../ch03_vector_construction/03_complete.ipynb) turns these coordinates into vectors with `tl.arange`, `tl.full`, `tl.zeros`, and `tl.cast`.
