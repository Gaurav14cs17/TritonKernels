# Chapter 21 — Atomic and Synchronization Operations

Part 8: Advanced Triton operations and production.

> **Goal.** Treat every atomic as one indivisible read-modify-write on **one address**, then separate what a single atomic does, what happens when many programs touch the same address, and what the hardware promises about other addresses. The last step replaces the atomic with a deterministic reduction.

Notebook: [`21_complete.ipynb`](21_complete.ipynb) · Helpers: [`../../ttools.py`](../../ttools.py) · Apparatus: [Setup](../../docs/SETUP.md) · [Notation](../../docs/NOTATION.md) · [Glossary](../../docs/GLOSSARY.md) · [API index](../../docs/API_INDEX.md) · [Formula index](../../docs/FORMULA_INDEX.md) · [Diagrams](../../docs/DIAGRAMS.md) · [Troubleshooting](../../docs/TROUBLESHOOTING.md) · [Exercise hints](../../docs/SOLUTIONS.md)

[Index](../../00_index.ipynb) · [← Previous](../../part7_construction/ch20_fusion/20_complete.ipynb) · [Next →](../ch22_dot_gemm/22_complete.ipynb)

**Estimated time:** 70–85 min · **Difficulty:** Advanced

**Thread.**

- **Recurring idea.** An atomic makes one read-modify-write indivisible **per address**; it does not make a floating-point reduction deterministic.
- **Chain.** Non-atomic RMW → lost update → `atomic_add`/`atomic_max`/`atomic_min`/CAS → contention → races → memory ordering → synchronization limits → deterministic segmented reduction.
- **Payoff.** Choose between an atomic, a partitioned write, and a fixed reduction tree, and say which one is correct and which one is reproducible.

## Learning objectives

- Read `tl.atomic_add`, `atomic_max`, `atomic_min`, and `atomic_cas` as indivisible read-modify-writes and state the value each returns.
- State the **atomicity/order rule**: atomicity is per address; ordering of unrelated accesses requires a matching `release`/`acquire` pair.
- Write the lost-update interleaving and compute the final value by hand.
- Choose the correct identity (`0`, `-inf`, `+inf`, `0` flag) and detect duplicate ids before reaching for an atomic.
- Quantify contention and reduce it with per-block local accumulation.
- Replace a floating-point atomic reduction with a deterministic two-pass or sorted segmented reduction.

## Prerequisites

- Chapter 18: grids, masks, grid-stride loops.
- Chapter 20: fused kernels.
- Chapter 5: pointers, loads, stores, and the read-modify-write pattern.
- Basic concurrency vocabulary: race, atomic, barrier, lock-free.

**Connections**

- **From Chapter 5.** The load-modify-store pattern becomes one indivisible transaction.
- **Inside the chapter.** §21.1–§21.3 build the atomic RMW, §21.4 generalizes it with CAS, §21.5–§21.6 cost and failure, §21.7–§21.8 ordering and barriers, and §21.9 replaces the atomic with a fixed tree.
- **To the methods.** Method 01 models schedules on the CPU, Method 02 runs the atomic kernels, and Method 03 runs the deterministic alternatives.

## Study ladder

1. **Method 01 — host-side schedule model.** A Python replay of `load`, `add`, and `store` shows the lost update; a plain-store scatter shows last-writer wins on duplicate ids; a one-shot CAS shows success, failure, and retry.
2. **Method 02 — atomic Triton kernels.** Duplicate-id `tl.atomic_add`, global and per-bin `tl.atomic_max`/`tl.atomic_min`, an atomic histogram, and a CAS retry loop, each checked against a PyTorch reference.
3. **Method 03 — deterministic reduction.** Per-block partials plus a second pass, and a stable sort plus one program per contiguous segment; both remove the atomic from the accumulation path and fix the addition tree.

**Study tip.**

- Two programs both execute `y = y + 1` on one cell, initially `0`.
- Write the six-step interleaving in which both load `0`, both add one, and both store `1`: the final value is `1`, not `2`.
- Write the interleaving in which program 0 finishes before program 1 loads: the final value is `2`.
- Every atomic forces the second schedule; every deterministic reduction avoids the question.

## Mathematical definition

- **Atomic add.** For address $p$ and value $v$: $y_{\text{old}}\leftarrow *p$, $*p\leftarrow y_{\text{old}}+v$, and the operation returns $y_{\text{old}}$.
- **Atomicity.** No other atomic to **that address** can be placed between the read and the write; the property is local to one address.
- **Linearizability.** Every atomic takes effect at one instant between invocation and return, and all programs agree on the order; the hardware chooses that order, not the program id.
- **Max and min.** $*p\leftarrow\max(*p,v)$ and $*p\leftarrow\min(*p,v)$.
- **CAS.** $v$ when $*p=c$, otherwise $*p$; in both cases the operation returns $*p_{\text{old}}$.
- **Scatter-add.** For a bin map $\pi$: $\text{out}_j=\sum_{m:\,\pi(m)=j}v_m$.

## Worked example

Let $y_0=0$ and let two programs each request $+1$.

| step | $P_0$ | $P_1$ | `y` |
|---|---|---|---|
| 1 | `r0 = load(y) = 0` | — | 0 |
| 2 | — | `r1 = load(y) = 0` | 0 |
| 3 | `r0 = r0 + 1 = 1` | — | 0 |
| 4 | — | `r1 = r1 + 1 = 1` | 0 |
| 5 | `store(y, r0)` | — | 1 |
| 6 | — | `store(y, r1)` | 1 |

- **Lost update.** The result is $1$; one increment disappeared. If $P_0$ finishes before $P_1$ loads, the result is $2$.
- **Atomic guarantee.** `atomic_add(y, 1)` removes every interleaving inside the bracket, so the only legal orders are $P_0\prec P_1: y:0\to1\to2$ and $P_1\prec P_0: y:0\to1\to2$; the final value is $2$ and the returned values are $\{0,1\}$ in the chosen order.
- **Duplicate ids.** With `ids = [2, 0, 2]` and `values = [1, 4, 5]`, the result is $\text{out}_0=4$, $\text{out}_1=0$, $\text{out}_2=1+5=6$. A plain store races on index 2 and leaves $1$ or $5$, never $6$; an atomic add leaves $6$ for either order.
- **CAS loop.** Start $y=0$. $P_0$'s `CAS(0,1)` succeeds; $P_1$ reads $0$, its `CAS(0,1)` fails and returns $1$, then `CAS(1,2)` succeeds. The final value is $2$.
- **Order-sensitive floating point.** With $a=10^{20}$, $b=1$, $c=-10^{20}$ in `float32`: $(a+b)+c=0$ and $(a+c)+b=1$. Both are legal atomic arrival orders.

## Rules

- **Atomicity/order rule.** An atomic is indivisible only with respect to its own address; if other addresses must be visible before or after it, pair `sem="release"` with `sem="acquire"`, and use `sem="acq_rel"` for a lock or CAS. The default is `acq_rel`, not sequentially consistent.
- **Identity rule.** Initialize every atomic output to the identity of its update: `0` for add, `-inf` for max, `+inf` for min, `0` for a one-shot flag. A stale or uninitialized buffer makes a correct kernel return a wrong answer.
- **Ownership rule.** If exactly one program writes an address, no atomic is needed; otherwise partition the output, use an atomic, or accumulate per-block partials and merge in a second pass.
- **Contention rule.** $T_{\text{hot}}\approx P\tau$ and $T_{\text{spread}}\approx(P/K)\tau$; local accumulation changes $N$ atomics to $PK$ and the hot-address load from $N$ to $P$, so $T_{\text{naive}}/T_{\text{local}}\approx B$.
- **Determinism rule.** Atomics give a correct approximate sum, not a reproducible bit pattern. Reproducibility needs a fixed tree: two-pass partials or sorted segments. Integer addition is associative, so an integer histogram is reproducible even when its merge uses atomics.
- **Synchronization rule.** There is no grid-wide barrier inside a normal Triton kernel; `tl.debug_barrier()` is CTA-wide. The portable barrier is the kernel launch boundary, and a spin loop waiting for a non-resident producer deadlocks.

## Chapter map

| § | Operation | Rule | Failure mode | Kernel/demo |
|---|---|---|---|---|
| **21.1** | `tl.atomic_add()` | $y\mathrel{+}=v$, returns old $y$ | lost update without the atomic | duplicate-id scatter-add |
| **21.2** | `tl.atomic_max()` | $y\leftarrow\max(y,v)$ | initialized to `0` for negative data | global/per-bin extrema |
| **21.3** | `tl.atomic_min()` | $y\leftarrow\min(y,v)$ | initialized to `0` for positive data | global/per-bin extrema |
| **21.4** | Compare-and-swap | write only if $*p=c$ | ABA, no reload after failure | CAS increment counter |
| **21.5** | Atomic contention | $T\approx(\text{conflicts/address})\tau$ | one atomic per lane on a hot address | local-bin histogram |
| **21.6** | Race conditions | load-add-store is not indivisible | schedule-dependent lost update | Method 01 model |
| **21.7** | Memory ordering | `relaxed` < `acquire`/`release` < `acq_rel` | relaxed flag orders nothing else | release/acquire publish |
| **21.8** | Synchronization limitations | no grid-wide barrier in one kernel | spin-barrier deadlock | two-pass reduction |
| **21.9** | Safe reduction alternatives | fixed tree or sorted segments | atomic sum not reproducible | segmented scatter-add |
| **★** | Capstone | local bins + one atomic per bin | contention on the merge | 64-bin histogram |

## How to run

```bash
cd part8_advanced/ch21_atomics
jupyter notebook 21_complete.ipynb
```

- **CPU.** The schedule model, worked example, contention arithmetic, and deterministic reference run; GPU tests print `SKIP <name>: ...`.
- **Interpreter smoke test.** `cd 00_Basic && TRITON_INTERPRET=1 PYTHONPATH=$PWD python tools/interpreter_smoke.py`.
- **Atomic support.** Triton 3.7's interpreter supports `atomic_add`, `atomic_max`, `atomic_min`, `atomic_cas`, `sem`, and `scope`; if an older interpreter lacks atomics, the atomic tests are GPU-only and Method 03 is the CPU-safe reference. The interpreter runs programs one at a time, so it checks values and API use, not hardware interleavings or contention.
- **Worked assertions.** The notebook checks the lost-update result $1$, the serialized result $2$, the duplicate-id output `[4, 0, 6]`, the CAS retry, the `float32` order example, and the contention numbers $P=4096$, $PK=262{,}144$, $4\times$ fewer atomics, and $256\times$ less hot-address serialization.

## Pitfalls

| Pitfall | What happens | Fix |
|---|---|---|
| Non-atomic read-modify-write | duplicate ids lose all but the last update | `tl.atomic_add`, partition, or two-pass |
| Atomic sum for reproducibility | arrival order changes the floating-point tree | two-pass or segmented reduction |
| Duplicate ids in a plain store | last writer wins, silently | detect collisions or use an atomic |
| One atomic per lane on a hot address | serialization proportional to $N$ | local bins, one atomic per bin per block |
| CAS ABA | a value returns to its old state and the stale CAS succeeds | version tag or monotonic counter |
| Missing barrier | there is no grid-wide barrier to miss inside one kernel | second kernel launch |
| Atomics for a full reduction | correct, slow, non-reproducible | `tl.sum` plus partials |
| Wrong identity or mask | `0` is wrong for max/min in general; padded lanes write bin 0 | choose the identity and pass the mask |

## Study checklist

- [ ] Write the two interleavings and their final values from memory.
- [ ] State why `+=` is not atomic in Triton.
- [ ] Give the identity for add, max, min, and a one-shot flag.
- [ ] Compute the contention model for $N=2^{20}$, $B=256$, $K=64$.
- [ ] Explain why a successful CAS does not order unrelated memory.
- [ ] Explain why a grid-wide spin barrier can deadlock.
- [ ] Choose two-pass versus sorted segments for a duplicate-id scatter.
- [ ] Break the identity or the mask and predict the wrong answer.
- [ ] Run the CPU cells and the interpreter smoke test.
- [ ] Do exercises 5, 7, and 10: ABA, determinism, and ownership.

**Where this leads.**

- **Part II.** Model the lost update, the serialized scatter, and the CAS retry on the CPU.
- **Part III.** Run the same operations as atomic Triton kernels and check them against PyTorch.
- **Part IV.** Replace the atomic with two-pass partials or sorted segments when the bits must repeat.

## Exercises

1. **Lost update.** Enumerate all six interleavings of two non-atomic increments; which produce `1`, and which produce `2`? Verify with `non_atomic_rmw`.
2. **Return values.** Store every old value from the scatter-add and show that they are a permutation of the prefix sums, not program order.
3. **Extrema identity.** Run the extrema kernel with the wrong identity on all-negative and all-positive tensors; explain both failures.
4. **CAS lock.** Implement a one-bit spin lock with `atomic_cas`; explain why a lock held by a non-resident program deadlocks.
5. **ABA.** Construct a schedule where `0 -> 1 -> 0` lets a stale CAS succeed; add a version tag and show that the stale CAS now fails.
6. **Contention.** Model $N=2^{22}$, $B=256$, `NBINS` in `{8, 64, 256}`; for which configuration does $PK\gt N$?
7. **Deterministic scatter.** Sort once and launch one program per segment; compare its bit pattern with the atomic version on `float32`.
8. **Two-pass traffic.** Derive the extra bytes read and written by the two-pass sum for $N=2^{20}$, `BLOCK=256`, `float32`.
9. **Scope.** Change the release/acquire pair to `scope="cta"`. Does the test still pass? When would it be wrong?
10. **Ownership.** With unique ids, delete every atomic, prove that no two programs write the same address, and compare the runtime.
