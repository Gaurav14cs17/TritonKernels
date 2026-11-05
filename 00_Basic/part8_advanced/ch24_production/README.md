# Chapter 24 — Production and Profiling

Part 8: Advanced Triton operations and production.

> **Goal.** Turn a Triton kernel into a production op: integrate it with PyTorch,
> wrap it in `torch.autograd.Function`, prove the gradient is correct, tune and
> benchmark it, and explain the result with the roofline and occupancy models.
> **Correctness first, then timing; a fast wrong kernel is a regression.**

Notebook: [`24_complete.ipynb`](24_complete.ipynb) · Helpers:
[`../../ttools.py`](../../ttools.py) · Apparatus:
[Setup](../../docs/SETUP.md) · [Notation](../../docs/NOTATION.md) ·
[Glossary](../../docs/GLOSSARY.md) · [API index](../../docs/API_INDEX.md) ·
[Formula index](../../docs/FORMULA_INDEX.md) · [Diagrams](../../docs/DIAGRAMS.md) ·
[Troubleshooting](../../docs/TROUBLESHOOTING.md) · [Exercise hints](../../docs/SOLUTIONS.md)

[Index](../../00_index.ipynb) · [Chapter notebook](24_complete.ipynb) · [← Previous](../ch23_transformers/23_complete.ipynb)

**Estimated time:** 100–130 min · **Difficulty:** Advanced

## Learning objectives
- State the `torch.autograd.Function` contract: saved state, returned gradient, and the shape/dtype/device invariants.
- Write a Triton forward kernel and its vector–Jacobian backward kernel, then gate both with `torch.testing.assert_close`.
- Use `torch.autograd.gradcheck` in `float64` to validate the analytic gradient.
- Choose an autotune key and a benchmark protocol (warm-up, synchronize, device timers, median).
- Read a roofline point and an occupancy limit, and keep the same source valid on NVIDIA and AMD.
- Apply the correctness/benchmark rule before reporting any latency.

## Prerequisites
- Chapter 23: transformer kernels and the forward/backward pattern.
- PyTorch autograd: `requires_grad`, `backward`, `grad`, and the chain rule.
- Chapter 18: 1-D launch grids, `triton.cdiv`, and tail masks.
- No prior Nsight or autotuning experience assumed.

## Study ladder
- **Method 01 — plain PyTorch and a manual VJP.** A scalar loop defines `y=x^2` and `dx=2xg`; it is slow and it is the oracle.
- **Method 02 — Triton forward/backward inside `torch.autograd.Function`.** The same maps become two kernels; `assert_close`, `gradcheck`, tail tests, and non-contiguous tests gate the result.
- **Method 03 — tune, benchmark, and profile.** `@triton.autotune` picks a config per runtime key; warm-up/event/median measures it; the roofline and occupancy worksheets explain the number.

## **Connections**
- **Backward.** Builds on Chapter 18's grid/mask skeleton and Chapter 23's forward/backward pattern.
- **Forward.** Feeds a verified custom op back into training loops, profiling runs, and multi-vendor deployment.
- **Thread.** The backward is the local Jacobian applied to the upstream gradient; correctness gates every later timing number.
- **Chain.** PyTorch integration -> `torch.autograd.Function` -> forward/backward kernels -> gradient checks -> autotune -> benchmark -> Nsight -> roofline/occupancy -> portability.
- **Pattern.** Each Part I subsection follows Connection -> Rule -> Formula -> Example -> Where -> Triton -> Failure -> Bridge.
- **Reading path.** Start at 24.1 for the API boundary, then 24.2–24.4 for the contract, 24.5–24.6 for the gates, 24.7–24.12 for measurement, and 24.13 for portability.
- **Notebook flow.** Part I is the rule book; Part II is the scalar oracle; Part III is the Triton wrapper; Part IV is tuning, benchmarking, and profiling; Part V is portability.

## The production contract
- **Contract.** `forward` saves only the tensor the derivative needs; `backward` receives the upstream gradient and returns one gradient per forward input, with the input's shape, dtype, and device.

```python
class Square(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x * x

    @staticmethod
    def backward(ctx, grad_out):
        (x,) = ctx.saved_tensors
        return 2.0 * x * grad_out
```
- **Correctness.** `assert_close` checks $|a_i-e_i|\le\operatorname{atol}+\operatorname{rtol}|e_i|$ per element; `gradcheck` compares the analytic VJP with the central difference $\hat g_i=\frac{f(x+\varepsilon e_i)-f(x-\varepsilon e_i)}{2\varepsilon}$ in `float64`.
- **Roofline.** With $\mathrm{AI}=F/Q$ and ridge $\mathrm{AI}^\star=P_{\text{peak}}/\mathrm{BW}$, time is bounded by $t\ge\max(F/P_{\text{peak}},Q/\mathrm{BW})$ and attainable performance by $P_{\text{attain}}\le\min(P_{\text{peak}},\mathrm{BW}\cdot\mathrm{AI})$.
- **Occupancy.** With warp size $w$, $B$ resident blocks, and $W_{\max}$ warps/SM: $\text{warps/block}=\text{threads/block}/w$, $B=\min(B_{\text{reg}},B_{\text{smem}},B_{\text{warp}},B_{\text{hw}})$, and $\text{occupancy}=B\cdot\text{warps/block}/W_{\max}$.

## Worked numerical example
- **Square.** The tiny kernel is the same map that Method 02 wraps in `torch.autograd.Function`:

$$
y=x^2,\qquad x=(1,2,3,4),\qquad y=(1,4,9,16),\qquad dx=2x=(2,4,6,8).
$$

- **Backward.** With an upstream gradient of ones, the manual backward is $\partial L/\partial x_i=2x_i=(2,4,6,8)$.
- **Gradient check.** At $x=2$ with $h=10^{-6}$, $\frac{(2+h)^2-(2-h)^2}{2h}=\frac{4h}{2h}=4=2x$.
- **Tolerance.** The analytic and numerical values differ only by rounding, so the check passes with $|g-\hat g|\le\operatorname{atol}+\operatorname{rtol}|\hat g|=10^{-5}+10^{-3}\cdot4=4.01\times10^{-3}$.
- **Roofline bytes.** For $N=2^{20}$ `float32`, a square reads one array and writes one array: $Q=2\cdot4\cdot N=8{,}388{,}608$ bytes and $F=N=1{,}048{,}576$ FLOPs.
- **Roofline intensity.** $\mathrm{AI}=F/Q=0.125$ and $\mathrm{AI}^\star=P_{\text{peak}}/\mathrm{BW}=20\times10^{12}/1.6\times10^{12}=12.5$.
- **Roofline bound.** Since $0.125\ll12.5$, the kernel is memory-bound: $t_{\text{mem}}=Q/\mathrm{BW}=5.24\ \mu s$, $t_{\text{compute}}=F/P_{\text{peak}}=0.052\ \mu s$, and $t\ge5.24\ \mu s$.
- **Occupancy.** With 64 registers/thread, 256 threads/block, warp size 32, register file 65,536 registers/SM, and $W_{\max}=64$ warps/SM: $\text{warps/block}=8$, $B_{\text{reg}}=\lfloor65{,}536/(64\cdot256)\rfloor=4$, and $B_{\text{warp}}=\lfloor64/8\rfloor=8$.
- **Occupancy result.** With no shared memory and a hardware block limit above 4, $B=4$ and $\text{occupancy}=4\cdot8/64=0.5$.

## **Rules**
- **Correctness first.** Assert the forward and backward before timing; a fast wrong kernel ships a regression.
- **Save the input.** Use `ctx.save_for_backward(x)`, not `ctx.x`; `backward` needs state.
- **Match the input.** Return `dx` with the input's shape, dtype, and device; autograd rejects or silently broadcasts a mismatch.
- **`gradcheck` in `float64`.** Use `eps=1e-6` and small tensors; `float32` noise causes false failures.
- **Tolerance by dtype.** Use fp32 `1e-5` and fp16/bf16 `1e-2`; tighter tolerances produce false failures, looser ones hide errors.
- **Autotune key.** Include every runtime shape that changes the winner; otherwise a config is reused for the wrong size.
- **Benchmark protocol.** Warm up, synchronize, use device timers, and report a median; wall clock measures the launch.
- **Roofline units.** Use one system for bytes and FLOPs and count every HBM access; otherwise the point moves across the ridge.
- **Occupancy limiter.** Compute registers, shared memory, warp slots, and the block limit; do not assume 100% occupancy.
- **Portable warp size.** Derive threads from `num_warps * w`; hard-coding 32 breaks AMD.

## Chapter map
| § | Topic | Rule/formula | Check | Demo |
|---|---|---|---|---|
| **24.1** | PyTorch integration | $Y=K(X)$ preserves shape, dtype, device | output metadata | manual square |
| **24.2** | `torch.autograd.Function` | $\partial L/\partial X=(\partial L/\partial Y)(\partial Y/\partial X)$ | contract + `gradcheck` | `ManualSquare` |
| **24.3** | Forward/backward kernels | $dx_i=2x_i g_i$ | tail, non-contiguous input | `TritonSquare` |
| **24.4** | Gradient correctness | $E\le\mathrm{atol}+\mathrm{rtol}\max|\hat g|$ | autograd oracle | forward/backward test |
| **24.5** | `assert_close` | $|a_i-e_i|\le\mathrm{atol}+\mathrm{rtol}|e_i|$ | accept and reject cases | tolerance gate |
| **24.6** | `gradcheck` | central difference in `float64` | `gradcheck` returns true | contract test |
| **24.7** | Autotuning | $c^\star=\arg\min_c\operatorname{median}(t_c)$ | key contains the shape | `tuned_scale` |
| **24.8** | Benchmarking | $t_{\text{med}}$, $\mathrm{BW}_{\text{eff}}=Q/t$ | correctness before timing | `test_benchmark` |
| **24.9** | Nsight Compute | counters: memory, compute, stalls | one kernel at a time | profiler checklist |
| **24.10** | Nsight Systems | $T=\sum T_k+T_{\text{gaps}}+T_{\text{copies}}$ | timeline has no gaps | NVTX ranges |
| **24.11** | Roofline | $t\ge\max(F/P_{\text{peak}},Q/\mathrm{BW})$ | intensity versus ridge | roofline worksheet |
| **24.12** | Occupancy | $B=\min(B_{\text{reg}},B_{\text{smem}},B_{\text{warp}},B_{\text{hw}})$ | limiter and fraction | occupancy worksheet |
| **24.13** | Portability | $\text{threads}=\texttt{num\_warps}\cdot w$ | NVIDIA $w=32$, AMD $w=64$ | portability asserts |
| **★** | Capstone | correctness → gradient → tune → benchmark → profile | every gate green | production checklist |

## Common failure modes
- **Missing `ctx.save_for_backward(x)`.** `backward` has no input; autograd raises or the derivative is wrong.
- **Wrong gradient shape.** Return the same shape, dtype, and device as the forward input.
- **`gradcheck` on `float32`.** Finite differences are dominated by rounding; use `float64` and tiny inputs.
- **Benchmark without warm-up or synchronize.** Compilation and cold caches dominate, or the timer stops before the kernel finishes.
- **Autotune key missing a shape.** A config tuned for one size is reused for another.
- **Grid hard-codes `BLOCK`.** The winning config changes `BLOCK`, but the grid does not; use `lambda meta: (cdiv(n, meta["BLOCK"]),)`.
- **Roofline unit error.** Fix one system for bytes and FLOPs and count every HBM byte.
- **Occupancy assumption.** Compute the register, shared-memory, warp, and block limits; do not assume 100%.
- **Non-contiguous `grad_out`.** Call `.contiguous()` before the kernel.
- **Hard-coded warp size.** Derive threads from `num_warps * warp_size`; 32 is not universal.

## How to run
- **Notebook.** From this directory, start `jupyter notebook 24_complete.ipynb`.
- **CPU-only.** The worked example, Method 01, the contract `gradcheck`, the tolerance gate, the autotune selection, the roofline/occupancy worksheets, and the portability asserts run; each GPU test prints `SKIP <name>: ...`.
- **Interpreter smoke test.**

```bash
cd 00_Basic
TRITON_INTERPRET=1 PYTHONPATH=$PWD python tools/interpreter_smoke.py
```

- **GPU tools.** The autotuner needs a live driver; `test_tuned_scale` is a no-op without CUDA/ROCm. All other kernels run under the CPU interpreter.

```bash
ncu --set full -o report python train_step.py
nsys profile --trace=cuda,nvtx -o trace python train_step.py
```

## Study checklist
- [ ] Write the `forward`/`backward` signature and the shape rule from memory.
- [ ] Check the manual VJP $2x$ at $x=2$ with a central difference.
- [ ] Run `gradcheck` on `ManualSquare` in `float64` and explain the tolerance.
- [ ] Break the tail mask and show which lane fails at $N=1003$, `BLOCK=256`.
- [ ] Feed a transposed tensor and a zero-stride `.sum()` gradient; explain both `.contiguous()` calls.
- [ ] Compute the roofline point for $N=2^{20}$ `float32` and the occupancy for 64 registers/thread.
- [ ] Benchmark with and without warm-up and synchronization; explain the gap.
- [ ] Name two NVIDIA/AMD differences that can change a launch.

## Where this leads
- **Next step.** Carry the same correctness-then-measurement loop into any custom kernel you ship.
- **Production loop.** Integration, autograd, gradient checks, tuning, benchmarking, profiling, and portability are one pipeline, not seven unrelated topics.

## Exercises
1. **Contract audit.** Add a `scale` argument to `TritonSquare`; return the correct gradient for both inputs and say which gradients are `None`.
2. **Tail hunt.** Change `BLOCK` from 256 to 64 and test $N=1003$; which lane would be wrong if the backward forgot its mask?
3. **Gradcheck tolerances.** Run `gradcheck` with `eps=1e-4`, `1e-6`, and `1e-8`; explain the U-shaped error curve.
4. **Non-contiguous gradient.** Feed a transposed tensor and a zero-stride expanded gradient; which `.contiguous()` calls are required?
5. **Autotune key.** Add a dtype or shape component to the key; show what goes wrong when it is missing.
6. **Benchmark protocol.** Time `bench_scale` with and without warm-up and with and without `synchronize`; explain the gap.
7. **Roofline.** Compute the roofline point for a fused $y=ax+b$ kernel with $N=2^{20}$ `float32`; how many bytes and FLOPs, and which roof binds?
8. **Occupancy.** Vary registers/thread from 16 to 128 at 256 threads/block; at what value does the register file become the limiter?
9. **Portability.** Rewrite the grid so the same source works with warp size 32 and 64; what changes, and what must not change?
10. **Profiling.** Run `ncu` on `bench_scale` and `nsys` on the forward/backward pair; which answers "why is this kernel slow", and which "where did the time go"?

## Summary
- **Contract.** A Triton kernel becomes a trainable op through `torch.autograd.Function`: save the input, return the VJP, match shape/dtype/device.
- **Backward.** The backward is another elementwise kernel with the same grid and mask; tail and non-contiguous gradients are part of the contract.
- **Gate.** `assert_close` is the per-element gate; `gradcheck` is the finite-difference gate in `float64`; both run before timing.
- **Tune.** Autotuning chooses a config per runtime key; benchmarking uses warm-up, synchronization, device timers, and a median.
- **Explain.** The roofline decides memory versus compute; occupancy decides latency hiding; neither replaces a measured number.
- **Ship.** Portability is source discipline: no hard-coded warp size, no vendor-only intrinsics in the common path, and the same correctness gate everywhere.
