# TritonKernels

A **study pack** for [OpenAI Triton](https://triton-lang.org/) GPU kernels, mirroring the official [Getting Started tutorials](https://triton-lang.org/main/getting-started/tutorials/index.html).

Each numbered folder is one lesson with the same layout:

- **README.md** — goal, study ladder, math, whiteboard figures, how to run
- **`tutorial.py`** — naive → intermediate → Triton (`tl`)
- **`<folder>.ipynb`** — same ladder with LaTeX + runnable cells
- **`docs/visual-guide.md`** — mental model + figures
- **`docs/README.md`** — deeper formulas, pitfalls, exercises
- **`docs/images/`** — whiteboard theory boards (method / logic)

**Ladder everywhere:** Naive Python / pseudocode → intermediate step → Triton kernel.

---

## How to study (recommended path)

| Day / block | Folder | Focus until you can … |
|-------------|--------|------------------------|
| 1 | [01](01_vector_addition/) | Explain `pid`, `offsets`, **mask**, and why vector-add is memory-bound |
| 1–2 | [02](02_fused_softmax/) | Count HBM traffic: multi-pass vs fused row softmax |
| 2–3 | [03](03_matrix_multiplication/) | Draw blocked GEMM + $B_K$ + **grouped** pid (richest visuals) |
| 3 | [04](04_low_memory_dropout/) | Explain why a seed beats a full mask tensor |
| 3–4 | [05](05_layer_normalization/) | Fuse mean / var / affine on one row |
| 4–5 | [06](06_fused_attention/) | Write online-softmax updates; why no full $S=QK^\top$ |
| 5 | [07](07_libdevice/) | When `libdevice` beats a hand poly (accuracy vs FLOPs) |
| 5–6 | [08](08_group_gemm/) | Contrast Group GEMM vs `bmm` vs tutorial-03 “grouped” L2 |
| 6 | [09](09_persistent_matmul/) | Classic 1-tile grid vs persistent workers / work queue |
| 6–7 | [10](10_block_scaled_matmul/) | Scalar identity $(s_A s_B)AB$ → idea of in-kernel block scales |

**Rule of thumb:** finish **01 → 03** before 06–10. Tutorial **03** is the GEMM backbone for 08–10.

### Before you start (prerequisites)

- Python + NumPy comfort; basic **PyTorch** tensors on GPU
- Matrix multiply $C=AB$ and row-wise reductions (softmax / layer-norm idea)
- Optional but helpful: CUDA **grid → block → warp (32)** mental model (see whiteboard in 01 / 03 and the cheat sheet under [`asset/`](asset/))

### Mini glossary

| Term | Meaning here |
|------|----------------|
| **Program / pid** | Triton launch unit (≈ CUDA block) |
| **Mask** | Predicate so partial tiles do not OOB-load/store |
| **Fusion** | Keep intermediates in registers/SRAM; fewer HBM trips |
| **Tile / block** | Submatrix owned by one program (`BLOCK_M/N/K`) |
| **Grouped (03)** | L2-friendly **pid swizzle inside one GEMM** |
| **Group GEMM (08)** | **Many** GEMMs with different shapes in one launch story |
| **Persistent (09)** | Few long-lived programs that claim many tiles |

---

## Tutorial map

| Folder | Topic | Official doc |
|--------|--------|--------------|
| [01_vector_addition](01_vector_addition/) | Vector addition | [01-vector-add](https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html) |
| [02_fused_softmax](02_fused_softmax/) | Fused softmax | [02-fused-softmax](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html) |
| [03_matrix_multiplication](03_matrix_multiplication/) | Matrix multiply (GEMM) | [03-matrix-multiplication](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html) |
| [04_low_memory_dropout](04_low_memory_dropout/) | Low-memory dropout | [04-low-memory-dropout](https://triton-lang.org/main/getting-started/tutorials/04-low-memory-dropout.html) |
| [05_layer_normalization](05_layer_normalization/) | Layer normalization | [05-layer-norm](https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html) |
| [06_fused_attention](06_fused_attention/) | Fused attention | [06-fused-attention](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html) |
| [07_libdevice](07_libdevice/) | Libdevice math | [07-libdevice](https://triton-lang.org/main/getting-started/tutorials/07-libdevice.html) |
| [08_group_gemm](08_group_gemm/) | Group GEMM | [08-group-gemm](https://triton-lang.org/main/getting-started/tutorials/08-group-gemm.html) |
| [09_persistent_matmul](09_persistent_matmul/) | Persistent matmul | [09-persistent-matmul](https://triton-lang.org/main/getting-started/tutorials/09-persistent-matmul.html) |
| [10_block_scaled_matmul](10_block_scaled_matmul/) | Block-scaled matmul | [10-block-scaled-matmul](https://triton-lang.org/main/getting-started/tutorials/10-block-scaled-matmul.html) |

### Shared primer

GPU fundamentals cheat sheet (CPU vs GPU, warps, coalescing, fusion): see [`asset/`](asset/) and the copy under [`01_vector_addition/docs/images/`](01_vector_addition/docs/images/).

---

## Setup and run

```bash
pip install -r requirements.txt
cd 01_vector_addition
jupyter notebook 01_vector_addition.ipynb
# or
python tutorial.py
```

Each folder’s `tutorial.py` / notebook includes **CPU pseudo tests** (`test_pseudo_*`) and GPU checks that skip without CUDA.

```bash
# from a tutorial folder, e.g.
cd 03_matrix_multiplication && python -c "import tutorial; tutorial.test_pseudo_matches_torch()"
```

## Upstream

Official materials: [tutorials index](https://triton-lang.org/main/getting-started/tutorials/index.html).
