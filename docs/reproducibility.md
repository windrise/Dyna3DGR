# Release dependency repair

This repair makes the released source tree self-contained at the source level
when cloned with its pinned submodules. It does not establish equivalence to the
original Dyna3DGR experiments or promise compatibility with every version allowed
by the broad Python requirements.

## Source provenance and licenses

- `utils/discretize_grid.cu` was recovered from the related, same-owner
  [3DGR-CAR source at f482e19f60f54ea3079ff505d1d819427fbeb545](https://github.com/windrise/3DGR-CAR/blob/f482e19f60f54ea3079ff505d1d819427fbeb545/3dgs-car/utils/discretize_grid.cu).
  The original source blob is `d6f32a4f2a251b88c8831ed27b6d6c3ef7856949`.
  Its exported functions and argument order match Dyna3DGR's released wrapper;
  the related project's autograd wrapper has the same forward/backward contract.
  The missing original Dyna3DGR kernel is unavailable, so byte identity and
  original-experiment equivalence cannot be established. The upstream
  **MIT / Copyright (c) 2024 USTC BME Miracle Lab** notice is retained in
  [`licenses/3DGR-CAR-MIT.txt`](../licenses/3DGR-CAR-MIT.txt).
- The bundled rasterizer matches the tracked files of
  [ashawkey/diff-gaussian-rasterization@d986da0d4cf2dfeb43b9a379b6e9fa0a7f3f7eea](https://github.com/ashawkey/diff-gaussian-rasterization/tree/d986da0d4cf2dfeb43b9a379b6e9fa0a7f3f7eea).
  Its missing GLM gitlink is restored at the exact upstream pin,
  [g-truc/glm@5c46b9c07008ae65cb81ab79cd677ecc1934b903](https://github.com/g-truc/glm/tree/5c46b9c07008ae65cb81ab79cd677ecc1934b903).
  GLM offers an MIT license option in its included `copying.txt`.
- `submodules/simple-knn` points to official
  [Inria simple-knn@86710c2d4b46680c02301765dd79e465819c8f19](https://gitlab.inria.fr/bkerbl/simple-knn/-/tree/86710c2d4b46680c02301765dd79e465819c8f19).
  [SC-GS](https://github.com/CVMI-Lab/SC-GS/tree/3a9d2ad4e4fc058b0763d446ae9e6b1be120b872)
  used `44f764299fa305faf6ec5ebd99939e0508331503`; the selected revision differs
  only by adding its full license and a `<cfloat>` include, leaving the `distCUDA2`
  implementation/interface unchanged.
- The rasterizer and simple-knn retain their **non-commercial research/evaluation**
  license terms in their own `LICENSE.md` files. The repository-root MIT license
  does not replace those component licenses.

## Kernel changes relative to the recovered source

The Gaussian formula, two-sigma support, integer support-bound truncation,
voxel traversal, and center/intensity/inverse-covariance derivative equations are
preserved. Implementation hardening is explicitly limited to:

1. Load the source relative to `Compute_intensity.py`, lazily on first use. Let
   the installed PyTorch choose the C++ standard rather than forcing C++14.
2. Replace the atomic work queue, per-thread shared arrays and divergent block
   barriers with independent grid-stride Gaussian work and local arrays. This
   avoids a barrier whose participants differed within a block, and removes
   unused queue allocations. Accumulation still uses floating-point atomics;
   bitwise equality and deterministic accumulation order are not promised.
3. Pass sizes as launch arguments rather than mutable global constant memory;
   guard the input device and use PyTorch's current CUDA stream. Check launch
   errors, tensor shapes/dtypes/devices/contiguity and 32-bit indexing limits.
4. Handle zero Gaussians without a zero-block launch, and make backward's
   incoming gradient contiguous (including `output.sum().backward()`).
5. Clone the input grid before accumulation. The mathematical output remains
   `input_grid + Gaussian sum`, with an identity gradient to the input grid,
   while avoiding an unreported mutation of an autograd input.

Inputs are contiguous float32 CUDA tensors. The operator supports a single
`(1, D, H, W, 1)` grid. Coordinates are in D/H/W order. Centers and support scales
are normalized. The recovered support bounds multiply by the axis dimension,
not dimension minus one, even when points come from an endpoint-inclusive
`linspace`. Support is axis-aligned and does not account for covariance rotation.
These inherited approximation choices are preserved, not redesigned.

The backward supplies first-order gradients for centers, intensities, inverse
covariances and the initial grid. It does not differentiate discrete support
bounds or grid coordinates. The direct support-scale gradient remains `None`;
model scales and rotations can still receive gradients via inverse covariance.
Higher-order derivatives are not a validated contract.

The two legacy `ControlNodeWarp` node initializers now call the existing
`StandardGaussianModel` constructor and `create_from_points_cloud` method.
The normal `_ca` initialization path and scientific training math are unchanged.
Other old visualization-only APIs are outside this focused repair.

## Validation

Run from the repository root:

```bash
python scripts/check_source_dependencies.py
python -m unittest discover -s tests -v
```

The first command verifies source presence and exact dependency revisions; it
does not build CUDA. CPU tests cover lazy/cached source loading, actionable
missing-source/GPU errors, wrapper argument/gradient order, the corrected legacy
initializer calls, required Python imports, and kernel safety/provenance guards.
These are focused contract checks, not a substitute for executing the extension.

On a machine with CUDA-enabled PyTorch, a CUDA toolkit and an NVIDIA GPU, the
same suite additionally compiles the extension and checks:

- Non-cubic grid, overlapping Gaussians, nonzero initial grid and no input mutation
- Forward and weighted backward against a PyTorch reference with identical support
- N = 0, 1, 255, 256 and 257, including partial thread blocks and expanded gradients
- Rejection of invalid CPU/dtype/shape/contiguity inputs
- Non-default stream, plus a non-current device when multiple GPUs are available

For extra launch/memory diagnostics on that machine:

```bash
compute-sanitizer --tool memcheck python -m unittest discover -s tests -p 'test_intensity_cuda.py' -v
compute-sanitizer --tool synccheck python -m unittest discover -s tests -p 'test_intensity_cuda.py' -v
```

### Repair-time results and remaining gates

The repair workspace had no PyTorch, `nvcc`, NVIDIA driver/GPU tools or GPU.
The CPU contract tests and pinned-source checks were run; the four CUDA test
methods were explicitly skipped. No CUDA compilation, GPU numerical comparison,
sanitizer run, extension installation or full training was performed there.

Before treating this repair as a validated runnable research release:

1. Record a working Python/PyTorch/CUDA/PyTorch3D/compiler version set.
2. Build both local extensions and run the CUDA test suite without skips.
3. Run sanitizer checks and a bounded toy-data training smoke test on the target GPU.
4. Compare a known checkpoint/experiment against its expected metrics before
   claiming reproduction of the paper. That comparison requires a trusted
   baseline; recovering a related kernel is not itself such evidence.
