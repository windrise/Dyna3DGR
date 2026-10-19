"""Differentiable CUDA intensity accumulation for the released Dyna3DGR model."""
from functools import lru_cache
from pathlib import Path

import torch
from torch.autograd import Function
from torch.utils.cpp_extension import load


@lru_cache(maxsize=1)
def _load_extension():
    """Build on first use, from this checkout rather than an author's machine."""
    source = Path(__file__).resolve().with_name("discretize_grid.cu")
    if not source.is_file():
        raise FileNotFoundError(f"Missing bundled CUDA source: {source}")
    if not torch.cuda.is_available():
        raise RuntimeError("Dyna3DGR intensity computation requires a CUDA-enabled PyTorch and GPU")
    return load(
        name="dyna3dgr_compute_intensity_cuda",
        sources=[str(source)],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3"],
        # Let the installed PyTorch choose its supported C++ language standard.
        verbose=False,
    )


class IntensityComputation(Function):
    @staticmethod
    def forward(ctx, gaussian_centers, grid_points, intensities, inv_covariances, scalings, intensity_grid):
        ctx.save_for_backward(gaussian_centers, grid_points, intensities, inv_covariances, scalings, intensity_grid)
        return _load_extension().compute_intensity(
            gaussian_centers, grid_points, intensities, inv_covariances, scalings, intensity_grid
        )

    @staticmethod
    def backward(ctx, grad_output):
        gaussian_centers, grid_points, intensities, inv_covariances, scalings, intensity_grid = ctx.saved_tensors
        gradients = _load_extension().compute_intensity_backward(
            # Reductions such as output.sum() produce stride-zero gradients.
            grad_output.contiguous(),
            gaussian_centers, grid_points, intensities, inv_covariances, scalings, intensity_grid
        )
        grad_gaussian_centers, grad_intensities, grad_inv_covariances, grad_intensity_grid = gradients
        return grad_gaussian_centers, None, grad_intensities, grad_inv_covariances, None, grad_intensity_grid


def compute_intensity(gaussian_centers, grid_points, intensities, inv_covariances, scalings, intensity_grid):
    """Return a new grid containing input_grid + the truncated Gaussian sum.

    Inputs must be contiguous float32 CUDA tensors on one device. Centers and
    scales are (N, 3), intensities (N,) or (N, 1), inverse covariances (N, 3, 3),
    and the grid is (1, D, H, W, 1). Grid points contain D*H*W coordinates, with
    a final dimension of 3, in D/H/W order. Grid-point and support-scale gradients
    are intentionally not defined, matching the recovered operator. Gradients
    through covariance construction still propagate to model scale/rotation.
    """
    return IntensityComputation.apply(gaussian_centers, grid_points, intensities, inv_covariances, scalings, intensity_grid)
