// Recovered from windrise/3DGR-CAR, commit f482e19f60f54ea3079ff505d1d819427fbeb545.
// Copyright (c) 2024 USTC BME Miracle Lab. MIT license: ../licenses/3DGR-CAR-MIT.txt.
// See docs/reproducibility.md for provenance, hardening changes and validation limits.
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <climits>

namespace {
void check_tensor(const torch::Tensor& tensor, const torch::Tensor& centers,
                  const char* name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.scalar_type() == torch::kFloat32, name, " must be float32");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(tensor.device() == centers.device(),
                name, " must be on the same CUDA device as gaussian_centers");
}

void check_inputs(const torch::Tensor& centers, const torch::Tensor& grid_points,
                  const torch::Tensor& intensities, const torch::Tensor& inv_covariances,
                  const torch::Tensor& scalings, const torch::Tensor& intensity_grid) {
    check_tensor(centers, centers, "gaussian_centers");
    check_tensor(grid_points, centers, "grid_points");
    check_tensor(intensities, centers, "intensities");
    check_tensor(inv_covariances, centers, "inv_covariances");
    check_tensor(scalings, centers, "scalings");
    check_tensor(intensity_grid, centers, "intensity_grid");
    TORCH_CHECK(centers.dim() == 2 && centers.size(1) == 3,
                "gaussian_centers must have shape (N, 3)");
    const auto n = centers.size(0);
    TORCH_CHECK(n <= INT_MAX / 9, "too many Gaussians for 32-bit kernel indexing");
    TORCH_CHECK(scalings.sizes() == centers.sizes(), "scalings must have shape (N, 3)");
    TORCH_CHECK((intensities.dim() == 1 ||
                (intensities.dim() == 2 && intensities.size(1) == 1)) &&
                intensities.size(0) == n, "intensities must have shape (N,) or (N, 1)");
    TORCH_CHECK(inv_covariances.dim() == 3 && inv_covariances.size(0) == n &&
                inv_covariances.size(1) == 3 && inv_covariances.size(2) == 3,
                "inv_covariances must have shape (N, 3, 3)");
    TORCH_CHECK(intensity_grid.dim() == 5 && intensity_grid.size(0) == 1 &&
                intensity_grid.size(4) == 1 && intensity_grid.size(1) > 0 &&
                intensity_grid.size(2) > 0 && intensity_grid.size(3) > 0,
                "intensity_grid must have shape (1, D, H, W, 1) with nonempty spatial axes");
    TORCH_CHECK(intensity_grid.numel() <= INT_MAX / 3,
                "grid is too large for 32-bit kernel indexing");
    TORCH_CHECK(grid_points.dim() >= 2 && grid_points.size(-1) == 3 &&
                grid_points.numel() == 3 * intensity_grid.numel(),
                "grid_points must contain one 3D coordinate per voxel in D/H/W order");
}
}  // namespace

// CUDA kernel function to compute intensity
__global__ void compute_intensity_forward_kernel(
    const float* __restrict__ grid_points,
    const float* __restrict__ inv_covariances,
    const float* __restrict__ gaussian_centers,
    const float* __restrict__ intensities,
    const float* __restrict__ scalings,
    float* __restrict__ intensity_grid,
    int const_num_gaussians,
    int const_grid_size_z,
    int const_grid_size_x,
    int const_grid_size_y
    ) {

    float inv_cov[9];
    float centers[3];
    float scales[3];

    // Each thread processes independent Gaussians with a grid-stride loop.
    for (int gaussian_idx = blockIdx.x * blockDim.x + threadIdx.x;
         gaussian_idx < const_num_gaussians;
         gaussian_idx += blockDim.x * gridDim.x) {

        for (int i = 0; i < 9; ++i) {
            inv_cov[i] = inv_covariances[gaussian_idx * 9 + i];
        }

        centers[0] = gaussian_centers[gaussian_idx * 3];
        centers[1] = gaussian_centers[gaussian_idx * 3 + 1];
        centers[2] = gaussian_centers[gaussian_idx * 3 + 2];

        scales[0] = scalings[gaussian_idx * 3];
        scales[1] = scalings[gaussian_idx * 3 + 1];
        scales[2] = scalings[gaussian_idx * 3 + 2];

        // Compute mean coordinates
        int center_idx = gaussian_idx * 3;
        float mean_z = centers[0] * (float)const_grid_size_z;
        float mean_x = centers[1] * (float)const_grid_size_x;
        float mean_y = centers[2] * (float)const_grid_size_y;

        //
        float coeff = 2.0;

        float norm_expand_z = coeff * scales[0];
        float norm_expand_x = coeff * scales[1];
        float norm_expand_y = coeff * scales[2];

        float expand_z = norm_expand_z * const_grid_size_z;
        float expand_x = norm_expand_x * const_grid_size_x;
        float expand_y = norm_expand_y * const_grid_size_y;


        int z_start = max(0, (int)(mean_z - expand_z));
        int x_start = max(0, (int)(mean_x - expand_x));
        int y_start = max(0, (int)(mean_y - expand_y));

        int z_end = min(const_grid_size_z-1, (int)(mean_z + expand_z));
        int x_end = min(const_grid_size_x-1, (int)(mean_x + expand_x));
        int y_end = min(const_grid_size_y-1, (int)(mean_y + expand_y));

        int grid_xy = const_grid_size_x * const_grid_size_y;
        float intensity = intensities[gaussian_idx];

        for (int z = z_start; z <= z_end; ++z){
            for (int x = x_start; x <= x_end; ++x) {
                for (int y = y_start; y <= y_end; ++y){
                    // Compute distance from the grid point to the Gaussian center
                    int grid_idx = 3 * (z * grid_xy + x * const_grid_size_y + y);
                    float dz = grid_points[grid_idx] - gaussian_centers[center_idx];
                    float dx = grid_points[grid_idx + 1] - gaussian_centers[center_idx + 1];
                    float dy = grid_points[grid_idx + 2] - gaussian_centers[center_idx + 2];

                    // Compute the expoential term
                    float power = -0.5f * (
                        dz * (inv_cov[0] * dz + inv_cov[1] * dx + inv_cov[2] * dy) +
                        dx * (inv_cov[3] * dz + inv_cov[4] * dx + inv_cov[5] * dy) +
                        dy * (inv_cov[6] * dz + inv_cov[7] * dx + inv_cov[8] * dy)
                    );

                    // Compute the density value
                    float intensity_value = intensity * __expf(power);

                    // Atomic add
                    atomicAdd(&intensity_grid[z * grid_xy + x * const_grid_size_y + y], intensity_value);

                }
            }
        }
    }
}


torch::Tensor compute_intensity(
    torch::Tensor gaussian_centers,
    torch::Tensor grid_points,
    torch::Tensor intensities,
    torch::Tensor inv_covariances,
    torch::Tensor scalings,
    torch::Tensor intensity_grid) {
    check_inputs(gaussian_centers, grid_points, intensities,
                 inv_covariances, scalings, intensity_grid);
    const c10::cuda::CUDAGuard device_guard(gaussian_centers.device());
    // Preserve input_grid + Gaussian sum without mutating an autograd input.
    auto output = intensity_grid.clone();
    const int num_gaussians = gaussian_centers.size(0);
    if (num_gaussians == 0) return output;
    const int threads = 256;
    const int blocks = std::min((num_gaussians + threads - 1) / threads, 65535);
    compute_intensity_forward_kernel<<<blocks, threads, 0,
        c10::cuda::getCurrentCUDAStream()>>>(
        grid_points.data_ptr<float>(), inv_covariances.data_ptr<float>(),
        gaussian_centers.data_ptr<float>(), intensities.data_ptr<float>(),
        scalings.data_ptr<float>(), output.data_ptr<float>(), num_gaussians,
        intensity_grid.size(1), intensity_grid.size(2), intensity_grid.size(3));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}


__forceinline__ __device__ void compute_grad_gaussian_center(
    float dz, float dx, float dy,
    const float* __restrict__ inv_cov,
    float intensity_value,
    float grad_output_val,
    float* __restrict__ grad_gaussian_center) {

    float common_term = 0.5f * intensity_value * grad_output_val;
    atomicAdd(&grad_gaussian_center[0], common_term * (2 * inv_cov[0] * dz + (inv_cov[1] + inv_cov[3]) * dx + (inv_cov[2] + inv_cov[6]) * dy));
    atomicAdd(&grad_gaussian_center[1], common_term * ((inv_cov[3] + inv_cov[1]) * dz + 2 * inv_cov[4] * dx + (inv_cov[5] + inv_cov[7]) * dy));
    atomicAdd(&grad_gaussian_center[2], common_term * ((inv_cov[6] + inv_cov[2]) * dz + (inv_cov[7] + inv_cov[5]) * dx + 2 * inv_cov[8] * dy));
}

__forceinline__ __device__ void compute_grad_inv_covariance(
    float dz, float dx, float dy,
    float intensity_value,
    float grad_output_val,
    float* __restrict__ grad_inv_covariance) {

    float grad_common = -0.5f * intensity_value * grad_output_val;
    atomicAdd(&grad_inv_covariance[0], grad_common * dz * dz);
    atomicAdd(&grad_inv_covariance[1], grad_common * dz * dx);
    atomicAdd(&grad_inv_covariance[2], grad_common * dz * dy);
    atomicAdd(&grad_inv_covariance[3], grad_common * dx * dz);
    atomicAdd(&grad_inv_covariance[4], grad_common * dx * dx);
    atomicAdd(&grad_inv_covariance[5], grad_common * dx * dy);
    atomicAdd(&grad_inv_covariance[6], grad_common * dy * dz);
    atomicAdd(&grad_inv_covariance[7], grad_common * dy * dx);
    atomicAdd(&grad_inv_covariance[8], grad_common * dy * dy);
}

__global__ void compute_intensity_backward_kernel(
    const float* __restrict__ grad_output,
    const float* __restrict__ grid_points,
    const float* __restrict__ inv_covariances,
    const float* __restrict__ gaussian_centers,
    const float* __restrict__ intensities,
    const float* __restrict__ scalings,

    float* __restrict__ grad_gaussian_centers,
    float* __restrict__ grad_intensities,
    float* __restrict__ grad_inv_covariances, // 包含了scaling和rotation
    int const_num_gaussians,
    int const_grid_size_z,
    int const_grid_size_x,
    int const_grid_size_y
){

    float inv_cov[9];
    float centers[3];
    float scales[3];

    for (int gaussian_idx = blockIdx.x * blockDim.x + threadIdx.x;
         gaussian_idx < const_num_gaussians;
         gaussian_idx += blockDim.x * gridDim.x) {

        for (int i = 0; i < 9; ++i) {
            inv_cov[i] = inv_covariances[gaussian_idx * 9 + i];
        }

        centers[0] = gaussian_centers[gaussian_idx * 3];
        centers[1] = gaussian_centers[gaussian_idx * 3 + 1];
        centers[2] = gaussian_centers[gaussian_idx * 3 + 2];

        scales[0] = scalings[gaussian_idx * 3];
        scales[1] = scalings[gaussian_idx * 3 + 1];
        scales[2] = scalings[gaussian_idx * 3 + 2];

        // Compute mean coordinates
        int center_idx = gaussian_idx * 3;
        float mean_z = centers[0] * (float)const_grid_size_z;
        float mean_x = centers[1] * (float)const_grid_size_x;
        float mean_y = centers[2] * (float)const_grid_size_y;

        //
        float coeff = 2.0;
        float norm_expand_z = coeff * scales[0];
        float norm_expand_x = coeff * scales[1];
        float norm_expand_y = coeff * scales[2];

        float expand_z = norm_expand_z * const_grid_size_z;
        float expand_x = norm_expand_x * const_grid_size_x;
        float expand_y = norm_expand_y * const_grid_size_y;

        // float expand_z = 5.0;
        // float expand_x = 5.0;
        // float expand_y = 5.0;

        int z_start = max(0, (int)(mean_z - expand_z));
        int x_start = max(0, (int)(mean_x - expand_x));
        int y_start = max(0, (int)(mean_y - expand_y));

        int z_end = min(const_grid_size_z-1, (int)(mean_z + expand_z));
        int x_end = min(const_grid_size_x-1, (int)(mean_x + expand_x));
        int y_end = min(const_grid_size_y-1, (int)(mean_y + expand_y));

        int grid_xy = const_grid_size_x * const_grid_size_y;
        float intensity = intensities[gaussian_idx];

        for (int z = z_start; z <= z_end; ++z){
            for (int x = x_start; x <= x_end; ++x) {
                for (int y = y_start; y <= y_end; ++y){
                    // Compute distance from the grid point to the Gaussian center
                    int grid_idx = 3 * (z * grid_xy + x * const_grid_size_y + y);
                    float dz = grid_points[grid_idx] - gaussian_centers[center_idx];
                    float dx = grid_points[grid_idx + 1] - gaussian_centers[center_idx + 1];
                    float dy = grid_points[grid_idx + 2] - gaussian_centers[center_idx + 2];

                    // Compute the exponential term
                    float power = -0.5f * (
                        dz * (inv_cov[0] * dz + inv_cov[1] * dx + inv_cov[2] * dy) +
                        dx * (inv_cov[3] * dz + inv_cov[4] * dx + inv_cov[5] * dy) +
                        dy * (inv_cov[6] * dz + inv_cov[7] * dx + inv_cov[8] * dy)
                    );

                    float grad_output_val = grad_output[z * grid_xy + x * const_grid_size_y + y];

                    // Compute the density value
                    float intensity_value = intensity * __expf(power);

                    // Compute gradient w.r.t. intensity
                    float grad_intensity = __expf(power) * grad_output_val;
                    atomicAdd(&grad_intensities[gaussian_idx], grad_intensity);

                    // Gradient w.r.t. gaussian centers
                    compute_grad_gaussian_center(
                        dz, dx, dy,
                        inv_cov,
                        intensity_value,
                        grad_output_val,
                        &grad_gaussian_centers[center_idx]
                    );

                    // Gradient w.r.t. inverse covariances
                    compute_grad_inv_covariance(
                        dz, dx, dy,
                        intensity_value,
                        grad_output_val,
                        &grad_inv_covariances[gaussian_idx * 9]
                    );

                }
            }
        }
    }
}

std::vector<torch::Tensor> compute_intensity_backward(
    torch::Tensor grad_output,
    torch::Tensor gaussian_centers,
    torch::Tensor grid_points,
    torch::Tensor intensities,
    torch::Tensor inv_covariances,
    torch::Tensor scalings,
    torch::Tensor intensity_grid) {
    check_inputs(gaussian_centers, grid_points, intensities,
                 inv_covariances, scalings, intensity_grid);
    check_tensor(grad_output, gaussian_centers, "grad_output");
    TORCH_CHECK(grad_output.sizes() == intensity_grid.sizes(),
                "grad_output must have the same shape as intensity_grid");
    const c10::cuda::CUDAGuard device_guard(gaussian_centers.device());
    auto grad_gaussian_centers = torch::zeros_like(gaussian_centers);
    auto grad_intensities = torch::zeros_like(intensities);
    auto grad_inv_covariances = torch::zeros_like(inv_covariances);
    const int num_gaussians = gaussian_centers.size(0);
    if (num_gaussians > 0) {
        const int threads = 256;
        const int blocks = std::min((num_gaussians + threads - 1) / threads, 65535);
        compute_intensity_backward_kernel<<<blocks, threads, 0,
            c10::cuda::getCurrentCUDAStream()>>>(
            grad_output.data_ptr<float>(), grid_points.data_ptr<float>(),
            inv_covariances.data_ptr<float>(), gaussian_centers.data_ptr<float>(),
            intensities.data_ptr<float>(), scalings.data_ptr<float>(),
            grad_gaussian_centers.data_ptr<float>(), grad_intensities.data_ptr<float>(),
            grad_inv_covariances.data_ptr<float>(), num_gaussians,
            intensity_grid.size(1), intensity_grid.size(2), intensity_grid.size(3));
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return {grad_gaussian_centers, grad_intensities, grad_inv_covariances, grad_output};
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("compute_intensity", &compute_intensity, "Compute Intensity (CUDA)");
    m.def("compute_intensity_backward", &compute_intensity_backward, "Compute intensity backward (CUDA)");
}
