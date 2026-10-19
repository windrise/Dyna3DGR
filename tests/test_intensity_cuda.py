"""Small CUDA/reference checks, automatically skipped without a GPU/toolkit.

No training or patient data is used. The reference preserves recovered integer
support bounds, including dimension (not dimension - 1) and two-sigma truncation.
"""
import unittest

try:
    import torch
    from torch.utils.cpp_extension import CUDA_HOME
except ImportError:
    torch = None
    CUDA_HOME = None

CUDA_READY = torch is not None and torch.cuda.is_available() and CUDA_HOME is not None


def reference(centers, points, intensities, inverse, scales, seed):
    dims = seed.shape[1:4]
    coords = points.reshape(*dims, 3)
    axes = torch.meshgrid(*(torch.arange(d, device=seed.device) for d in dims), indexing="ij")
    output = seed.reshape(*dims).clone()
    size = centers.new_tensor(dims)
    for i in range(centers.shape[0]):
        # Detach the discrete support: the recovered backward differentiates
        # the Gaussian value, but not jumps caused by integer bound changes.
        mean = centers[i].detach() * size
        extent = (2.0 * scales[i].detach()) * size
        lower = torch.trunc(mean - extent).to(torch.int64).clamp_min(0)
        upper = torch.minimum(torch.trunc(mean + extent).to(torch.int64), size.to(torch.int64) - 1)
        mask = torch.ones(dims, dtype=torch.bool, device=seed.device)
        for axis in range(3):
            mask = mask & (axes[axis] >= lower[axis]) & (axes[axis] <= upper[axis])
        delta = coords - centers[i]
        power = -0.5 * torch.einsum("...i,ij,...j->...", delta, inverse[i], delta)
        output = output + intensities.reshape(-1)[i] * power.exp() * mask
    return output.reshape_as(seed)


@unittest.skipUnless(CUDA_READY, "requires CUDA-enabled PyTorch, NVIDIA GPU and CUDA toolkit")
class IntensityCudaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from utils.Compute_intensity import _load_extension, compute_intensity
        cls.compute = staticmethod(compute_intensity)
        _load_extension()

    def inputs(self, n, device="cuda"):
        # Deliberately non-cubic grid; overlapping support; nonzero seed.
        generator = torch.Generator(device=device).manual_seed(41)
        centers = (0.2 + 0.6 * torch.rand((n, 3), device=device, generator=generator)).requires_grad_()
        intensities = (0.1 + torch.rand((n, 1), device=device, generator=generator)).requires_grad_()
        inverse = torch.eye(3, device=device).repeat(n, 1, 1) * 12.0
        if n:
            inverse[:, 0, 1] = 0.3  # Include nonsymmetric entries in gradient checks.
        inverse.requires_grad_()
        scales = torch.full((n, 3), 0.21, device=device)
        points = torch.stack(torch.meshgrid(*(torch.linspace(0, 1, d, device=device)
                                               for d in (5, 4, 3)), indexing="ij"), -1)
        points = points.reshape(1, 5, 4, 3, 1, 3).contiguous()
        seed = torch.full((1, 5, 4, 3, 1), 0.125, device=device, requires_grad=True)
        return centers, points, intensities, inverse, scales, seed

    def test_reference_forward_backward_and_partial_blocks(self):
        for n in (1, 255, 256, 257):
            with self.subTest(n=n):
                args = self.inputs(n)
                centers, points, intensities, inverse, scales, seed = args
                saved_seed = seed.detach().clone()
                actual = self.compute(*args)
                expected = reference(*args)
                torch.testing.assert_close(actual, expected, rtol=3e-5, atol=3e-5)
                self.assertNotEqual(actual.data_ptr(), seed.data_ptr())
                torch.testing.assert_close(seed, saved_seed)
                weights = torch.linspace(-0.7, 1.1, seed.numel(), device=seed.device).reshape_as(seed)
                differentiable = (centers, intensities, inverse, seed)
                actual_grad = torch.autograd.grad((actual * weights).sum(), differentiable)
                expected_grad = torch.autograd.grad((expected * weights).sum(), differentiable)
                for actual_component, expected_component in zip(actual_grad, expected_grad):
                    torch.testing.assert_close(actual_component, expected_component, rtol=1e-4, atol=1e-4)

    def test_expanded_reduction_gradient_and_empty_input(self):
        for n in (0, 1):
            with self.subTest(n=n):
                args = self.inputs(n)
                output = self.compute(*args)
                output.sum().backward()
                torch.testing.assert_close(args[-1].grad, torch.ones_like(args[-1]))
                for index in (0, 2, 3):
                    self.assertIsNotNone(args[index].grad)
                    self.assertTrue(torch.isfinite(args[index].grad).all().item())
                if n == 0:
                    torch.testing.assert_close(output, args[-1])

    def test_rejects_unsafe_inputs(self):
        args = list(self.inputs(2))
        invalid = [
            (0, args[0].double(), "float32"),
            (0, args[0].cpu(), "CUDA tensor"),
            (0, torch.zeros(2, 6, device="cuda")[:, ::2], "contiguous"),
            (3, args[3][:, :2, :].contiguous(), "inv_covariances"),
            (5, torch.zeros(2, 5, 4, 3, 1, device="cuda"), "intensity_grid"),
        ]
        for index, value, message in invalid:
            with self.subTest(message=message):
                altered = list(args)
                altered[index] = value
                with self.assertRaisesRegex(RuntimeError, message):
                    self.compute(*altered)

    def test_current_stream_and_device(self):
        original_device = torch.cuda.current_device()
        devices = [original_device]
        if torch.cuda.device_count() > 1:
            devices.append((original_device + 1) % torch.cuda.device_count())
        for device_index in devices:
            with self.subTest(device=device_index):
                device = torch.device("cuda", device_index)
                stream = torch.cuda.Stream(device=device)
                with torch.cuda.stream(stream):
                    args = self.inputs(3, device=device)
                    # With >1 GPU, exercise CUDAGuard when current device differs.
                    torch.cuda.set_device(original_device)
                    actual = self.compute(*args)
                    actual.sum().backward()
                    expected = reference(*args)
                stream.synchronize()
                torch.testing.assert_close(actual, expected, rtol=3e-5, atol=3e-5)
        torch.cuda.set_device(original_device)


if __name__ == "__main__":
    unittest.main()
