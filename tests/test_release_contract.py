"""CPU-only release checks; do not require PyTorch or build CUDA at import time."""
import ast
from contextlib import contextmanager
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]


@contextmanager
def stubbed_wrapper():
    loader = mock.Mock(return_value=mock.Mock())
    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(is_available=lambda: True)
    autograd = types.ModuleType("torch.autograd")
    autograd.Function = type("Function", (), {})
    cpp_extension = types.ModuleType("torch.utils.cpp_extension")
    cpp_extension.load = loader
    modules = {"torch": torch, "torch.autograd": autograd,
               "torch.utils": types.ModuleType("torch.utils"),
               "torch.utils.cpp_extension": cpp_extension}
    with mock.patch.dict(sys.modules, modules):
        spec = importlib.util.spec_from_file_location("intensity_under_test", ROOT / "utils/Compute_intensity.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        yield module, loader


class LoaderContractTests(unittest.TestCase):
    def test_import_is_lazy_and_load_is_cached(self):
        with stubbed_wrapper() as (module, loader):
            loader.assert_not_called()
            first = module._load_extension()
            self.assertIs(first, module._load_extension())
            loader.assert_called_once()
            kwargs = loader.call_args.kwargs
            self.assertEqual(kwargs["sources"], [str(ROOT / "utils/discretize_grid.cu")])
            self.assertTrue(Path(kwargs["sources"][0]).is_file())
            self.assertFalse(any(flag.startswith("-std=") for flag in kwargs["extra_cflags"]))

    def test_load_does_not_depend_on_working_directory(self):
        with stubbed_wrapper() as (module, loader), tempfile.TemporaryDirectory() as temp:
            previous = Path.cwd()
            try:
                os.chdir(temp)
                module._load_extension()
            finally:
                os.chdir(previous)
            self.assertEqual(loader.call_args.kwargs["sources"],
                             [str(ROOT / "utils/discretize_grid.cu")])

    def test_missing_source_is_actionable(self):
        with stubbed_wrapper() as (module, loader), tempfile.TemporaryDirectory() as temp:
            module.__file__ = str(Path(temp) / "Compute_intensity.py")
            with self.assertRaisesRegex(FileNotFoundError, "Missing bundled CUDA source"):
                module._load_extension()
            loader.assert_not_called()

    def test_missing_cuda_fails_only_on_first_use(self):
        with stubbed_wrapper() as (module, loader):
            module.torch.cuda.is_available = lambda: False
            with self.assertRaisesRegex(RuntimeError, "CUDA-enabled PyTorch and GPU"):
                module._load_extension()
            loader.assert_not_called()

    def test_forward_and_backward_argument_contract(self):
        with stubbed_wrapper() as (module, loader):
            tensors = tuple(object() for _ in range(6))
            ctx = mock.Mock()
            ctx.saved_tensors = tensors
            extension = loader.return_value
            self.assertIs(module.IntensityComputation.forward(ctx, *tensors),
                          extension.compute_intensity.return_value)
            ctx.save_for_backward.assert_called_once_with(*tensors)
            extension.compute_intensity.assert_called_once_with(*tensors)
            gradient = mock.Mock()
            extension.compute_intensity_backward.return_value = ("center", "intensity", "inverse", "grid")
            result = module.IntensityComputation.backward(ctx, gradient)
            gradient.contiguous.assert_called_once_with()
            extension.compute_intensity_backward.assert_called_once_with(
                gradient.contiguous.return_value, *tensors)
            self.assertEqual(result, ("center", None, "intensity", "inverse", None, "grid"))


class RepositoryContractTests(unittest.TestCase):
    def test_legacy_node_initializers_use_released_model_api(self):
        tree = ast.parse((ROOT / "utils/time_utils.py").read_text())
        found = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name in ("as_gaussians", "init_gaussians"):
                found.add(node.name)
                calls = [child for child in ast.walk(node) if isinstance(child, ast.Call)]
                constructors = [call for call in calls if isinstance(call.func, ast.Name)
                                and call.func.id == "StandardGaussianModel"]
                self.assertEqual(len(constructors), 1)
                self.assertNotIn("sh_degree", [kw.arg for kw in constructors[0].keywords])
                methods = [call.func.attr for call in calls if isinstance(call.func, ast.Attribute)]
                self.assertIn("create_from_points_cloud", methods)
                self.assertNotIn("create_from_pcd", methods)
        self.assertEqual(found, {"as_gaussians", "init_gaussians"})

    def test_cuda_safety_guards_and_provenance_are_bundled(self):
        source = (ROOT / "utils/discretize_grid.cu").read_text()
        self.assertNotIn("__syncthreads", source)
        self.assertNotIn("__constant__", source)
        self.assertNotIn("work_counter", source)
        self.assertEqual(source.count("C10_CUDA_KERNEL_LAUNCH_CHECK();"), 2)
        self.assertEqual(source.count("c10::cuda::getCurrentCUDAStream()"), 2)
        self.assertIn("intensity_grid.clone()", source)
        self.assertIn("f482e19f60f54ea3079ff505d1d819427fbeb545", source)
        self.assertIn("Copyright (c) 2024 USTC BME Miracle Lab",
                      (ROOT / "licenses/3DGR-CAR-MIT.txt").read_text())

    def test_unconditional_training_and_build_dependencies_listed(self):
        requirements = (ROOT / "requirements.txt").read_text().splitlines()
        self.assertIn("SimpleITK", requirements)
        self.assertIn("ninja", requirements)


if __name__ == "__main__":
    unittest.main()
