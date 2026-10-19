"""Verify pinned source dependencies without importing PyTorch or invoking CUDA."""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
DEPENDENCIES = {
    "submodules/simple-knn": (
        "86710c2d4b46680c02301765dd79e465819c8f19", ("setup.py", "simple_knn.cu", "LICENSE.md")
    ),
    "submodules/diff-gaussian-rasterization/third_party/glm": (
        "5c46b9c07008ae65cb81ab79cd677ecc1934b903", ("glm/glm.hpp", "copying.txt")
    ),
}


def main():
    errors = []
    for relative, (expected, files) in DEPENDENCIES.items():
        path = ROOT / relative
        missing = [name for name in files if not (path / name).is_file()]
        if missing:
            errors.append(f"{relative}: missing {', '.join(missing)}")
            continue
        result = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"],
                                capture_output=True, text=True)
        actual = result.stdout.strip()
        if result.returncode or actual != expected:
            errors.append(f"{relative}: expected {expected}, got {actual or 'no Git revision'}")
        else:
            print(f"OK {relative} @ {actual}")
    if errors:
        for error in errors:
            print(f"ERROR {error}", file=sys.stderr)
        print("Run: git submodule update --init --recursive", file=sys.stderr)
        return 1
    print("Pinned source dependencies are present. CUDA compilation has not been checked.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
