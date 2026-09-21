"""Create a transferable build context, without Docker, models or host venvs."""

import argparse
import hashlib
import io
import tarfile
from pathlib import Path

EXPECTED = "48832f11f26134a2b0cc0234bfa1f7872c901ccf7b0835715d100f38c91b3e49"
WHEEL_NAME = "sglang_kernel-0.4.1-cp310-abi3-linux_x86_64.whl"


def main():
    here = Path(__file__).resolve().parent
    repo = here.parents[3]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=repo / "output_results/vsr/vsr-docker-context.tar.gz",
    )
    parser.add_argument(
        "--wheel",
        type=Path,
        default=repo / "output_results/vsr/kernel210-wheel" / WHEEL_NAME,
    )
    args = parser.parse_args()
    if hashlib.sha256(args.wheel.read_bytes()).hexdigest() != EXPECTED:
        raise RuntimeError("Kernel wheel does not match the validated build")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(f"Refusing to replace {args.output}; choose --output")
    relative = here.relative_to(repo)
    temporary = args.output.with_name(args.output.name + ".partial")
    with tarfile.open(temporary, "w:gz") as archive:
        roots = [repo / "python", here]
        for root in roots:
            for path in sorted(root.rglob("*")):
                if any(
                    p
                    in {
                        "__pycache__",
                        ".git",
                        "vendor",
                        ".env",
                        ".pytest_cache",
                        ".clang-format",
                    }
                    or p.endswith(".egg-info")
                    for p in path.relative_to(root).parts
                ):
                    continue
                if path.is_symlink():
                    raise RuntimeError(f"Unexpected source symlink: {path}")
                if path.is_file() and path.suffix != ".pyc":
                    archive.add(
                        path, arcname=str(path.relative_to(repo)), recursive=False
                    )
        for name in ["serve_vsr.py", "test_server_api.py"]:
            path = here.parent / "scripts" / name
            archive.add(path, arcname=str(path.relative_to(repo)))
        archive.add(args.wheel, arcname=str(relative / "vendor" / WHEEL_NAME))
        archive.add(
            repo / "output_results/vsr/kernel210_wheel_provenance.json",
            arcname=str(relative / "vendor/provenance.json"),
        )
        data = f"{EXPECTED}  {WHEEL_NAME}\n".encode()
        info = tarfile.TarInfo(str(relative / "vendor/SHA256SUMS"))
        info.size = len(data)
        archive.addfile(info, io.BytesIO(data))
    temporary.replace(args.output)
    print(f"{args.output} ({args.output.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
