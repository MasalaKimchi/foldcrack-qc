"""Build and exercise an isolated wheel without repository-relative imports."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import venv
import zipfile
from pathlib import Path

REQUIRED_WHEEL_RESOURCES = frozenset(
    {
        "foldcrack_qc/resources/datasets.json",
        "foldcrack_qc/resources/public_data/cosmx_gastric_v1.json",
        "foldcrack_qc/resources/public_data/cosmx_phgg_v1.json",
        "foldcrack_qc/resources/public_data/qualifai_comet_v2.json",
    }
)


def _isolated_environment() -> dict[str, str]:
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    return environment


def _venv_python(environment_root: Path) -> Path:
    if os.name == "nt":  # pragma: no cover - exercised by Windows CI
        return environment_root / "Scripts" / "python.exe"
    return environment_root / "bin" / "python"


def _venv_console_script(environment_root: Path, name: str) -> Path:
    if os.name == "nt":  # pragma: no cover - exercised by Windows CI
        return environment_root / "Scripts" / f"{name}.exe"
    return environment_root / "bin" / name


def run_wheel_smoke(repository_root: Path) -> dict[str, object]:
    """Build, install, and smoke-test one wheel in a temporary environment."""

    root = repository_root.resolve()
    if not (root / "pyproject.toml").is_file():
        raise FileNotFoundError(f"No pyproject.toml below repository root: {root}")

    environment = _isolated_environment()
    with tempfile.TemporaryDirectory(prefix="foldcrack-wheel-smoke-") as temporary:
        scratch = Path(temporary)
        build_source = scratch / "source"
        build_source.mkdir()
        for name in ("pyproject.toml", "README.md"):
            shutil.copy2(root / name, build_source / name)
        shutil.copytree(
            root / "src",
            build_source / "src",
            ignore=shutil.ignore_patterns(
                "__pycache__",
                "*.egg-info",
                "*.pyc",
            ),
        )
        wheelhouse = scratch / "wheelhouse"
        wheelhouse.mkdir()
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "wheel",
                ".",
                "--no-deps",
                "--no-build-isolation",
                "--wheel-dir",
                str(wheelhouse),
            ],
            cwd=build_source,
            env=environment,
            check=True,
            capture_output=True,
            text=True,
        )
        wheels = sorted(wheelhouse.glob("foldcrack_qc-*.whl"))
        if len(wheels) != 1:
            raise RuntimeError(f"Expected exactly one wheel, found: {wheels}")
        wheel = wheels[0]
        with zipfile.ZipFile(wheel) as archive:
            wheel_members = frozenset(archive.namelist())
        missing = sorted(REQUIRED_WHEEL_RESOURCES - wheel_members)
        if missing:
            raise RuntimeError(f"Wheel is missing runtime resources: {missing}")

        environment_root = scratch / "venv"
        venv.EnvBuilder(with_pip=True, clear=True).create(environment_root)
        python = _venv_python(environment_root)
        subprocess.run(
            [
                str(python),
                "-m",
                "pip",
                "install",
                "--no-deps",
                str(wheel),
            ],
            cwd=scratch,
            env=environment,
            check=True,
            capture_output=True,
            text=True,
        )
        console_script = _venv_console_script(environment_root, "foldcrack-qc")
        if not console_script.is_file():
            raise RuntimeError("Installed wheel did not create foldcrack-qc")

        datasets = subprocess.run(
            [str(console_script), "datasets", "--json"],
            cwd=scratch,
            env=environment,
            check=True,
            capture_output=True,
            text=True,
        )
        registry = json.loads(datasets.stdout)
        if not isinstance(registry, list) or not registry:
            raise RuntimeError("Installed-wheel dataset registry is empty or invalid")

        source_only_test = subprocess.run(
            [str(console_script), "test"],
            cwd=scratch,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )
        expected_message = "requires a source checkout containing tests/"
        if source_only_test.returncode != 2 or expected_message not in (
            source_only_test.stderr
        ):
            raise RuntimeError(
                "Installed-wheel test command did not fail with the documented "
                "source-checkout-only contract"
            )

        return {
            "wheel": wheel.name,
            "registry_records": len(registry),
            "packaged_resources": len(REQUIRED_WHEEL_RESOURCES),
            "console_entrypoint": True,
            "source_only_test_exit": source_only_test.returncode,
        }


def main() -> int:
    repository_root = Path(__file__).resolve().parents[1]
    result = run_wheel_smoke(repository_root)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
