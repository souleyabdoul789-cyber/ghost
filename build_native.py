"""Build Ghost's C engine for the current host platform and architecture."""
from __future__ import annotations

import os
import platform
import shlex
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def native_filename(system: str | None = None) -> str:
    system = (system or platform.system()).lower()
    if system == "windows":
        return "ghost.dll"
    if system == "darwin":
        return "libghost.dylib"
    return "libghost.so"


def _compiler() -> list[str]:
    configured = os.environ.get("CC")
    if configured:
        parts = shlex.split(configured)
        if parts and (shutil.which(parts[0]) or Path(parts[0]).exists()):
            return parts
        raise RuntimeError(f"Compilateur CC introuvable : {configured}")

    for candidate in ("cc", "clang", "gcc", "cl"):
        found = shutil.which(candidate)
        if found:
            return [found]
    raise RuntimeError("Ghost nécessite un compilateur C : clang, gcc, cc ou MSVC cl.")


def build_native(destination_dir: str | Path) -> Path:
    destination_dir = Path(destination_dir).resolve()
    destination_dir.mkdir(parents=True, exist_ok=True)
    system = platform.system().lower()
    compiler = _compiler()
    compiler_name = Path(compiler[0]).name.lower()
    output = destination_dir / native_filename(system)
    sources = [str(ROOT / "c" / "tensor.c"), str(ROOT / "c" / "nn.c")]
    extra = shlex.split(os.environ.get("CFLAGS", ""))

    with tempfile.TemporaryDirectory(prefix="ghost-native-") as tmp:
        temporary = Path(tmp) / output.name
        if system == "windows" and compiler_name in {"cl", "cl.exe"}:
            command = compiler + [
                "/nologo", "/O2", "/LD", "/DGHOST_BUILD_DLL",
                *extra, *sources, "/link", f"/OUT:{temporary}",
            ]
        elif system == "windows":
            command = compiler + [
                "-std=c99", "-O2", "-Wall", "-Wextra", "-shared",
                "-DGHOST_BUILD_DLL", *extra, "-o", str(temporary), *sources, "-lm",
            ]
        elif system == "darwin":
            command = compiler + [
                "-std=c99", "-O2", "-Wall", "-Wextra", "-fPIC", "-dynamiclib",
                *extra, "-o", str(temporary), *sources, "-lm",
            ]
        else:
            command = compiler + [
                "-std=c99", "-O2", "-Wall", "-Wextra", "-fPIC", "-shared",
                *extra, "-o", str(temporary), *sources, "-lm",
            ]
        subprocess.run(command, cwd=tmp, check=True)
        if not temporary.is_file():
            raise RuntimeError("Le compilateur n'a pas produit la bibliothèque native attendue.")
        os.replace(temporary, output)
    return output


if __name__ == "__main__":
    target = build_native(ROOT / "python" / "ghost")
    print(f"ghost native engine built: {target}")
    print(f"Host: {platform.system()} / {platform.machine()}")
