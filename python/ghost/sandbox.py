"""ghost.sandbox — exécution de tests Python/C++ dans un environnement Linux isolé.

Aucune exécution de code généré n'est permise sans bubblewrap. Le runner échoue
fermée si les namespaces Linux ne sont pas disponibles ; il ne retombe jamais sur
subprocess sans isolation ni ne donne accès au réseau ou au dossier utilisateur.
"""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time


class SandboxUnavailable(RuntimeError):
    pass


class SandboxExecutionError(RuntimeError):
    pass


class CodeSandbox:
    def __init__(self, *, timeout_seconds=8, memory_mb=384, cpu_seconds=5,
                 output_limit=32000, source_limit=64000):
        self.bwrap = shutil.which("bwrap")
        self.timeout_seconds = max(1.0, min(float(timeout_seconds), 60.0))
        self.memory_bytes = max(64, min(int(memory_mb), 2048)) * 1024 * 1024
        self.cpu_seconds = max(1, min(int(cpu_seconds), 30))
        self.output_limit = max(1024, min(int(output_limit), 256000))
        self.source_limit = max(1024, min(int(source_limit), 512000))

    @property
    def available(self):
        return bool(self.bwrap and sys.platform.startswith("linux"))

    def _require(self):
        if not self.available:
            raise SandboxUnavailable(
                "Exécution bloquée : bubblewrap/bwrap avec namespaces Linux est requis. "
                "Aucun mode non isolé n'est proposé en secours."
            )

    @staticmethod
    def _limits(memory_bytes, cpu_seconds):
        def apply():
            try:
                import resource
                resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds + 1))
                resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
                resource.setrlimit(resource.RLIMIT_FSIZE, (4 * 1024 * 1024, 4 * 1024 * 1024))
                resource.setrlimit(resource.RLIMIT_NOFILE, (32, 32))
                if hasattr(resource, "RLIMIT_CORE"):
                    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
            except Exception:
                os._exit(126)
        return apply

    @staticmethod
    def _mount_args():
        args = ["--unshare-all", "--die-with-parent", "--new-session", "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp"]
        # Bibliothèques en lecture seule uniquement ; le home utilisateur et /var ne sont pas montés.
        for path, destination in (("/usr", "/usr"), ("/bin", "/bin"),
                                  ("/lib", "/lib"), ("/lib64", "/lib64")):
            source = Path(path)
            if source.exists():
                resolved = source.resolve()
                if destination in {"/bin", "/lib", "/lib64"} and resolved == Path("/usr") / destination.lstrip("/"):
                    args.extend(["--symlink", "usr/" + destination.lstrip("/"), destination])
                    continue
                args.extend(["--ro-bind", str(resolved), destination])
        args.extend(["--dir", "/etc"])
        cache = Path("/etc/ld.so.cache")
        if cache.exists():
            args.extend(["--ro-bind", str(cache.resolve()), "/etc/ld.so.cache"])
        return args

    def _run(self, workdir: Path, command: list[str]):
        self._require()
        args = [self.bwrap, *self._mount_args(), "--bind", str(workdir), "/work",
                "--chdir", "/work", "--setenv", "HOME", "/tmp",
                "--setenv", "TMPDIR", "/tmp", "--setenv", "PATH", "/usr/local/bin:/usr/bin:/bin",
                "--", *command]
        started = time.monotonic()
        try:
            proc = subprocess.run(
                args, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                timeout=self.timeout_seconds, check=False,
                preexec_fn=self._limits(self.memory_bytes, self.cpu_seconds),
            )
        except subprocess.TimeoutExpired as exc:
            raise SandboxExecutionError("Limite de temps dépassée dans le bac à sable.") from exc
        elapsed = time.monotonic() - started
        stdout = proc.stdout[:self.output_limit].decode("utf-8", errors="replace")
        stderr = proc.stderr[:self.output_limit].decode("utf-8", errors="replace")
        if len(proc.stdout) > self.output_limit:
            stdout += "\n[stdout tronqué]"
        if len(proc.stderr) > self.output_limit:
            stderr += "\n[stderr tronqué]"
        return {"exit_code": proc.returncode, "stdout": stdout, "stderr": stderr,
                "elapsed_seconds": elapsed, "isolated": True, "network": "disabled"}

    def run_python(self, source: str):
        self._require()
        if not isinstance(source, str) or len(source.encode("utf-8")) > self.source_limit:
            raise ValueError("Source Python vide ou trop volumineuse.")
        with tempfile.TemporaryDirectory(prefix="ghost-sandbox-") as tmp:
            work = Path(tmp)
            (work / "main.py").write_text(source, encoding="utf-8")
            python = str(Path(sys.executable).resolve())
            return self._run(work, [python, "-I", "-S", "-B", "/work/main.py"])

    def run_cpp(self, source: str):
        self._require()
        if not isinstance(source, str) or len(source.encode("utf-8")) > self.source_limit:
            raise ValueError("Source C++ vide ou trop volumineuse.")
        compiler = shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
        if not compiler:
            raise SandboxUnavailable("Aucun compilateur C++ n'est installé sur l'hôte.")
        compiler = str(Path(compiler).resolve())
        with tempfile.TemporaryDirectory(prefix="ghost-sandbox-cpp-") as tmp:
            work = Path(tmp)
            (work / "main.cpp").write_text(source, encoding="utf-8")
            compile_result = self._run(work, [compiler, "-std=c++17", "-O2", "-pipe", "/work/main.cpp", "-o", "/work/program"])
            if compile_result["exit_code"] != 0:
                return {"stage": "compile", **compile_result}
            run_result = self._run(work, ["/work/program"])
            return {"stage": "run", "compile": compile_result, **run_result}
