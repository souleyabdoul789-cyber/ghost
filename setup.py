"""Packaging de ghost : moteur C compilé pour la plateforme de construction."""
from pathlib import Path
import sys
from setuptools import setup
from setuptools.command.build_py import build_py
from setuptools.dist import Distribution
from setuptools.command.bdist_wheel import bdist_wheel
sys.path.insert(0, str(Path(__file__).parent.resolve()))
from build_native import build_native

ROOT = Path(__file__).parent.resolve()


class NativeBuildPy(build_py):
    def run(self):
        super().run()
        build_native(Path(self.build_lib) / "ghost")


class BinaryDistribution(Distribution):
    """Le wheel est spécifique à l'OS/architecture, pas à l'ABI Python.

    Le moteur est une bibliothèque C chargée via ctypes : il n'utilise pas
    l'ABI CPython et peut donc partager un tag py3-none sur une même plateforme.
    """
    def has_ext_modules(self):
        return True


class PlatformWheel(bdist_wheel):
    def get_tag(self):
        _python, _abi, platform_tag = super().get_tag()
        return "py3", "none", platform_tag


setup(cmdclass={"build_py": NativeBuildPy, "bdist_wheel": PlatformWheel},
      distclass=BinaryDistribution)
