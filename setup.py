"""Optional C++ extension for the VRPTW kernel; the package installs without it when no compiler is found."""
from setuptools import setup

ext = []
try:
    from pybind11.setup_helpers import Pybind11Extension

    ext = [Pybind11Extension("vrptw._alns_cpp", ["src/vrptw/cpp/alns_core.cpp"], cxx_std=17, optional=True)]
except ImportError:  # pybind11 not installed
    pass

setup(ext_modules=ext)
