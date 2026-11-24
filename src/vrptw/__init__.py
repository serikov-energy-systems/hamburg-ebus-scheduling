"""VRPTW: ALNS in Python and in C++ (pybind11), exact methods for small instances, Solomon benchmark."""
from . import alns_py
from .instance import Instance, check, load

try:  # the C++ kernel is optional: built by setup.py when pybind11 and a compiler are available
    from . import _alns_cpp
except ImportError:  # pragma: no cover
    _alns_cpp = None


def solve(inst: Instance, iterations: int = 5000, seed: int = 0, backend: str = "cpp", fast: bool = True):
    """Returns (routes, distance, vehicles, milliseconds). backend 'cpp' (fast=False tests insertions the naive way) or 'python'."""
    if backend == "cpp":
        if _alns_cpp is None:
            raise RuntimeError("C++ kernel not built; use backend='python' or build with: pip install -e .")
        return _alns_cpp.solve(*inst.args(), iterations, seed, fast)
    return alns_py.solve(*inst.args(), iterations, seed)


def profile(inst: Instance, iterations: int = 1000, seed: int = 0, fast: bool = True):
    """C++ run with the time shares of the feasibility test, destroy and repair: (routes, distance, vehicles, ms, profile)."""
    return _alns_cpp.profile(*inst.args(), iterations, seed, fast)
