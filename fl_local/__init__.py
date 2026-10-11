"""Local FL helpers.

Heavy torch/cache modules are loaded lazily.  This preserves the public names
while allowing pure integrity helpers (for example ``fl_local.e7_artifacts``)
to be imported by a no-GPU/no-torch audit environment.
"""

from importlib import import_module

__all__ = [
    "CFG_FULL",
    "load_or_build_nbaiot_cache",
    "load_or_build_cic_cache",
    "run_fl_experiment",
    "set_global_seed",
]


def __getattr__(name):
    if name in {"CFG_FULL", "run_fl_experiment", "set_global_seed"}:
        return getattr(import_module(".engine", __name__), name)
    if name == "load_or_build_nbaiot_cache":
        return getattr(import_module(".nbaiot_cache", __name__), name)
    if name == "load_or_build_cic_cache":
        return getattr(import_module(".cic_cache", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
