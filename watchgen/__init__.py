"""watchgen: Mini implementations of population genetics algorithms."""

import importlib as _importlib

_MODULES = [
    "mini_psmc",
    "mini_smcpp",
    "mini_lshmm",
    "mini_msprime",
    "mini_argweaver",
    "mini_tsinfer",
    "mini_singer",
    "mini_threads",
    "mini_tsdate",
    "mini_moments",
    "mini_dadi",
    "mini_momi2",
    "mini_gamma_smc",
    "mini_phlash",
    "mini_clues",
    "mini_slim",
    "mini_relate",
    "mini_discoal",
]


def __getattr__(name):
    if name in _MODULES:
        return _importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return __all__ + _MODULES


__all__ = []
