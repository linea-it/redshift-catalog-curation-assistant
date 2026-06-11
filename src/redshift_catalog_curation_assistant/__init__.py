from .example_module import greetings, meaning

try:
    from ._version import __version__
except ModuleNotFoundError:  # pragma: no cover - exercised before editable install
    __version__ = "0+unknown"

__all__ = ["greetings", "meaning", "__version__"]
