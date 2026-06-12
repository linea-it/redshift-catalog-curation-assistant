try:
    from ._version import __version__
except ModuleNotFoundError:  # pragma: no cover - exercised before editable install
    __version__ = "0+unknown"

__all__ = ["__version__"]
