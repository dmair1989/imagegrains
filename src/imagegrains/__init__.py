
"""A software library for segmenting and measuring of sedimentary particles in images."""
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("imagegrains")
except PackageNotFoundError:
    __version__ = "uninstalled"