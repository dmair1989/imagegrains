
"""A software library for segmenting and measuring of sedimentary particles in images."""
from importlib.metadata import PackageNotFoundError, version
#import data_loader, segmentation_helper, grainsizing, gsd_uncertainty, plotting

try:
    __version__ = version("imagegrains")
except PackageNotFoundError:
    __version__ = "uninstalled"