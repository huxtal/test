"""Lane/band detection and quantification for SDS-PAGE gel images."""

from .image import GelImage, load_gel
from .ladder import LADDERS, REFERENCE_BANDS, calibrate
from .quant import GelResult, analyse_gel

__all__ = ["GelImage", "load_gel", "LADDERS", "REFERENCE_BANDS", "calibrate",
           "GelResult", "analyse_gel"]
__version__ = "0.1.0"
