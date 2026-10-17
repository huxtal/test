"""Automated integration and fitting of TA Instruments Affinity ITC (.nitc) data."""

from .fit import FitResult, bootstrap, fit
from .integrate import integrate
from .models import MODELS, Design, get_model
from .nitc import ITCRun, read_nitc, read_tables

__all__ = ["FitResult", "bootstrap", "fit", "integrate", "MODELS", "Design",
           "get_model", "ITCRun", "read_nitc", "read_tables"]
__version__ = "0.1.0"
