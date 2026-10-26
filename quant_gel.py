#!/usr/bin/env python3
"""Quantify SDS-PAGE bands.  Run ``python quant_gel.py -h``."""

import sys

from gelquant.cli import main

if __name__ == "__main__":
    sys.exit(main())
