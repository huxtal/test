#!/usr/bin/env python3
"""Fit TA Instruments Affinity ITC .nitc files.  Run ``python fit_itc.py -h``."""

import sys

from itcfit.cli import main

if __name__ == "__main__":
    sys.exit(main())
