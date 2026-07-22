# -*- coding: utf-8 -*-
"""Task-oriented workflow entry points.

High-level functions take a full business table (``df, target=...``) and
return structured report objects; low-level tools under ``src.binning`` /
``src.features`` keep the sklearn-style ``X, y`` interface.
"""

from .profiling import profile_data
from .risk import profile_risk

__all__ = ["profile_data", "profile_risk"]
