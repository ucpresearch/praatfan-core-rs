"""
Compatibility layer for parselmouth API.

This module provides a parselmouth-compatible API using praatfan-gpl as the backend.
It allows existing parselmouth code to work with minimal changes.

Usage:
    # Instead of:
    # import parselmouth
    # from parselmouth.praat import call

    # Use:
    from praatfan_gpl.compat import parselmouth
    from praatfan_gpl.compat.parselmouth import call
"""

from . import parselmouth

__all__ = ['parselmouth']
