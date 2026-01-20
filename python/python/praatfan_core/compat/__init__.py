"""
Compatibility layer for parselmouth API.

This module provides a parselmouth-compatible API using praatfan-core as the backend.
It allows existing parselmouth code to work with minimal changes.

Usage:
    # Instead of:
    # import parselmouth
    # from parselmouth.praat import call

    # Use:
    from praatfan_core.compat import parselmouth
    from praatfan_core.compat.parselmouth import call
"""

from . import parselmouth

__all__ = ['parselmouth']
