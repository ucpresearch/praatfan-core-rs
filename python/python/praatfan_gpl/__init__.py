"""
praatfan_gpl - Praat-compatible acoustic analysis in Python

This package provides exact reimplementations of Praat's acoustic analysis
algorithms, designed to produce bit-accurate output matching Praat/parselmouth.

Example usage:

    import praatfan_gpl

    # Load an audio file
    sound = praatfan_gpl.Sound.from_file("speech.wav")

    # Compute pitch
    pitch = sound.to_pitch(0.01, 75.0, 600.0)
    f0 = pitch.get_value_at_time(0.5, "hertz", "linear")

    # Compute formants
    formant = sound.to_formant_burg(0.01, 5, 5500.0, 0.025, 50.0)
    f1 = formant.get_value_at_time(1, 0.5, "hertz", "linear")
    f2 = formant.get_value_at_time(2, 0.5, "hertz", "linear")

    # Compute intensity
    intensity = sound.to_intensity(100.0, 0.01)
    db = intensity.get_value_at_time(0.5, "cubic")

    # Compute spectrum
    spectrum = sound.to_spectrum(fast=True)
    cog = spectrum.get_center_of_gravity(2.0)

    # Compute spectrogram
    spectrogram = sound.to_spectrogram(0.005, 5000.0, 0.002, 20.0, "gaussian")
    data = spectrogram.values()  # numpy array [freq, time]

    # Compute harmonicity (HNR)
    hnr = sound.to_harmonicity_ac(0.01, 75.0, 0.1, 1.0)
    hnr_db = hnr.get_value_at_time(0.5, "linear")
"""

# Import the Rust extension module
from .praatfan_gpl import (
    Sound,
    Pitch,
    Formant,
    Intensity,
    Spectrum,
    Spectrogram,
    Harmonicity,
)

# Compatibility layer for parselmouth API
from . import compat

__all__ = [
    "Sound",
    "Pitch",
    "Formant",
    "Intensity",
    "Spectrum",
    "Spectrogram",
    "Harmonicity",
    "compat",
]

__version__ = "0.1.1"
