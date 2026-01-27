"""
Parselmouth-compatible API for praatfan-gpl.

This module provides a drop-in replacement for parselmouth, allowing existing
code to work with praatfan-gpl with minimal changes.

Supported functionality:
    - Sound loading and creation
    - Pitch analysis (To Pitch)
    - Formant analysis (To Formant (burg))
    - Intensity analysis (To Intensity)
    - Harmonicity analysis (To Harmonicity (cc), To Harmonicity (ac))
    - Spectrum analysis (To Spectrum)
    - Spectrogram analysis (To Spectrogram)
    - Value queries (Get value at time, Get bandwidth at time, etc.)

Usage:
    from praatfan_gpl.compat import parselmouth
    from praatfan_gpl.compat.parselmouth import call

    snd = parselmouth.Sound("audio.wav")
    pitch = call(snd, "To Pitch", 0.01, 75.0, 600.0)
    f0 = call(pitch, "Get value at time", 0.5, "Hertz", "Linear")
"""

from __future__ import annotations
from typing import Any, Optional
import numpy as np

from praatfan_gpl import (
    Sound as _Sound,
    Pitch as _Pitch,
    Formant as _Formant,
    Intensity as _Intensity,
    Harmonicity as _Harmonicity,
    Spectrum as _Spectrum,
    Spectrogram as _Spectrogram,
)


class Sound:
    """Parselmouth-compatible Sound wrapper."""

    def __init__(self, path_or_samples, sample_rate: Optional[float] = None):
        """
        Create a Sound from a file path or samples.

        Args:
            path_or_samples: Either a file path (str) or numpy array of samples
            sample_rate: Sample rate (required if path_or_samples is an array)
        """
        if isinstance(path_or_samples, str):
            self._inner = _Sound.from_file(path_or_samples)
        elif isinstance(path_or_samples, np.ndarray):
            if sample_rate is None:
                raise ValueError("sample_rate is required when creating from samples")
            self._inner = _Sound(path_or_samples.astype(np.float64), sample_rate)
        else:
            raise TypeError(f"Expected str or ndarray, got {type(path_or_samples)}")

    @property
    def values(self) -> np.ndarray:
        """Get samples as numpy array (parselmouth compatibility)."""
        return self._inner.samples()

    @property
    def sampling_frequency(self) -> float:
        """Get sample rate (parselmouth naming)."""
        return self._inner.sample_rate

    @property
    def duration(self) -> float:
        """Get duration in seconds."""
        return self._inner.duration

    @property
    def xmin(self) -> float:
        """Start time."""
        return self._inner.start_time

    @property
    def xmax(self) -> float:
        """End time."""
        return self._inner.end_time


class _WrappedPitch:
    """Wrapped Pitch object for call() compatibility."""

    def __init__(self, inner: _Pitch):
        self._inner = inner

    @property
    def values(self) -> np.ndarray:
        """Get pitch values as numpy array."""
        return self._inner.values()


class _WrappedFormant:
    """Wrapped Formant object for call() compatibility."""

    def __init__(self, inner: _Formant):
        self._inner = inner


class _WrappedIntensity:
    """Wrapped Intensity object for call() compatibility."""

    def __init__(self, inner: _Intensity):
        self._inner = inner

    @property
    def values(self) -> np.ndarray:
        """Get intensity values as numpy array."""
        return self._inner.values()


class _WrappedHarmonicity:
    """Wrapped Harmonicity object for call() compatibility."""

    def __init__(self, inner: _Harmonicity):
        self._inner = inner

    @property
    def values(self) -> np.ndarray:
        """Get HNR values as numpy array."""
        return self._inner.values()


class _WrappedSpectrum:
    """Wrapped Spectrum object for call() compatibility."""

    def __init__(self, inner: _Spectrum):
        self._inner = inner


class _WrappedSpectrogram:
    """Wrapped Spectrogram object for call() compatibility."""

    def __init__(self, inner: _Spectrogram):
        self._inner = inner

    @property
    def values(self) -> np.ndarray:
        """Get spectrogram values as numpy array."""
        return self._inner.values()

    @property
    def xmin(self) -> float:
        return self._inner.start_time

    @property
    def xmax(self) -> float:
        return self._inner.start_time + self._inner.time_step * self._inner.num_frames

    @property
    def ymin(self) -> float:
        return self._inner.freq_min

    @property
    def ymax(self) -> float:
        return self._inner.freq_max

    def get_time_from_frame_number(self, frame: int) -> float:
        """Get time from 1-based frame number (parselmouth convention)."""
        return self._inner.get_time_from_frame(frame - 1)


def call(obj: Any, method: str, *args) -> Any:
    """
    Call a Praat method on an object.

    This function emulates parselmouth's call() function, translating
    Praat scripting commands to praatfan-core method calls.

    Args:
        obj: The object to call the method on (Sound, Pitch, Formant, etc.)
        method: The Praat method name (e.g., "To Pitch", "Get value at time")
        *args: Method arguments

    Returns:
        The result of the method call
    """
    method_lower = method.lower().replace(" ", "_").replace("(", "").replace(")", "")

    # Sound methods
    if isinstance(obj, Sound):
        return _call_sound(obj, method, method_lower, args)

    # Pitch methods
    if isinstance(obj, _WrappedPitch):
        return _call_pitch(obj, method, method_lower, args)

    # Formant methods
    if isinstance(obj, _WrappedFormant):
        return _call_formant(obj, method, method_lower, args)

    # Intensity methods
    if isinstance(obj, _WrappedIntensity):
        return _call_intensity(obj, method, method_lower, args)

    # Harmonicity methods
    if isinstance(obj, _WrappedHarmonicity):
        return _call_harmonicity(obj, method, method_lower, args)

    # Spectrum methods
    if isinstance(obj, _WrappedSpectrum):
        return _call_spectrum(obj, method, method_lower, args)

    # Spectrogram methods
    if isinstance(obj, _WrappedSpectrogram):
        return _call_spectrogram(obj, method, method_lower, args)

    raise NotImplementedError(f"call() not implemented for {type(obj).__name__}")


def _call_sound(obj: Sound, method: str, method_lower: str, args: tuple) -> Any:
    """Handle Sound method calls."""
    snd = obj._inner

    if method_lower == "get_total_duration":
        return snd.duration

    if method_lower == "to_pitch":
        time_step, pitch_floor, pitch_ceiling = args
        return _WrappedPitch(snd.to_pitch(time_step, pitch_floor, pitch_ceiling))

    if method_lower == "to_intensity":
        min_pitch, time_step = args
        return _WrappedIntensity(snd.to_intensity(min_pitch, time_step))

    if method_lower == "to_formant_burg":
        time_step, max_formants, max_formant_hz, window_length, pre_emphasis = args
        return _WrappedFormant(snd.to_formant_burg(
            time_step, int(max_formants), max_formant_hz, window_length, pre_emphasis
        ))

    if method_lower == "to_harmonicity_cc":
        time_step, min_pitch, silence_threshold, periods_per_window = args
        return _WrappedHarmonicity(snd.to_harmonicity_cc(
            time_step, min_pitch, silence_threshold, periods_per_window
        ))

    if method_lower == "to_harmonicity_ac":
        time_step, min_pitch, silence_threshold, periods_per_window = args
        return _WrappedHarmonicity(snd.to_harmonicity_ac(
            time_step, min_pitch, silence_threshold, periods_per_window
        ))

    if method_lower == "to_spectrum":
        fast = args[0] if args else True
        if isinstance(fast, str):
            fast = fast.lower() == "yes"
        return _WrappedSpectrum(snd.to_spectrum(fast))

    if method_lower == "to_spectrogram":
        time_step, max_freq, window_length, freq_step, window_shape = args
        return _WrappedSpectrogram(snd.to_spectrogram(
            window_length, max_freq, time_step, freq_step, window_shape.lower()
        ))

    if method_lower == "extract_part":
        start_time, end_time, window_shape, relative_width, preserve_times = args
        if isinstance(preserve_times, str):
            preserve_times = preserve_times.lower() in ("yes", "true", "1")
        new_snd = snd.extract_part(
            start_time, end_time, window_shape.lower(), relative_width, preserve_times
        )
        result = Sound.__new__(Sound)
        result._inner = new_snd
        return result

    if method_lower == "filter_pre-emphasis" or method_lower == "filter_pre_emphasis":
        from_freq = args[0]
        new_snd = snd.pre_emphasis(from_freq)
        result = Sound.__new__(Sound)
        result._inner = new_snd
        return result

    raise NotImplementedError(f"Sound.{method} not implemented")


def _call_pitch(obj: _WrappedPitch, method: str, method_lower: str, args: tuple) -> Any:
    """Handle Pitch method calls."""
    pitch = obj._inner

    if method_lower == "get_value_at_time":
        time, unit, interpolation = args
        return pitch.get_value_at_time(time, unit.lower(), interpolation.lower())

    raise NotImplementedError(f"Pitch.{method} not implemented")


def _call_formant(obj: _WrappedFormant, method: str, method_lower: str, args: tuple) -> Any:
    """Handle Formant method calls."""
    formant = obj._inner

    if method_lower == "get_value_at_time":
        formant_number, time, unit, interpolation = args
        return formant.get_value_at_time(
            int(formant_number), time, unit.lower(), interpolation.lower()
        )

    if method_lower == "get_bandwidth_at_time":
        formant_number, time, unit, interpolation = args
        return formant.get_bandwidth_at_time(
            int(formant_number), time, unit.lower(), interpolation.lower()
        )

    raise NotImplementedError(f"Formant.{method} not implemented")


def _call_intensity(obj: _WrappedIntensity, method: str, method_lower: str, args: tuple) -> Any:
    """Handle Intensity method calls."""
    intensity = obj._inner

    if method_lower == "get_value_at_time":
        time, interpolation = args
        return intensity.get_value_at_time(time, interpolation.lower())

    raise NotImplementedError(f"Intensity.{method} not implemented")


def _call_harmonicity(obj: _WrappedHarmonicity, method: str, method_lower: str, args: tuple) -> Any:
    """Handle Harmonicity method calls."""
    harmonicity = obj._inner

    if method_lower == "get_value_at_time":
        time, interpolation = args
        return harmonicity.get_value_at_time(time, interpolation.lower())

    raise NotImplementedError(f"Harmonicity.{method} not implemented")


def _call_spectrum(obj: _WrappedSpectrum, method: str, method_lower: str, args: tuple) -> Any:
    """Handle Spectrum method calls."""
    spectrum = obj._inner

    if method_lower == "get_centre_of_gravity" or method_lower == "get_center_of_gravity":
        power = args[0]
        return spectrum.get_center_of_gravity(power)

    if method_lower == "get_standard_deviation":
        power = args[0]
        return spectrum.get_standard_deviation(power)

    if method_lower == "get_skewness":
        power = args[0]
        return spectrum.get_skewness(power)

    if method_lower == "get_kurtosis":
        power = args[0]
        return spectrum.get_kurtosis(power)

    if method_lower == "get_band_energy":
        freq_min, freq_max = args
        return spectrum.get_band_energy(freq_min, freq_max)

    raise NotImplementedError(f"Spectrum.{method} not implemented")


def _call_spectrogram(obj: _WrappedSpectrogram, method: str, method_lower: str, args: tuple) -> Any:
    """Handle Spectrogram method calls."""
    spectrogram = obj._inner

    if method_lower == "get_time_from_frame_number":
        frame = args[0]
        return spectrogram.get_time_from_frame(int(frame) - 1)  # 1-based to 0-based

    raise NotImplementedError(f"Spectrogram.{method} not implemented")


# Submodule for praat-style call
class praat:
    """Submodule providing Praat-style call function."""
    call = staticmethod(call)
