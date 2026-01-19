//! Spectrogram (time-frequency representation)
//!
//! This module computes spectrograms from audio signals using the Short-Time
//! Fourier Transform (STFT). A spectrogram shows how the frequency content
//! of a signal varies over time.
//!
//! The algorithm:
//! 1. Divide signal into overlapping frames
//! 2. Apply window function (Gaussian for Praat compatibility)
//! 3. Compute FFT of each frame
//! 4. Store power/magnitude spectrum for each frame

use crate::utils::Fft;
use crate::window::WindowShape;
use crate::Sound;

/// Spectrogram representation (time-frequency matrix)
#[derive(Debug, Clone)]
pub struct Spectrogram {
    /// Power values in Pa²/Hz, stored as [frequency_bin][time_frame]
    /// Each inner Vec is a time series for one frequency bin
    data: Vec<Vec<f64>>,
    /// Time of first frame center
    start_time: f64,
    /// Time step between frames
    time_step: f64,
    /// Minimum frequency (always 0)
    freq_min: f64,
    /// Maximum frequency (Nyquist or user-specified)
    freq_max: f64,
    /// Frequency step (Hz per bin)
    freq_step: f64,
    /// Number of time frames
    num_frames: usize,
    /// Number of frequency bins
    num_freq_bins: usize,
}

impl Spectrogram {
    /// Compute spectrogram from a Sound
    ///
    /// # Arguments
    /// * `sound` - Input audio signal
    /// * `time_step` - Time between analysis frames (seconds)
    /// * `window_length` - Duration of analysis window (seconds)
    /// * `max_frequency` - Maximum frequency to include (Hz, 0 for Nyquist)
    /// * `frequency_step` - Frequency resolution (Hz, 0 for automatic)
    /// * `window_shape` - Window function (typically Gaussian)
    ///
    /// # Returns
    /// Spectrogram with power spectral density values
    pub fn from_sound(
        sound: &Sound,
        time_step: f64,
        window_length: f64,
        max_frequency: f64,
        frequency_step: f64,
        window_shape: WindowShape,
    ) -> Self {
        let sample_rate = sound.sample_rate();
        let samples = sound.samples();
        let nyquist = sample_rate / 2.0;

        // Validate parameters
        let max_frequency = if max_frequency <= 0.0 || max_frequency > nyquist {
            nyquist
        } else {
            max_frequency
        };

        let window_length = window_length.max(0.001).min(1.0);
        let time_step = if time_step <= 0.0 {
            window_length / 8.0
        } else {
            time_step
        };

        // Window parameters
        let window_samples = (window_length * sample_rate).round() as usize;
        let half_window = window_samples / 2;

        // FFT size (power of 2 for efficiency)
        let fft_size = if frequency_step > 0.0 {
            // User-specified frequency resolution
            let desired_bins = (sample_rate / frequency_step).ceil() as usize;
            desired_bins.next_power_of_two()
        } else {
            window_samples.next_power_of_two()
        };

        let freq_step = sample_rate / fft_size as f64;
        let num_freq_bins = (max_frequency / freq_step).ceil() as usize + 1;
        let num_freq_bins = num_freq_bins.min(fft_size / 2 + 1);

        // Generate window (Gaussian with sigma appropriate for spectrogram)
        let window = match window_shape {
            WindowShape::Gaussian => WindowShape::Gaussian.generate(window_samples, Some(0.4)),
            _ => window_shape.generate(window_samples, None),
        };

        // Compute window energy for normalization
        let window_energy: f64 = window.iter().map(|&w| w * w).sum();

        // Frame timing
        let first_frame_time = sound.start_time() + window_length / 2.0;
        let last_frame_time = sound.end_time() - window_length / 2.0;

        if first_frame_time >= last_frame_time || samples.is_empty() {
            return Self {
                data: Vec::new(),
                start_time: sound.start_time(),
                time_step,
                freq_min: 0.0,
                freq_max: max_frequency,
                freq_step,
                num_frames: 0,
                num_freq_bins: 0,
            };
        }

        let num_frames = ((last_frame_time - first_frame_time) / time_step).floor() as usize + 1;

        // Initialize data storage [freq_bin][time_frame]
        let mut data: Vec<Vec<f64>> = (0..num_freq_bins)
            .map(|_| Vec::with_capacity(num_frames))
            .collect();

        let mut fft = Fft::new();

        for frame_idx in 0..num_frames {
            let frame_time = first_frame_time + frame_idx as f64 * time_step;
            let center_sample = sound.time_to_index_clamped(frame_time);

            // Extract windowed frame
            let start_sample = center_sample.saturating_sub(half_window);
            let end_sample = (center_sample + half_window).min(samples.len());

            let mut windowed = vec![0.0; window_samples];
            for (i, sample_idx) in (start_sample..end_sample).enumerate() {
                if i < window.len() {
                    windowed[i] = samples[sample_idx] * window[i];
                }
            }

            // Compute power spectrum
            let power_spectrum = fft.power_spectrum(&windowed, fft_size);

            // Store power values (normalized to Pa²/Hz)
            // Normalization: divide by window energy and frequency resolution
            let norm_factor = if window_energy > 0.0 {
                1.0 / (window_energy * freq_step)
            } else {
                1.0
            };

            for (freq_bin, power) in power_spectrum.iter().take(num_freq_bins).enumerate() {
                data[freq_bin].push(power * norm_factor);
            }
        }

        Self {
            data,
            start_time: first_frame_time,
            time_step,
            freq_min: 0.0,
            freq_max: max_frequency,
            freq_step,
            num_frames,
            num_freq_bins,
        }
    }

    /// Get the power at a specific time and frequency
    ///
    /// # Arguments
    /// * `time` - Time in seconds
    /// * `frequency` - Frequency in Hz
    ///
    /// # Returns
    /// Power spectral density in Pa²/Hz, or None if out of range
    pub fn get_power_at(&self, time: f64, frequency: f64) -> Option<f64> {
        if self.data.is_empty() {
            return None;
        }

        // Get frame index
        let frame_pos = (time - self.start_time) / self.time_step;
        if frame_pos < -0.5 || frame_pos > self.num_frames as f64 - 0.5 {
            return None;
        }
        let frame = frame_pos.round() as usize;
        let frame = frame.min(self.num_frames - 1);

        // Get frequency bin
        if frequency < self.freq_min || frequency > self.freq_max {
            return None;
        }
        let freq_bin = (frequency / self.freq_step).round() as usize;
        let freq_bin = freq_bin.min(self.num_freq_bins - 1);

        self.data.get(freq_bin).and_then(|row| row.get(frame).copied())
    }

    /// Get power in dB at a specific time and frequency
    pub fn get_power_db_at(&self, time: f64, frequency: f64) -> Option<f64> {
        self.get_power_at(time, frequency).map(|p| {
            if p > 0.0 {
                10.0 * p.log10()
            } else {
                f64::NEG_INFINITY
            }
        })
    }

    /// Get the entire power matrix
    ///
    /// Returns a reference to the internal data stored as [frequency_bin][time_frame]
    pub fn values(&self) -> &Vec<Vec<f64>> {
        &self.data
    }

    /// Get power values for a specific frequency bin across all times
    pub fn get_frequency_slice(&self, frequency: f64) -> Option<&[f64]> {
        if frequency < self.freq_min || frequency > self.freq_max {
            return None;
        }
        let freq_bin = (frequency / self.freq_step).round() as usize;
        self.data.get(freq_bin).map(|v| v.as_slice())
    }

    /// Get power values for a specific time frame across all frequencies
    pub fn get_time_slice(&self, time: f64) -> Option<Vec<f64>> {
        let frame_pos = (time - self.start_time) / self.time_step;
        if frame_pos < -0.5 || frame_pos > self.num_frames as f64 - 0.5 {
            return None;
        }
        let frame = frame_pos.round() as usize;
        let frame = frame.min(self.num_frames.saturating_sub(1));

        Some(
            self.data
                .iter()
                .map(|row| row.get(frame).copied().unwrap_or(0.0))
                .collect(),
        )
    }

    /// Get the time of a specific frame
    pub fn get_time_from_frame(&self, frame: usize) -> f64 {
        self.start_time + frame as f64 * self.time_step
    }

    /// Get the frequency of a specific bin
    pub fn get_frequency_from_bin(&self, bin: usize) -> f64 {
        bin as f64 * self.freq_step
    }

    /// Get the frame index nearest to a time
    pub fn get_frame_from_time(&self, time: f64) -> usize {
        let pos = (time - self.start_time) / self.time_step;
        (pos.round() as usize).min(self.num_frames.saturating_sub(1))
    }

    /// Get the frequency bin nearest to a frequency
    pub fn get_bin_from_frequency(&self, frequency: f64) -> usize {
        let bin = (frequency / self.freq_step).round() as usize;
        bin.min(self.num_freq_bins.saturating_sub(1))
    }

    /// Get the number of time frames
    pub fn num_frames(&self) -> usize {
        self.num_frames
    }

    /// Get the number of frequency bins
    pub fn num_freq_bins(&self) -> usize {
        self.num_freq_bins
    }

    /// Get the time step between frames
    pub fn time_step(&self) -> f64 {
        self.time_step
    }

    /// Get the frequency step (resolution)
    pub fn freq_step(&self) -> f64 {
        self.freq_step
    }

    /// Get the minimum frequency
    pub fn freq_min(&self) -> f64 {
        self.freq_min
    }

    /// Get the maximum frequency
    pub fn freq_max(&self) -> f64 {
        self.freq_max
    }

    /// Get the start time (first frame center)
    pub fn start_time(&self) -> f64 {
        self.start_time
    }

    /// Get the end time (last frame center)
    pub fn end_time(&self) -> f64 {
        if self.num_frames == 0 {
            self.start_time
        } else {
            self.start_time + (self.num_frames - 1) as f64 * self.time_step
        }
    }

    /// Get the total duration
    pub fn duration(&self) -> f64 {
        self.end_time() - self.start_time
    }

    /// Get the total energy (sum of all power values)
    pub fn total_energy(&self) -> f64 {
        self.data
            .iter()
            .flat_map(|row| row.iter())
            .sum::<f64>()
            * self.time_step
            * self.freq_step
    }
}

// Add to_spectrogram method to Sound
impl Sound {
    /// Compute spectrogram from this sound
    ///
    /// # Arguments
    /// * `time_step` - Time between frames (0.0 for automatic)
    /// * `max_frequency` - Maximum frequency to show (0.0 for Nyquist)
    /// * `window_length` - Analysis window duration (typically 0.005 for wideband, 0.03 for narrowband)
    /// * `frequency_step` - Frequency resolution (0.0 for automatic)
    /// * `window_shape` - Window function (typically Gaussian)
    ///
    /// # Returns
    /// Spectrogram object
    pub fn to_spectrogram(
        &self,
        time_step: f64,
        max_frequency: f64,
        window_length: f64,
        frequency_step: f64,
        window_shape: WindowShape,
    ) -> Spectrogram {
        Spectrogram::from_sound(
            self,
            time_step,
            window_length,
            max_frequency,
            frequency_step,
            window_shape,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spectrogram_basic() {
        let sound = Sound::create_tone(440.0, 0.5, 44100.0, 0.5, 0.0);

        let spectrogram = sound.to_spectrogram(0.005, 5000.0, 0.03, 0.0, WindowShape::Gaussian);

        // Should have frames and frequency bins
        assert!(spectrogram.num_frames() > 0);
        assert!(spectrogram.num_freq_bins() > 0);
    }

    #[test]
    fn test_spectrogram_pure_tone_peak() {
        // Create a pure tone at 1000 Hz
        let freq = 1000.0;
        let sound = Sound::create_tone(freq, 0.3, 44100.0, 0.5, 0.0);

        let spectrogram = sound.to_spectrogram(0.01, 5000.0, 0.03, 0.0, WindowShape::Gaussian);

        // Find the frequency bin with maximum power at the middle frame
        let middle_frame = spectrogram.num_frames() / 2;
        let middle_time = spectrogram.get_time_from_frame(middle_frame);

        let time_slice = spectrogram.get_time_slice(middle_time).unwrap();

        let mut max_power = 0.0;
        let mut max_bin = 0;
        for (bin, &power) in time_slice.iter().enumerate() {
            if power > max_power {
                max_power = power;
                max_bin = bin;
            }
        }

        let peak_freq = spectrogram.get_frequency_from_bin(max_bin);

        // Peak should be near 1000 Hz
        assert!(
            (peak_freq - freq).abs() < 100.0,
            "Peak at {} Hz, expected near {} Hz",
            peak_freq,
            freq
        );
    }

    #[test]
    fn test_spectrogram_dimensions() {
        let sample_rate = 16000.0;
        let duration = 0.5;
        let sound = Sound::create_tone(500.0, duration, sample_rate, 0.5, 0.0);

        let time_step = 0.01;
        let window_length = 0.025;
        let max_freq = 5000.0;

        let spectrogram = sound.to_spectrogram(
            time_step,
            max_freq,
            window_length,
            0.0,
            WindowShape::Gaussian,
        );

        // Check that max frequency is respected
        assert!(spectrogram.freq_max() <= max_freq + spectrogram.freq_step());

        // Check time step
        assert!((spectrogram.time_step() - time_step).abs() < 0.001);
    }

    #[test]
    fn test_spectrogram_silence() {
        let sound = Sound::create_silence(0.3, 44100.0);
        let spectrogram = sound.to_spectrogram(0.01, 5000.0, 0.03, 0.0, WindowShape::Gaussian);

        // Total energy should be very low for silence
        let total = spectrogram.total_energy();
        assert!(total < 1e-10, "Silence should have near-zero energy, got {}", total);
    }

    #[test]
    fn test_spectrogram_query() {
        let sound = Sound::create_tone(800.0, 0.3, 44100.0, 0.5, 0.0);
        let spectrogram = sound.to_spectrogram(0.01, 5000.0, 0.03, 0.0, WindowShape::Gaussian);

        // Query at specific time and frequency
        let t = 0.15;
        let f = 800.0;

        let power = spectrogram.get_power_at(t, f);
        assert!(power.is_some());

        let power_db = spectrogram.get_power_db_at(t, f);
        assert!(power_db.is_some());

        // Power at the tone frequency should be higher than at distant frequency
        let power_at_tone = spectrogram.get_power_at(t, 800.0).unwrap();
        let power_away = spectrogram.get_power_at(t, 3000.0).unwrap();
        assert!(power_at_tone > power_away * 10.0);
    }
}
