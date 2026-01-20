//! Spectrogram (time-frequency representation)
//!
//! This module computes spectrograms from audio signals using the Short-Time
//! Fourier Transform (STFT), matching Praat's Sound_to_Spectrogram_e exactly.
//!
//! Key differences from naive STFT:
//! - For Gaussian window, physical width = 2 × effective width
//! - FFT bins are combined ("binned") into spectrogram frequency bins
//! - Specific normalization: power / (windowssq × binWidth_samples)

use crate::utils::Fft;
use crate::window::WindowShape;
use crate::Sound;

/// Spectrogram representation (time-frequency matrix)
#[derive(Debug, Clone)]
pub struct Spectrogram {
    /// Power values in Pa²/Hz, stored as [frequency_bin][time_frame]
    data: Vec<Vec<f64>>,
    /// Time of first frame center
    start_time: f64,
    /// Time step between frames
    time_step: f64,
    /// Minimum frequency (always 0)
    freq_min: f64,
    /// Maximum frequency
    freq_max: f64,
    /// Frequency step (Hz per bin)
    freq_step: f64,
    /// Number of time frames
    num_frames: usize,
    /// Number of frequency bins
    num_freq_bins: usize,
    /// First frequency bin center
    first_freq: f64,
}

impl Spectrogram {
    /// Compute spectrogram from a Sound (matches Praat's Sound_to_Spectrogram exactly)
    ///
    /// # Arguments
    /// * `sound` - Input audio signal
    /// * `effective_analysis_width` - Effective window duration (seconds)
    /// * `max_frequency` - Maximum frequency to include (Hz, 0 for Nyquist)
    /// * `time_step` - Time between frames (0 for automatic)
    /// * `frequency_step` - Frequency resolution (Hz, 0 for automatic)
    /// * `window_shape` - Window function (typically Gaussian)
    ///
    /// # Algorithm (from Praat's Sound_and_Spectrogram.cpp)
    /// For Gaussian window:
    /// - physicalAnalysisWidth = 2 × effectiveAnalysisWidth
    /// - effectiveTimeWidth = effectiveAnalysisWidth / sqrt(π)
    /// - effectiveFreqWidth = 1 / effectiveTimeWidth
    pub fn from_sound(
        sound: &Sound,
        effective_analysis_width: f64,
        max_frequency: f64,
        minimum_time_step: f64,
        minimum_freq_step: f64,
        window_shape: WindowShape,
    ) -> Self {
        let dx = 1.0 / sound.sample_rate();
        let nx = sound.num_samples();
        let nyquist = 0.5 / dx;

        // For Gaussian, physical width is 2× effective width
        let physical_analysis_width = match window_shape {
            WindowShape::Gaussian => 2.0 * effective_analysis_width,
            _ => effective_analysis_width,
        };

        // Effective widths for determining oversampling
        let effective_time_width = effective_analysis_width / std::f64::consts::PI.sqrt();
        let effective_freq_width = 1.0 / effective_time_width;

        // Maximum oversampling factors
        let maximum_time_oversampling = 8.0;
        let maximum_freq_oversampling = 8.0;

        // Determine actual time step
        let minimum_time_step_2 = effective_time_width / maximum_time_oversampling;
        let time_step = minimum_time_step.max(minimum_time_step_2);

        // Determine actual frequency step
        let minimum_freq_step_2 = effective_freq_width / maximum_freq_oversampling;
        let mut freq_step = minimum_freq_step.max(minimum_freq_step_2);

        // Physical duration
        let physical_duration = dx * nx as f64;

        // Validate
        let max_frequency = if max_frequency <= 0.0 || max_frequency > nyquist {
            nyquist
        } else {
            max_frequency
        };

        if physical_analysis_width > physical_duration {
            return Self::empty(sound.start_time(), time_step, 0.0, max_frequency, freq_step);
        }

        // Compute window samples
        let approximate_nsamp_window = (physical_analysis_width / dx).floor() as i64;
        let halfnsamp_window = approximate_nsamp_window / 2 - 1;
        if halfnsamp_window < 1 {
            return Self::empty(sound.start_time(), time_step, 0.0, max_frequency, freq_step);
        }
        let nsamp_window = (halfnsamp_window * 2) as usize;

        // Compute number of time frames
        let number_of_times = 1 + ((physical_duration - physical_analysis_width) / time_step).floor() as usize;
        let t1 = sound.start_time() + 0.5 * dx
            + 0.5 * ((nx as f64 - 1.0) * dx - (number_of_times as f64 - 1.0) * time_step);

        // Compute FFT size and frequency sampling
        let mut number_of_freqs = (max_frequency / freq_step).floor() as usize;
        if number_of_freqs < 1 {
            return Self::empty(sound.start_time(), time_step, 0.0, max_frequency, freq_step);
        }

        // FFT size must be power of 2, large enough for window and frequency requirements
        let mut nsamp_fft = 1usize;
        while nsamp_fft < nsamp_window || nsamp_fft < (2.0 * number_of_freqs as f64 * (nyquist / max_frequency)) as usize {
            nsamp_fft *= 2;
        }
        let half_nsamp_fft = nsamp_fft / 2;

        // Compute actual frequency step (binning)
        let bin_width_samples = (freq_step * dx * nsamp_fft as f64).floor().max(1.0) as usize;
        let bin_width_hertz = 1.0 / (dx * nsamp_fft as f64);
        freq_step = bin_width_samples as f64 * bin_width_hertz;
        number_of_freqs = (max_frequency / freq_step).floor() as usize;
        if number_of_freqs < 1 {
            return Self::empty(sound.start_time(), time_step, 0.0, max_frequency, freq_step);
        }

        // First frequency (center of first bin)
        let first_freq = 0.5 * (freq_step - bin_width_hertz);

        // Generate window
        let samples = sound.samples();
        let n_samples_per_window_f = physical_analysis_width / dx;
        let mut window = vec![0.0; nsamp_window];
        let mut windowssq: f64 = 0.0;

        for i in 0..nsamp_window {
            let i_1based = (i + 1) as f64;
            let phase = i_1based / n_samples_per_window_f;

            let w = match window_shape {
                WindowShape::Gaussian => {
                    let imid = 0.5 * (nsamp_window as f64 + 1.0);
                    let edge = (-12.0_f64).exp();
                    let phase_gauss = (i_1based - imid) / n_samples_per_window_f;
                    ((-48.0 * phase_gauss * phase_gauss).exp() - edge) / (1.0 - edge)
                }
                WindowShape::Hanning => 0.5 * (1.0 - (2.0 * std::f64::consts::PI * phase).cos()),
                WindowShape::Hamming => 0.54 - 0.46 * (2.0 * std::f64::consts::PI * phase).cos(),
                WindowShape::Triangular => 1.0 - (2.0 * phase - 1.0).abs(),
                WindowShape::Parabolic => 1.0 - (2.0 * phase - 1.0).powi(2),
                WindowShape::Rectangular | WindowShape::Kaiser => 1.0,
            };
            window[i] = w;
            windowssq += w * w;
        }

        let one_by_bin_width = 1.0 / windowssq / bin_width_samples as f64;

        // Initialize data storage [freq_bin][time_frame]
        let mut data: Vec<Vec<f64>> = (0..number_of_freqs)
            .map(|_| vec![0.0; number_of_times])
            .collect();

        let mut fft = Fft::new();

        for iframe in 0..number_of_times {
            let t = t1 + iframe as f64 * time_step;

            // Find sample indices
            // leftSample = floor((t - x1) / dx), rightSample = leftSample + 1 (1-based)
            let x1 = sound.start_time() + 0.5 * dx;
            let left_sample_1based = ((t - x1) / dx).floor() as i64 + 1;
            let right_sample_1based = left_sample_1based + 1;

            let start_sample_1based = right_sample_1based - halfnsamp_window;
            let end_sample_1based = left_sample_1based + halfnsamp_window;

            // Convert to 0-based and clamp
            let start_sample = (start_sample_1based - 1).max(0) as usize;
            let end_sample = ((end_sample_1based - 1) as usize).min(nx - 1);

            // Prepare FFT input
            let mut fft_data = vec![0.0; nsamp_fft];

            // Apply window and copy to FFT buffer
            for (j, isamp) in (start_sample..=end_sample).enumerate() {
                if j < nsamp_window && isamp < nx {
                    fft_data[j] = samples[isamp] * window[j];
                }
            }

            // Compute FFT
            let spectrum = fft.real_fft(&fft_data, nsamp_fft);

            // Compute power spectrum
            let mut power_spectrum = vec![0.0; half_nsamp_fft + 1];
            power_spectrum[0] = spectrum[0].re * spectrum[0].re; // DC
            for i in 1..half_nsamp_fft {
                power_spectrum[i] = spectrum[i].re * spectrum[i].re + spectrum[i].im * spectrum[i].im;
            }
            power_spectrum[half_nsamp_fft] = spectrum[half_nsamp_fft].re * spectrum[half_nsamp_fft].re; // Nyquist

            // Binning: combine multiple FFT bins into spectrogram bins
            for iband in 0..number_of_freqs {
                let lower_sample = iband * bin_width_samples;
                let higher_sample = lower_sample + bin_width_samples;

                // Sum power in this band
                let mut power = 0.0;
                for k in lower_sample..higher_sample.min(power_spectrum.len()) {
                    power += power_spectrum[k];
                }

                data[iband][iframe] = power * one_by_bin_width;
            }
        }

        Self {
            data,
            start_time: t1,
            time_step,
            freq_min: 0.0,
            freq_max: max_frequency,
            freq_step,
            num_frames: number_of_times,
            num_freq_bins: number_of_freqs,
            first_freq,
        }
    }

    /// Create an empty spectrogram
    fn empty(start_time: f64, time_step: f64, freq_min: f64, freq_max: f64, freq_step: f64) -> Self {
        Self {
            data: Vec::new(),
            start_time,
            time_step,
            freq_min,
            freq_max,
            freq_step,
            num_frames: 0,
            num_freq_bins: 0,
            first_freq: 0.0,
        }
    }

    /// Compute spectrogram from multiple Sound channels, averaging power spectra
    ///
    /// This matches Praat's behavior for multi-channel audio: each channel is processed
    /// separately, then the power spectra are averaged. This is different from averaging
    /// the samples first (which our standard `from_sound` does when given mono-converted audio).
    ///
    /// For stereo: `(|FFT(ch1)|² + |FFT(ch2)|²) / 2` (correct for Praat compatibility)
    /// vs: `|FFT((ch1+ch2)/2)|²` (what you get with sample averaging)
    ///
    /// # Arguments
    /// * `sounds` - Slice of Sound objects (one per channel, must all have same sample rate/duration)
    /// * Other arguments same as `from_sound`
    ///
    /// # Panics
    /// Panics if sounds is empty or if sounds have different sample rates.
    pub fn from_sounds_averaged(
        sounds: &[Sound],
        effective_analysis_width: f64,
        max_frequency: f64,
        minimum_time_step: f64,
        minimum_freq_step: f64,
        window_shape: WindowShape,
    ) -> Self {
        if sounds.is_empty() {
            panic!("from_sounds_averaged requires at least one Sound");
        }

        if sounds.len() == 1 {
            // Single channel - just use the normal method
            return Self::from_sound(
                &sounds[0],
                effective_analysis_width,
                max_frequency,
                minimum_time_step,
                minimum_freq_step,
                window_shape,
            );
        }

        // Verify all sounds have same sample rate
        let sample_rate = sounds[0].sample_rate();
        for sound in sounds.iter().skip(1) {
            assert!(
                (sound.sample_rate() - sample_rate).abs() < 0.01,
                "All sounds must have the same sample rate"
            );
        }

        // Compute spectrogram for first channel
        let mut result = Self::from_sound(
            &sounds[0],
            effective_analysis_width,
            max_frequency,
            minimum_time_step,
            minimum_freq_step,
            window_shape,
        );

        // Add power from remaining channels
        for sound in sounds.iter().skip(1) {
            let channel_spec = Self::from_sound(
                sound,
                effective_analysis_width,
                max_frequency,
                minimum_time_step,
                minimum_freq_step,
                window_shape,
            );

            // Add power values
            for (freq_bin, row) in result.data.iter_mut().enumerate() {
                if freq_bin < channel_spec.data.len() {
                    for (frame, power) in row.iter_mut().enumerate() {
                        if frame < channel_spec.data[freq_bin].len() {
                            *power += channel_spec.data[freq_bin][frame];
                        }
                    }
                }
            }
        }

        // Divide by number of channels to get average
        let num_channels = sounds.len() as f64;
        for row in result.data.iter_mut() {
            for power in row.iter_mut() {
                *power /= num_channels;
            }
        }

        result
    }

    /// Get the power at a specific time and frequency
    pub fn get_power_at(&self, time: f64, frequency: f64) -> Option<f64> {
        if self.data.is_empty() || self.num_frames == 0 {
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
        let freq_bin = ((frequency - self.first_freq) / self.freq_step).round() as usize;
        if freq_bin >= self.num_freq_bins {
            return None;
        }

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

    /// Get the entire power matrix [frequency_bin][time_frame]
    pub fn values(&self) -> &Vec<Vec<f64>> {
        &self.data
    }

    /// Get power values for a specific frequency bin across all times
    pub fn get_frequency_slice(&self, frequency: f64) -> Option<&[f64]> {
        if frequency < self.freq_min || frequency > self.freq_max {
            return None;
        }
        let freq_bin = ((frequency - self.first_freq) / self.freq_step).round() as usize;
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
        self.first_freq + bin as f64 * self.freq_step
    }

    /// Get the frame index nearest to a time
    pub fn get_frame_from_time(&self, time: f64) -> usize {
        let pos = (time - self.start_time) / self.time_step;
        (pos.round() as usize).min(self.num_frames.saturating_sub(1))
    }

    /// Get the frequency bin nearest to a frequency
    pub fn get_bin_from_frequency(&self, frequency: f64) -> usize {
        let bin = ((frequency - self.first_freq) / self.freq_step).round() as usize;
        bin.min(self.num_freq_bins.saturating_sub(1))
    }

    pub fn num_frames(&self) -> usize { self.num_frames }
    pub fn num_freq_bins(&self) -> usize { self.num_freq_bins }
    pub fn time_step(&self) -> f64 { self.time_step }
    pub fn freq_step(&self) -> f64 { self.freq_step }
    pub fn freq_min(&self) -> f64 { self.freq_min }
    pub fn freq_max(&self) -> f64 { self.freq_max }
    pub fn start_time(&self) -> f64 { self.start_time }

    pub fn end_time(&self) -> f64 {
        if self.num_frames == 0 {
            self.start_time
        } else {
            self.start_time + (self.num_frames - 1) as f64 * self.time_step
        }
    }

    pub fn duration(&self) -> f64 { self.end_time() - self.start_time }

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
    /// Compute spectrogram from this sound (matches Praat's To Spectrogram command)
    ///
    /// # Arguments
    /// * `time_step` - Time between frames (0.0 for automatic)
    /// * `max_frequency` - Maximum frequency to show (0.0 for Nyquist)
    /// * `window_length` - Effective analysis window duration
    /// * `frequency_step` - Frequency resolution (0.0 for automatic)
    /// * `window_shape` - Window function (typically Gaussian)
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
            window_length,
            max_frequency,
            time_step,
            frequency_step,
            window_shape,
        )
    }
}

/// Compute spectrogram from multiple channel sounds with Praat-compatible power averaging
///
/// This function matches Praat's behavior for multi-channel audio files.
/// Praat computes the spectrogram for each channel separately, then averages
/// the power spectra.
///
/// # Arguments
/// * `sounds` - Slice of Sound objects (one per channel)
/// * Other arguments same as `Sound::to_spectrogram`
///
/// # Example
/// ```no_run
/// use praatfan_core::{Sound, spectrogram_from_channels, WindowShape};
///
/// // Load stereo file keeping channels separate
/// let channels = Sound::from_file_channels("stereo.wav").unwrap();
///
/// // Compute spectrogram with Praat-compatible averaging
/// let spec = spectrogram_from_channels(&channels, 0.002, 5000.0, 0.005, 20.0, WindowShape::Gaussian);
/// ```
pub fn spectrogram_from_channels(
    sounds: &[Sound],
    time_step: f64,
    max_frequency: f64,
    window_length: f64,
    frequency_step: f64,
    window_shape: WindowShape,
) -> Spectrogram {
    Spectrogram::from_sounds_averaged(
        sounds,
        window_length,
        max_frequency,
        time_step,
        frequency_step,
        window_shape,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spectrogram_basic() {
        let sound = Sound::create_tone(440.0, 0.5, 44100.0, 0.5, 0.0);
        let spectrogram = sound.to_spectrogram(0.005, 5000.0, 0.03, 0.0, WindowShape::Gaussian);

        assert!(spectrogram.num_frames() > 0);
        assert!(spectrogram.num_freq_bins() > 0);
    }

    #[test]
    fn test_spectrogram_pure_tone_peak() {
        let freq = 1000.0;
        let sound = Sound::create_tone(freq, 0.3, 44100.0, 0.5, 0.0);
        let spectrogram = sound.to_spectrogram(0.01, 5000.0, 0.03, 0.0, WindowShape::Gaussian);

        // Find peak at middle time
        let middle_frame = spectrogram.num_frames() / 2;
        let middle_time = spectrogram.get_time_from_frame(middle_frame);

        if let Some(time_slice) = spectrogram.get_time_slice(middle_time) {
            let mut max_power = 0.0;
            let mut max_bin = 0;
            for (bin, &power) in time_slice.iter().enumerate() {
                if power > max_power {
                    max_power = power;
                    max_bin = bin;
                }
            }

            let peak_freq = spectrogram.get_frequency_from_bin(max_bin);
            assert!(
                (peak_freq - freq).abs() < 200.0,
                "Peak at {} Hz, expected near {} Hz",
                peak_freq,
                freq
            );
        }
    }

    #[test]
    fn test_spectrogram_silence() {
        let sound = Sound::create_silence(0.3, 44100.0);
        let spectrogram = sound.to_spectrogram(0.01, 5000.0, 0.03, 0.0, WindowShape::Gaussian);

        let total = spectrogram.total_energy();
        assert!(total < 1e-10, "Silence should have near-zero energy, got {}", total);
    }
}
