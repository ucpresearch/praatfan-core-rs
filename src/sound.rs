//! Sound type for audio data representation and manipulation
//!
//! The Sound type is the fundamental data structure representing audio samples.
//! It supports loading from various audio formats (WAV, MP3, FLAC, OGG),
//! extracting segments, and applying filters.

use std::fs::File;
use std::path::Path;

use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::DecoderOptions;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

use crate::window::WindowShape;
use crate::{PraatError, Result};

/// Audio samples with associated sample rate and timing information
#[derive(Debug, Clone)]
pub struct Sound {
    /// Audio samples (mono, normalized to [-1, 1] range)
    samples: Vec<f64>,
    /// Sample rate in Hz
    sample_rate: f64,
    /// Start time of the first sample (usually 0.0)
    start_time: f64,
}

impl Sound {
    /// Create a Sound from raw samples
    ///
    /// # Arguments
    /// * `samples` - Audio samples (will be cloned)
    /// * `sample_rate` - Sample rate in Hz
    ///
    /// # Example
    /// ```
    /// use praat_core::Sound;
    ///
    /// let samples = vec![0.0, 0.5, 1.0, 0.5, 0.0, -0.5, -1.0, -0.5];
    /// let sound = Sound::from_samples(&samples, 44100.0);
    /// assert_eq!(sound.sample_rate(), 44100.0);
    /// ```
    pub fn from_samples(samples: &[f64], sample_rate: f64) -> Self {
        Self {
            samples: samples.to_vec(),
            sample_rate,
            start_time: 0.0,
        }
    }

    /// Create a Sound from owned samples (avoids cloning)
    pub fn from_samples_owned(samples: Vec<f64>, sample_rate: f64) -> Self {
        Self {
            samples,
            sample_rate,
            start_time: 0.0,
        }
    }

    /// Load a Sound from an audio file (supports WAV, MP3, FLAC, OGG)
    ///
    /// Supports mono and multi-channel files. Multi-channel files are converted
    /// to mono by averaging channels. Samples are normalized to [-1, 1] range.
    ///
    /// # Arguments
    /// * `path` - Path to the audio file
    ///
    /// # Errors
    /// Returns an error if the file cannot be read or format is not supported.
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();

        // Try symphonia first (supports more formats)
        match Self::from_file_symphonia(path) {
            Ok(sound) => return Ok(sound),
            Err(_) => {
                // Fall back to hound for WAV files
                if let Some(ext) = path.extension() {
                    if ext.to_string_lossy().to_lowercase() == "wav" {
                        return Self::from_file_wav(path);
                    }
                }
            }
        }

        // If symphonia failed and it's not a WAV, return the symphonia error
        Self::from_file_symphonia(path)
    }

    /// Load a Sound from any supported format using symphonia
    ///
    /// Supports WAV, MP3, FLAC, OGG Vorbis, and other formats.
    fn from_file_symphonia<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();
        let file = File::open(path).map_err(PraatError::Io)?;
        let mss = MediaSourceStream::new(Box::new(file), Default::default());

        // Create a hint based on file extension
        let mut hint = Hint::new();
        if let Some(ext) = path.extension() {
            hint.with_extension(&ext.to_string_lossy());
        }

        // Probe the format
        let format_opts = FormatOptions::default();
        let metadata_opts = MetadataOptions::default();
        let probed = symphonia::default::get_probe()
            .format(&hint, mss, &format_opts, &metadata_opts)
            .map_err(|e| PraatError::Analysis(format!("Failed to probe audio format: {}", e)))?;

        let mut format = probed.format;

        // Get the default track
        let track = format
            .default_track()
            .ok_or_else(|| PraatError::Analysis("No audio track found".to_string()))?;

        let track_id = track.id;
        let sample_rate = track
            .codec_params
            .sample_rate
            .ok_or_else(|| PraatError::Analysis("Unknown sample rate".to_string()))? as f64;
        let channels = track
            .codec_params
            .channels
            .map(|c| c.count())
            .unwrap_or(1);

        // Create decoder
        let decoder_opts = DecoderOptions::default();
        let mut decoder = symphonia::default::get_codecs()
            .make(&track.codec_params, &decoder_opts)
            .map_err(|e| PraatError::Analysis(format!("Failed to create decoder: {}", e)))?;

        // Decode all packets
        let mut all_samples: Vec<f64> = Vec::new();

        loop {
            let packet = match format.next_packet() {
                Ok(packet) => packet,
                Err(symphonia::core::errors::Error::IoError(ref e))
                    if e.kind() == std::io::ErrorKind::UnexpectedEof =>
                {
                    break;
                }
                Err(e) => {
                    return Err(PraatError::Analysis(format!("Error reading packet: {}", e)));
                }
            };

            // Skip packets from other tracks
            if packet.track_id() != track_id {
                continue;
            }

            // Decode the packet
            let decoded = match decoder.decode(&packet) {
                Ok(decoded) => decoded,
                Err(symphonia::core::errors::Error::DecodeError(_)) => continue,
                Err(e) => {
                    return Err(PraatError::Analysis(format!("Decode error: {}", e)));
                }
            };

            // Convert to f64 samples
            let spec = *decoded.spec();
            let num_frames = decoded.frames();

            let mut sample_buf = SampleBuffer::<f32>::new(num_frames as u64, spec);
            sample_buf.copy_interleaved_ref(decoded);
            let samples_f32 = sample_buf.samples();

            // Convert to mono and f64
            if channels == 1 {
                all_samples.extend(samples_f32.iter().map(|&s| s as f64));
            } else {
                for chunk in samples_f32.chunks(channels) {
                    let sum: f64 = chunk.iter().map(|&s| s as f64).sum();
                    all_samples.push(sum / channels as f64);
                }
            }
        }

        Ok(Self {
            samples: all_samples,
            sample_rate,
            start_time: 0.0,
        })
    }

    /// Load a Sound from a WAV file using hound
    fn from_file_wav<P: AsRef<Path>>(path: P) -> Result<Self> {
        let reader = hound::WavReader::open(path)?;
        let spec = reader.spec();
        let sample_rate = spec.sample_rate as f64;
        let channels = spec.channels as usize;

        let samples: Vec<f64> = match spec.sample_format {
            hound::SampleFormat::Int => {
                let max_value = (1_i64 << (spec.bits_per_sample - 1)) as f64;
                let int_samples: Vec<i32> = reader.into_samples::<i32>().map(|s| s.unwrap()).collect();

                // Convert to mono and normalize
                if channels == 1 {
                    int_samples.iter().map(|&s| s as f64 / max_value).collect()
                } else {
                    int_samples
                        .chunks(channels)
                        .map(|chunk| {
                            let sum: f64 = chunk.iter().map(|&s| s as f64).sum();
                            sum / (channels as f64 * max_value)
                        })
                        .collect()
                }
            }
            hound::SampleFormat::Float => {
                let float_samples: Vec<f32> =
                    reader.into_samples::<f32>().map(|s| s.unwrap()).collect();

                if channels == 1 {
                    float_samples.iter().map(|&s| s as f64).collect()
                } else {
                    float_samples
                        .chunks(channels)
                        .map(|chunk| {
                            let sum: f64 = chunk.iter().map(|&s| s as f64).sum();
                            sum / channels as f64
                        })
                        .collect()
                }
            }
        };

        Ok(Self {
            samples,
            sample_rate,
            start_time: 0.0,
        })
    }

    /// Get the sample rate in Hz
    pub fn sample_rate(&self) -> f64 {
        self.sample_rate
    }

    /// Get a reference to the audio samples
    pub fn samples(&self) -> &[f64] {
        &self.samples
    }

    /// Get the number of samples
    pub fn num_samples(&self) -> usize {
        self.samples.len()
    }

    /// Get the total duration in seconds
    pub fn duration(&self) -> f64 {
        self.samples.len() as f64 / self.sample_rate
    }

    /// Get the start time (time of first sample)
    pub fn start_time(&self) -> f64 {
        self.start_time
    }

    /// Get the end time (time just after last sample)
    pub fn end_time(&self) -> f64 {
        self.start_time + self.duration()
    }

    /// Get the time corresponding to a sample index
    ///
    /// Following Praat's convention, the time is at the center of the sample.
    pub fn index_to_time(&self, index: usize) -> f64 {
        self.start_time + (index as f64 + 0.5) / self.sample_rate
    }

    /// Get the sample index corresponding to a time
    ///
    /// Returns the nearest sample index. May return an index outside
    /// the valid range if the time is outside the sound's duration.
    pub fn time_to_index(&self, time: f64) -> isize {
        ((time - self.start_time) * self.sample_rate - 0.5).round() as isize
    }

    /// Get the sample index corresponding to a time, clamped to valid range
    pub fn time_to_index_clamped(&self, time: f64) -> usize {
        let index = self.time_to_index(time);
        if index < 0 {
            0
        } else if index >= self.samples.len() as isize {
            self.samples.len().saturating_sub(1)
        } else {
            index as usize
        }
    }

    /// Get the sample value at a given time using linear interpolation
    ///
    /// Returns None if the time is outside the sound's duration.
    pub fn get_value_at_time(&self, time: f64) -> Option<f64> {
        if time < self.start_time || time > self.end_time() {
            return None;
        }

        // Find the fractional sample position
        let position = (time - self.start_time) * self.sample_rate - 0.5;

        if position <= 0.0 {
            return Some(self.samples[0]);
        }

        let index = position.floor() as usize;
        if index >= self.samples.len() - 1 {
            return Some(*self.samples.last().unwrap());
        }

        // Linear interpolation
        let frac = position - index as f64;
        let v0 = self.samples[index];
        let v1 = self.samples[index + 1];
        Some(v0 + frac * (v1 - v0))
    }

    /// Extract a portion of the sound
    ///
    /// # Arguments
    /// * `start_time` - Start time of the extraction
    /// * `end_time` - End time of the extraction
    /// * `window_shape` - Window to apply (Rectangular for no windowing)
    /// * `relative_width` - Width of the window relative to the extracted duration
    /// * `preserve_times` - If true, preserve original time stamps
    ///
    /// This matches Praat's "Extract part" command.
    pub fn extract_part(
        &self,
        start_time: f64,
        end_time: f64,
        window_shape: WindowShape,
        relative_width: f64,
        preserve_times: bool,
    ) -> Result<Sound> {
        if start_time >= end_time {
            return Err(PraatError::InvalidParameter(
                "start_time must be less than end_time".to_string(),
            ));
        }

        let duration = end_time - start_time;
        let n_samples = (duration * self.sample_rate).round() as usize;

        if n_samples == 0 {
            return Err(PraatError::InvalidParameter(
                "Extraction would result in zero samples".to_string(),
            ));
        }

        // Calculate sample indices
        let first_sample = self.time_to_index(start_time).max(0) as usize;
        let last_sample = (self.time_to_index(end_time).max(0) as usize).min(self.samples.len());

        // Extract and optionally window the samples
        let mut extracted: Vec<f64> = self.samples[first_sample..last_sample].to_vec();

        // Apply window if not rectangular
        if window_shape != WindowShape::Rectangular {
            let window = window_shape.generate(extracted.len(), Some(relative_width));
            for (sample, &w) in extracted.iter_mut().zip(window.iter()) {
                *sample *= w;
            }
        }

        Ok(Sound {
            samples: extracted,
            sample_rate: self.sample_rate,
            start_time: if preserve_times { start_time } else { 0.0 },
        })
    }

    /// Apply pre-emphasis filter
    ///
    /// This is a simple first-order high-pass filter that boosts high frequencies.
    /// Commonly used before formant analysis to flatten the spectral slope.
    ///
    /// # Arguments
    /// * `from_frequency` - The frequency (Hz) above which to boost (typically 50 Hz)
    ///
    /// The filter is: y[n] = x[n] - alpha * x[n-1]
    /// where alpha = exp(-2 * pi * from_frequency / sample_rate)
    pub fn pre_emphasis(&self, from_frequency: f64) -> Sound {
        if self.samples.is_empty() {
            return self.clone();
        }

        let alpha = (-2.0 * std::f64::consts::PI * from_frequency / self.sample_rate).exp();

        let mut filtered = Vec::with_capacity(self.samples.len());
        filtered.push(self.samples[0]);

        for i in 1..self.samples.len() {
            filtered.push(self.samples[i] - alpha * self.samples[i - 1]);
        }

        Sound {
            samples: filtered,
            sample_rate: self.sample_rate,
            start_time: self.start_time,
        }
    }

    /// Apply de-emphasis filter (inverse of pre-emphasis)
    ///
    /// # Arguments
    /// * `from_frequency` - The frequency used in the original pre-emphasis
    pub fn de_emphasis(&self, from_frequency: f64) -> Sound {
        if self.samples.is_empty() {
            return self.clone();
        }

        let alpha = (-2.0 * std::f64::consts::PI * from_frequency / self.sample_rate).exp();

        let mut filtered = Vec::with_capacity(self.samples.len());
        filtered.push(self.samples[0]);

        for i in 1..self.samples.len() {
            filtered.push(self.samples[i] + alpha * filtered[i - 1]);
        }

        Sound {
            samples: filtered,
            sample_rate: self.sample_rate,
            start_time: self.start_time,
        }
    }

    /// Multiply samples by a constant
    pub fn scale(&self, factor: f64) -> Sound {
        Sound {
            samples: self.samples.iter().map(|&s| s * factor).collect(),
            sample_rate: self.sample_rate,
            start_time: self.start_time,
        }
    }

    /// Add two sounds together
    ///
    /// The sounds must have the same sample rate. The result has the duration
    /// of the longer sound.
    pub fn add(&self, other: &Sound) -> Result<Sound> {
        if (self.sample_rate - other.sample_rate).abs() > 0.01 {
            return Err(PraatError::InvalidParameter(
                "Sounds must have the same sample rate".to_string(),
            ));
        }

        let len = self.samples.len().max(other.samples.len());
        let mut result = vec![0.0; len];

        for (i, &s) in self.samples.iter().enumerate() {
            result[i] += s;
        }
        for (i, &s) in other.samples.iter().enumerate() {
            result[i] += s;
        }

        Ok(Sound {
            samples: result,
            sample_rate: self.sample_rate,
            start_time: self.start_time.min(other.start_time),
        })
    }

    /// Create a pure tone (sine wave)
    ///
    /// # Arguments
    /// * `frequency` - Frequency in Hz
    /// * `duration` - Duration in seconds
    /// * `sample_rate` - Sample rate in Hz
    /// * `amplitude` - Peak amplitude (0.0 to 1.0)
    /// * `phase` - Initial phase in radians
    pub fn create_tone(
        frequency: f64,
        duration: f64,
        sample_rate: f64,
        amplitude: f64,
        phase: f64,
    ) -> Sound {
        let n_samples = (duration * sample_rate).round() as usize;
        let omega = 2.0 * std::f64::consts::PI * frequency / sample_rate;

        let samples: Vec<f64> = (0..n_samples)
            .map(|i| amplitude * (omega * i as f64 + phase).sin())
            .collect();

        Sound {
            samples,
            sample_rate,
            start_time: 0.0,
        }
    }

    /// Create silence
    pub fn create_silence(duration: f64, sample_rate: f64) -> Sound {
        let n_samples = (duration * sample_rate).round() as usize;
        Sound {
            samples: vec![0.0; n_samples],
            sample_rate,
            start_time: 0.0,
        }
    }

    /// Get the root-mean-square (RMS) amplitude
    pub fn rms(&self) -> f64 {
        if self.samples.is_empty() {
            return 0.0;
        }
        let sum_sq: f64 = self.samples.iter().map(|&s| s * s).sum();
        (sum_sq / self.samples.len() as f64).sqrt()
    }

    /// Get the peak amplitude
    pub fn peak(&self) -> f64 {
        self.samples
            .iter()
            .map(|&s| s.abs())
            .fold(0.0, f64::max)
    }

    /// Get the minimum and maximum sample values
    pub fn min_max(&self) -> (f64, f64) {
        if self.samples.is_empty() {
            return (0.0, 0.0);
        }
        let min = self.samples.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = self.samples.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        (min, max)
    }

    /// Resample the sound to a new sample rate using sinc interpolation
    ///
    /// This method uses windowed sinc interpolation (Lanczos-like) for
    /// high-quality resampling. It's used internally for formant analysis
    /// where Praat resamples to 2 × max_formant before LPC.
    ///
    /// # Arguments
    /// * `new_sample_rate` - Target sample rate in Hz
    ///
    /// # Returns
    /// A new Sound at the target sample rate
    pub fn resample(&self, new_sample_rate: f64) -> Sound {
        if self.samples.is_empty() || (new_sample_rate - self.sample_rate).abs() < 0.01 {
            return self.clone();
        }

        let ratio = new_sample_rate / self.sample_rate;
        let new_num_samples = ((self.samples.len() as f64) * ratio).round() as usize;

        if new_num_samples == 0 {
            return Sound {
                samples: Vec::new(),
                sample_rate: new_sample_rate,
                start_time: self.start_time,
            };
        }

        // For downsampling, apply FFT-based anti-aliasing filter first (like Praat)
        let filtered_samples = if ratio < 1.0 {
            self.fft_lowpass_filter(ratio)
        } else {
            self.samples.clone()
        };

        let mut new_samples = vec![0.0; new_num_samples];

        // Sinc interpolation precision (Praat default is 50)
        let precision = 50;

        for i in 0..new_num_samples {
            // Time position in original signal coordinates
            let index = i as f64 / ratio;
            new_samples[i] = sinc_interpolate(&filtered_samples, index, precision);
        }

        Sound {
            samples: new_samples,
            sample_rate: new_sample_rate,
            start_time: self.start_time,
        }
    }

    /// Apply FFT-based low-pass filter for anti-aliasing
    fn fft_lowpass_filter(&self, cutoff_ratio: f64) -> Vec<f64> {
        use crate::utils::fft::Fft;
        use num_complex::Complex;

        let n = self.samples.len();
        // Pad to power of 2 with turnaround
        let anti_turn_around = 1000;
        let nfft = (n + 2 * anti_turn_around).next_power_of_two();

        // Create padded signal (mirror padding at edges like Praat)
        let mut padded = vec![0.0; nfft];
        for i in 0..anti_turn_around.min(n) {
            padded[anti_turn_around - 1 - i] = self.samples[i];
        }
        for i in 0..n {
            padded[anti_turn_around + i] = self.samples[i];
        }
        for i in 0..anti_turn_around.min(n) {
            padded[anti_turn_around + n + i] = self.samples[n - 1 - i];
        }

        // Forward FFT
        let mut fft = Fft::new();
        let mut spectrum = fft.real_fft(&padded, nfft);

        // Zero out frequencies above cutoff (low-pass filter)
        let cutoff_bin = ((cutoff_ratio * nfft as f64) / 2.0).ceil() as usize;
        for i in cutoff_bin..spectrum.len() {
            spectrum[i] = Complex::new(0.0, 0.0);
        }
        // Also zero the symmetric part for proper inverse FFT
        for i in 1..cutoff_bin.min(spectrum.len()) {
            spectrum[nfft - i] = Complex::new(0.0, 0.0);
        }

        // Inverse FFT
        let time_domain = fft.inverse_fft(&spectrum);

        // Extract the original portion
        let mut result = Vec::with_capacity(n);
        for i in 0..n {
            result.push(time_domain[anti_turn_around + i].re);
        }
        result
    }
}

/// Sinc interpolation matching Praat's NUM_interpolate_sinc
/// Based on Praat's sinc interpolation with Hann (raised cosine) window
fn sinc_interpolate(samples: &[f64], index: f64, max_depth: usize) -> f64 {
    let n = samples.len();
    if n == 0 {
        return 0.0;
    }

    // Get the integer and fractional parts
    let midleft = index.floor() as isize;
    let fraction = index - midleft as f64;

    // Handle exact integer index
    if fraction < 1e-10 {
        if midleft >= 0 && (midleft as usize) < n {
            return samples[midleft as usize];
        }
        return 0.0;
    }
    if fraction > 1.0 - 1e-10 {
        let idx = midleft + 1;
        if idx >= 0 && (idx as usize) < n {
            return samples[idx as usize];
        }
        return 0.0;
    }

    // Praat's sinc interpolation formula
    let mut result = 0.0;
    let max_depth = max_depth as isize;

    for i in -max_depth..=max_depth {
        let sample_idx = midleft + 1 + i;
        if sample_idx < 0 || sample_idx >= n as isize {
            continue;
        }

        // Distance from the interpolation point
        let d = fraction + i as f64;

        // Sinc function
        let sinc = if d.abs() < 1e-12 {
            1.0
        } else {
            let pid = std::f64::consts::PI * d;
            pid.sin() / pid
        };

        // Raised cosine (Hann) window
        // window_phase goes from 0 to 1 across the window
        let window_phase = (0.5 + 0.5 * d / (max_depth as f64 + 1.0)).clamp(0.0, 1.0);
        let window = 0.5 - 0.5 * (2.0 * std::f64::consts::PI * window_phase).cos();

        result += samples[sample_idx as usize] * sinc * window;
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_from_samples() {
        let samples = vec![0.0, 0.5, 1.0, 0.5, 0.0];
        let sound = Sound::from_samples(&samples, 44100.0);

        assert_eq!(sound.sample_rate(), 44100.0);
        assert_eq!(sound.num_samples(), 5);
        assert_relative_eq!(sound.duration(), 5.0 / 44100.0, epsilon = 1e-10);
    }

    #[test]
    fn test_pure_tone() {
        let freq = 440.0;
        let sample_rate = 44100.0;
        let duration = 0.01; // 10 ms
        let sound = Sound::create_tone(freq, duration, sample_rate, 1.0, 0.0);

        // Check that the first sample is 0 (sin(0) = 0)
        assert_relative_eq!(sound.samples()[0], 0.0, epsilon = 1e-10);

        // Check that we have the expected number of samples
        let expected_samples = (duration * sample_rate).round() as usize;
        assert_eq!(sound.num_samples(), expected_samples);
    }

    #[test]
    fn test_time_index_conversion() {
        let sound = Sound::from_samples(&vec![0.0; 1000], 44100.0);

        // Time at center of first sample
        let t0 = sound.index_to_time(0);
        assert_relative_eq!(t0, 0.5 / 44100.0, epsilon = 1e-10);

        // Round-trip
        let idx = sound.time_to_index(t0);
        assert_eq!(idx, 0);
    }

    #[test]
    fn test_pre_emphasis() {
        // Create a simple test signal
        let samples: Vec<f64> = (0..100).map(|i| (i as f64 * 0.1).sin()).collect();
        let sound = Sound::from_samples(&samples, 1000.0);

        let emphasized = sound.pre_emphasis(50.0);

        // Pre-emphasis should change the signal
        assert_ne!(emphasized.samples(), sound.samples());

        // De-emphasis should approximately recover the original
        let recovered = emphasized.de_emphasis(50.0);
        for i in 1..samples.len() {
            assert_relative_eq!(recovered.samples()[i], samples[i], epsilon = 1e-10);
        }
    }

    #[test]
    fn test_extract_part() {
        let sound = Sound::create_tone(440.0, 1.0, 44100.0, 1.0, 0.0);

        // Extract middle 100ms
        let extracted = sound
            .extract_part(0.45, 0.55, WindowShape::Rectangular, 1.0, false)
            .unwrap();

        assert_relative_eq!(extracted.duration(), 0.1, epsilon = 1e-3);
        assert_eq!(extracted.start_time(), 0.0);
    }

    #[test]
    fn test_rms() {
        // For a sine wave, RMS should be peak / sqrt(2)
        let amplitude = 0.8;
        let sound = Sound::create_tone(440.0, 1.0, 44100.0, amplitude, 0.0);

        let rms = sound.rms();
        let expected_rms = amplitude / 2.0_f64.sqrt();
        assert_relative_eq!(rms, expected_rms, epsilon = 0.01);
    }
}
