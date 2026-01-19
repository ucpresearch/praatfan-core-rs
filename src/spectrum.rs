//! Single-frame spectrum analysis
//!
//! This module computes FFT-based spectral analysis of audio signals,
//! providing magnitude/power spectra and spectral moments (center of gravity,
//! standard deviation, skewness, kurtosis).

use crate::utils::Fft;
use crate::Sound;

/// Single-frame spectrum representation
#[derive(Debug, Clone)]
pub struct Spectrum {
    /// Real parts of the spectrum (for each frequency bin)
    real: Vec<f64>,
    /// Imaginary parts of the spectrum
    imag: Vec<f64>,
    /// Frequency resolution (Hz per bin)
    df: f64,
    /// Maximum frequency (Nyquist)
    max_frequency: f64,
}

impl Spectrum {
    /// Create a spectrum from real and imaginary parts
    pub fn new(real: Vec<f64>, imag: Vec<f64>, df: f64, max_frequency: f64) -> Self {
        Self {
            real,
            imag,
            df,
            max_frequency,
        }
    }

    /// Compute spectrum from a Sound
    ///
    /// # Arguments
    /// * `sound` - Input audio signal
    /// * `fast` - If true, use FFT size that is power of 2 (faster but zero-padded)
    ///
    /// # Returns
    /// Spectrum containing frequency bins from 0 Hz to Nyquist frequency
    pub fn from_sound(sound: &Sound, fast: bool) -> Self {
        let samples = sound.samples();
        let n = samples.len();

        if n == 0 {
            return Self::new(Vec::new(), Vec::new(), 0.0, 0.0);
        }

        // Determine FFT size
        let fft_size = if fast {
            n.next_power_of_two()
        } else {
            n
        };

        let mut fft = Fft::new();
        let spectrum = fft.real_fft(samples, fft_size);

        // Extract positive frequencies (0 to Nyquist)
        let n_bins = fft_size / 2 + 1;
        let real: Vec<f64> = spectrum[..n_bins].iter().map(|c| c.re).collect();
        let imag: Vec<f64> = spectrum[..n_bins].iter().map(|c| c.im).collect();

        let df = sound.sample_rate() / fft_size as f64;
        let max_frequency = sound.sample_rate() / 2.0;

        Self {
            real,
            imag,
            df,
            max_frequency,
        }
    }

    /// Get the number of frequency bins
    pub fn num_bins(&self) -> usize {
        self.real.len()
    }

    /// Get the frequency of a specific bin
    pub fn get_frequency_from_bin(&self, bin: usize) -> f64 {
        bin as f64 * self.df
    }

    /// Get the bin index for a frequency (rounded to nearest)
    pub fn get_bin_from_frequency(&self, frequency: f64) -> usize {
        let bin = (frequency / self.df).round() as usize;
        bin.min(self.real.len().saturating_sub(1))
    }

    /// Get the frequency resolution (Hz per bin)
    pub fn df(&self) -> f64 {
        self.df
    }

    /// Get the maximum frequency (Nyquist)
    pub fn max_frequency(&self) -> f64 {
        self.max_frequency
    }

    /// Get the real part of the spectrum at a bin
    pub fn get_real(&self, bin: usize) -> Option<f64> {
        self.real.get(bin).copied()
    }

    /// Get the imaginary part of the spectrum at a bin
    pub fn get_imag(&self, bin: usize) -> Option<f64> {
        self.imag.get(bin).copied()
    }

    /// Get the magnitude at a frequency bin
    pub fn get_magnitude(&self, bin: usize) -> Option<f64> {
        if bin < self.real.len() {
            let r = self.real[bin];
            let i = self.imag[bin];
            Some((r * r + i * i).sqrt())
        } else {
            None
        }
    }

    /// Get the power (magnitude squared) at a frequency bin
    pub fn get_power(&self, bin: usize) -> Option<f64> {
        if bin < self.real.len() {
            let r = self.real[bin];
            let i = self.imag[bin];
            Some(r * r + i * i)
        } else {
            None
        }
    }

    /// Get the phase at a frequency bin (in radians)
    pub fn get_phase(&self, bin: usize) -> Option<f64> {
        if bin < self.real.len() {
            Some(self.imag[bin].atan2(self.real[bin]))
        } else {
            None
        }
    }

    /// Get the power spectral density (power per Hz)
    pub fn power_spectral_density(&self) -> Vec<f64> {
        self.real
            .iter()
            .zip(self.imag.iter())
            .map(|(&r, &i)| {
                let power = r * r + i * i;
                // Normalize by frequency resolution
                power / self.df
            })
            .collect()
    }

    /// Get band energy between two frequencies
    ///
    /// # Arguments
    /// * `freq_min` - Minimum frequency (Hz)
    /// * `freq_max` - Maximum frequency (Hz)
    ///
    /// # Returns
    /// Total energy in the frequency band (sum of power values × df)
    pub fn get_band_energy(&self, freq_min: f64, freq_max: f64) -> f64 {
        let bin_min = self.get_bin_from_frequency(freq_min.max(0.0));
        let bin_max = self.get_bin_from_frequency(freq_max.min(self.max_frequency));

        if bin_min >= self.real.len() || bin_max < bin_min {
            return 0.0;
        }

        let mut energy = 0.0;
        for bin in bin_min..=bin_max.min(self.real.len() - 1) {
            let r = self.real[bin];
            let i = self.imag[bin];
            energy += r * r + i * i;
        }

        energy * self.df
    }

    /// Get the band energy in dB
    pub fn get_band_energy_db(&self, freq_min: f64, freq_max: f64) -> f64 {
        let energy = self.get_band_energy(freq_min, freq_max);
        if energy > 0.0 {
            10.0 * energy.log10()
        } else {
            f64::NEG_INFINITY
        }
    }

    /// Compute spectral center of gravity (centroid)
    ///
    /// # Arguments
    /// * `power` - The power to use for weighting (typically 1.0 or 2.0)
    ///
    /// The center of gravity is the weighted mean frequency:
    /// CoG = Σ(f × |X(f)|^power) / Σ(|X(f)|^power)
    pub fn get_center_of_gravity(&self, power: f64) -> f64 {
        let mut weighted_sum = 0.0;
        let mut weight_sum = 0.0;

        for bin in 0..self.real.len() {
            let freq = self.get_frequency_from_bin(bin);
            let r = self.real[bin];
            let i = self.imag[bin];
            let magnitude = (r * r + i * i).sqrt();
            let weight = magnitude.powf(power);

            weighted_sum += freq * weight;
            weight_sum += weight;
        }

        if weight_sum > 0.0 {
            weighted_sum / weight_sum
        } else {
            0.0
        }
    }

    /// Compute spectral standard deviation
    ///
    /// # Arguments
    /// * `power` - The power to use for weighting
    ///
    /// The standard deviation measures the spread around the center of gravity
    pub fn get_standard_deviation(&self, power: f64) -> f64 {
        let cog = self.get_center_of_gravity(power);

        let mut weighted_var_sum = 0.0;
        let mut weight_sum = 0.0;

        for bin in 0..self.real.len() {
            let freq = self.get_frequency_from_bin(bin);
            let r = self.real[bin];
            let i = self.imag[bin];
            let magnitude = (r * r + i * i).sqrt();
            let weight = magnitude.powf(power);

            let deviation = freq - cog;
            weighted_var_sum += deviation * deviation * weight;
            weight_sum += weight;
        }

        if weight_sum > 0.0 {
            (weighted_var_sum / weight_sum).sqrt()
        } else {
            0.0
        }
    }

    /// Compute spectral skewness
    ///
    /// Skewness measures the asymmetry of the spectrum around the center of gravity.
    /// Positive values indicate more energy at higher frequencies.
    pub fn get_skewness(&self, power: f64) -> f64 {
        let cog = self.get_center_of_gravity(power);
        let std_dev = self.get_standard_deviation(power);

        if std_dev == 0.0 {
            return 0.0;
        }

        let mut weighted_sum = 0.0;
        let mut weight_sum = 0.0;

        for bin in 0..self.real.len() {
            let freq = self.get_frequency_from_bin(bin);
            let r = self.real[bin];
            let i = self.imag[bin];
            let magnitude = (r * r + i * i).sqrt();
            let weight = magnitude.powf(power);

            let z = (freq - cog) / std_dev;
            weighted_sum += z.powi(3) * weight;
            weight_sum += weight;
        }

        if weight_sum > 0.0 {
            weighted_sum / weight_sum
        } else {
            0.0
        }
    }

    /// Compute spectral kurtosis
    ///
    /// Kurtosis measures the "peakedness" of the spectrum.
    /// Higher values indicate a more peaked distribution.
    pub fn get_kurtosis(&self, power: f64) -> f64 {
        let cog = self.get_center_of_gravity(power);
        let std_dev = self.get_standard_deviation(power);

        if std_dev == 0.0 {
            return 0.0;
        }

        let mut weighted_sum = 0.0;
        let mut weight_sum = 0.0;

        for bin in 0..self.real.len() {
            let freq = self.get_frequency_from_bin(bin);
            let r = self.real[bin];
            let i = self.imag[bin];
            let magnitude = (r * r + i * i).sqrt();
            let weight = magnitude.powf(power);

            let z = (freq - cog) / std_dev;
            weighted_sum += z.powi(4) * weight;
            weight_sum += weight;
        }

        if weight_sum > 0.0 {
            // Subtract 3 to get excess kurtosis (normal distribution has kurtosis = 3)
            weighted_sum / weight_sum - 3.0
        } else {
            0.0
        }
    }

    /// Get the total energy (integral of power spectrum)
    pub fn get_total_energy(&self) -> f64 {
        self.get_band_energy(0.0, self.max_frequency)
    }
}

// Add to_spectrum method to Sound
impl Sound {
    /// Compute the spectrum of this sound
    ///
    /// # Arguments
    /// * `fast` - If true, use power-of-2 FFT size for faster computation
    ///
    /// # Returns
    /// Single-frame spectrum
    pub fn to_spectrum(&self, fast: bool) -> Spectrum {
        Spectrum::from_sound(self, fast)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_spectrum_pure_tone() {
        // Create a pure tone at 1000 Hz
        let sample_rate = 44100.0;
        let freq = 1000.0;
        let duration = 0.1;
        let sound = Sound::create_tone(freq, duration, sample_rate, 1.0, 0.0);

        let spectrum = sound.to_spectrum(true);

        // Find the bin with maximum magnitude
        let mut max_bin = 0;
        let mut max_mag = 0.0;
        for bin in 0..spectrum.num_bins() {
            let mag = spectrum.get_magnitude(bin).unwrap();
            if mag > max_mag {
                max_mag = mag;
                max_bin = bin;
            }
        }

        // The peak should be near 1000 Hz
        let peak_freq = spectrum.get_frequency_from_bin(max_bin);
        assert!((peak_freq - freq).abs() < 50.0, "Peak at {} Hz, expected near {} Hz", peak_freq, freq);
    }

    #[test]
    fn test_spectrum_center_of_gravity() {
        // A pure tone should have center of gravity near its frequency
        let freq = 2000.0;
        let sound = Sound::create_tone(freq, 0.1, 44100.0, 1.0, 0.0);
        let spectrum = sound.to_spectrum(true);

        let cog = spectrum.get_center_of_gravity(2.0);

        // CoG should be close to the tone frequency
        // (not exact due to spectral leakage and windowing)
        assert!((cog - freq).abs() < 200.0, "CoG = {} Hz, expected near {} Hz", cog, freq);
    }

    #[test]
    fn test_spectrum_band_energy() {
        // Create a tone at 1000 Hz
        let freq = 1000.0;
        let sound = Sound::create_tone(freq, 0.1, 44100.0, 1.0, 0.0);
        let spectrum = sound.to_spectrum(true);

        // Most energy should be in a band around 1000 Hz
        let energy_around_tone = spectrum.get_band_energy(800.0, 1200.0);
        let energy_away = spectrum.get_band_energy(3000.0, 4000.0);

        assert!(energy_around_tone > energy_away * 10.0,
            "Energy near tone: {}, energy away: {}", energy_around_tone, energy_away);
    }

    #[test]
    fn test_spectrum_standard_deviation() {
        // A pure tone should have low spectral standard deviation
        let sound = Sound::create_tone(1000.0, 0.1, 44100.0, 1.0, 0.0);
        let spectrum = sound.to_spectrum(true);

        let std_dev = spectrum.get_standard_deviation(2.0);

        // Standard deviation should be relatively small for a pure tone
        assert!(std_dev < 500.0, "Std dev = {} Hz, should be small for pure tone", std_dev);
    }

    #[test]
    fn test_spectrum_df() {
        let sample_rate = 44100.0;
        let duration = 0.1;
        let sound = Sound::create_silence(duration, sample_rate);
        let spectrum = sound.to_spectrum(true);

        // df should equal sample_rate / fft_size
        let expected_num_samples = (duration * sample_rate) as usize;
        let fft_size = expected_num_samples.next_power_of_two();
        let expected_df = sample_rate / fft_size as f64;

        assert_relative_eq!(spectrum.df(), expected_df, epsilon = 0.01);
    }
}
