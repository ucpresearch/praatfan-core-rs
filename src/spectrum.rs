//! Single-frame spectrum analysis
//!
//! This module computes FFT-based spectral analysis of audio signals,
//! providing magnitude/power spectra and spectral moments (center of gravity,
//! standard deviation, skewness, kurtosis).

use crate::utils::CompSum;
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

    /// Compute spectrum from a Sound (matches Praat's Sound_to_Spectrum)
    ///
    /// # Arguments
    /// * `sound` - Input audio signal
    /// * `fast` - If true, use FFT size that is power of 2 (faster but zero-padded)
    ///
    /// # Returns
    /// Spectrum containing frequency bins from 0 Hz to Nyquist frequency.
    /// The spectrum values are scaled by dx (sample period) to give spectral density in Pa/Hz.
    pub fn from_sound(sound: &Sound, fast: bool) -> Self {
        let samples = sound.samples();
        let n = samples.len();

        if n == 0 {
            return Self::new(Vec::new(), Vec::new(), 0.0, 0.0);
        }

        // Praat (Sound_to_Spectrum): fast -> next power of two (at least 2),
        // zero-padded; FFT via FFTPACK (NUMfft_forward), scaled by dx.
        let fft_size = if fast { n.next_power_of_two().max(2) } else { n };
        let mut data = vec![0.0f64; fft_size];
        data[..n].copy_from_slice(samples);
        let mut plan = crate::utils::fft_fftpack::RealFftPlan::new(fft_size);
        plan.forward(&mut data);

        // drftf1 layout: [DC, Re1, Im1, Re2, Im2, ..., (Re_{n/2} if n even)]
        let n_bins = fft_size / 2 + 1;
        let dx = 1.0 / sound.sample_rate(); // sample period
        let mut real = vec![0.0f64; n_bins];
        let mut imag = vec![0.0f64; n_bins];
        real[0] = data[0] * dx;
        for k in 1..n_bins - 1 {
            real[k] = data[2 * k - 1] * dx;
            imag[k] = data[2 * k] * dx;
        }
        if n_bins > 1 {
            if fft_size % 2 == 1 {
                real[n_bins - 1] = data[fft_size - 2] * dx;
                imag[n_bins - 1] = data[fft_size - 1] * dx;
            } else {
                real[n_bins - 1] = data[fft_size - 1] * dx;
            }
        }

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
            Some(crate::math::atan2(self.imag[bin], self.real[bin]))
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

    /// Get band energy between two frequencies (matches Praat's Spectrum_getBandEnergy)
    ///
    /// # Arguments
    /// * `freq_min` - Minimum frequency (Hz), 0 for start
    /// * `freq_max` - Maximum frequency (Hz), 0 for Nyquist
    ///
    /// # Returns
    /// Total energy in the frequency band in Pa² s.
    /// Uses factor of 2 for one-sided spectrum (positive + negative frequencies).
    pub fn get_band_energy(&self, freq_min: f64, freq_max: f64) -> f64 {
        if self.real.is_empty() {
            return 0.0;
        }

        // Handle 0 = auto for max frequency
        let freq_min = freq_min.max(0.0);
        let freq_max = if freq_max <= 0.0 { self.max_frequency } else { freq_max.min(self.max_frequency) };

        // Praat uses Sampled_getIntegral which properly handles bin widths at boundaries.
        // For simplicity, we'll use a simpler approach that's still accurate:
        // Each bin's contribution is: 2 * (re² + im²) * bin_width
        // where bin_width is df for interior bins, but may be less for boundary bins

        let mut energy = 0.0;

        for bin in 0..self.real.len() {
            let freq = self.get_frequency_from_bin(bin);

            // Calculate bin boundaries
            let bin_start = freq - 0.5 * self.df;
            let bin_end = freq + 0.5 * self.df;

            // Clip to frequency range and domain
            let effective_start = bin_start.max(freq_min).max(0.0);
            let effective_end = bin_end.min(freq_max).min(self.max_frequency);

            if effective_end > effective_start {
                let bin_width = effective_end - effective_start;
                let r = self.real[bin];
                let i = self.imag[bin];
                // Energy density = 2 * (re² + im²) for one-sided spectrum
                let energy_density = 2.0 * (r * r + i * i);
                energy += energy_density * bin_width;
            }
        }

        energy
    }

    /// Get the band energy in dB
    pub fn get_band_energy_db(&self, freq_min: f64, freq_max: f64) -> f64 {
        let energy = self.get_band_energy(freq_min, freq_max);
        if energy > 0.0 {
            10.0 * crate::math::log10(energy)
        } else {
            f64::NEG_INFINITY
        }
    }

    /// Bin energies as Praat weights them: `re^2 + im^2`, raised to
    /// `power / 2` unless that is 1.
    fn weighted_energy(&self, bin: usize, halfpower: f64) -> f64 {
        let (re, im) = (self.real[bin], self.imag[bin]);
        let energy = re * re + im * im;
        if halfpower != 1.0 { crate::math::pow(energy, halfpower) } else { energy }
    }

    /// Spectral centre of gravity: Praat's `Spectrum_getCentreOfGravity`,
    /// `sum(f |X|^power) / sum(|X|^power)` with long-double sums.
    /// NaN (Praat: undefined) for an all-zero spectrum.
    pub fn get_center_of_gravity(&self, power: f64) -> f64 {
        let halfpower = 0.5 * power;
        let (mut sum_e, mut sum_fe) = (CompSum::default(), CompSum::default());
        for bin in 0..self.real.len() {
            let energy = self.weighted_energy(bin, halfpower);
            let f = bin as f64 * self.df;
            sum_e.add(energy);
            sum_fe.add(f * energy);
        }
        let se = sum_e.value();
        if se == 0.0 { f64::NAN } else { sum_fe.value() / se }
    }

    /// Praat's `Spectrum_getCentralMoment`.
    pub fn get_central_moment(&self, moment: f64, power: f64) -> f64 {
        let fmean = self.get_center_of_gravity(power);
        if fmean.is_nan() {
            return f64::NAN;
        }
        let halfpower = 0.5 * power;
        let (mut sum_e, mut sum_fe) = (CompSum::default(), CompSum::default());
        for bin in 0..self.real.len() {
            let energy = self.weighted_energy(bin, halfpower);
            let f = bin as f64 * self.df;
            sum_e.add(energy);
            sum_fe.add(crate::math::pow(f - fmean, moment) * energy);
        }
        sum_fe.value() / sum_e.value()
    }

    /// Spectral standard deviation: `sqrt(m2)` (Praat).
    pub fn get_standard_deviation(&self, power: f64) -> f64 {
        libm::sqrt(self.get_central_moment(2.0, power))
    }

    /// Spectral skewness: `m3 / m2^1.5` (Praat); NaN if undefined.
    pub fn get_skewness(&self, power: f64) -> f64 {
        let m2 = self.get_central_moment(2.0, power);
        let m3 = self.get_central_moment(3.0, power);
        if m2.is_nan() || m3.is_nan() || m2 == 0.0 {
            return f64::NAN;
        }
        m3 / (m2 * libm::sqrt(m2))
    }

    /// Spectral (excess) kurtosis: `m4 / m2^2 - 3` (Praat); NaN if undefined.
    pub fn get_kurtosis(&self, power: f64) -> f64 {
        let m2 = self.get_central_moment(2.0, power);
        let m4 = self.get_central_moment(4.0, power);
        if m2.is_nan() || m4.is_nan() || m2 == 0.0 {
            return f64::NAN;
        }
        m4 / (m2 * m2) - 3.0
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
