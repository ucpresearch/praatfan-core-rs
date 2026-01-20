//! FFT wrapper for spectral analysis
//!
//! This module provides a convenient wrapper around rustfft for use in
//! acoustic analysis algorithms.

use num_complex::Complex;
use rustfft::FftPlanner;

/// FFT direction
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FftDirection {
    Forward,
    Inverse,
}

/// FFT processor with cached plans
pub struct Fft {
    planner: FftPlanner<f64>,
}

impl Fft {
    /// Create a new FFT processor
    pub fn new() -> Self {
        Self {
            planner: FftPlanner::new(),
        }
    }

    /// Compute FFT of real-valued input
    ///
    /// # Arguments
    /// * `input` - Real-valued input samples
    /// * `output_size` - Size of the FFT (will be zero-padded if larger than input)
    ///
    /// # Returns
    /// Complex-valued FFT result of length `output_size`
    pub fn real_fft(&mut self, input: &[f64], output_size: usize) -> Vec<Complex<f64>> {
        let fft_size = output_size.max(input.len());
        let fft = self.planner.plan_fft_forward(fft_size);

        let mut buffer: Vec<Complex<f64>> = input
            .iter()
            .map(|&x| Complex::new(x, 0.0))
            .chain(std::iter::repeat(Complex::new(0.0, 0.0)))
            .take(fft_size)
            .collect();

        fft.process(&mut buffer);
        buffer
    }

    /// Compute inverse FFT
    ///
    /// # Arguments
    /// * `input` - Complex-valued FFT coefficients
    ///
    /// # Returns
    /// Complex-valued result (take real part for real output)
    pub fn inverse_fft(&mut self, input: &[Complex<f64>]) -> Vec<Complex<f64>> {
        let fft_size = input.len();
        let fft = self.planner.plan_fft_inverse(fft_size);

        let mut buffer = input.to_vec();
        fft.process(&mut buffer);

        // Normalize by dividing by N
        let scale = 1.0 / fft_size as f64;
        for c in &mut buffer {
            *c *= scale;
        }

        buffer
    }

    /// Compute power spectrum (magnitude squared) of real input
    ///
    /// # Arguments
    /// * `input` - Real-valued input samples
    /// * `fft_size` - Size of the FFT
    ///
    /// # Returns
    /// Power spectrum values for frequencies 0 to Nyquist (fft_size/2 + 1 values)
    pub fn power_spectrum(&mut self, input: &[f64], fft_size: usize) -> Vec<f64> {
        let spectrum = self.real_fft(input, fft_size);

        // Return only the positive frequencies (DC to Nyquist)
        let n_freqs = fft_size / 2 + 1;
        spectrum[..n_freqs]
            .iter()
            .map(|c| c.norm_sqr())
            .collect()
    }

    /// Compute magnitude spectrum of real input
    ///
    /// # Arguments
    /// * `input` - Real-valued input samples
    /// * `fft_size` - Size of the FFT
    ///
    /// # Returns
    /// Magnitude spectrum values for frequencies 0 to Nyquist
    pub fn magnitude_spectrum(&mut self, input: &[f64], fft_size: usize) -> Vec<f64> {
        let spectrum = self.real_fft(input, fft_size);

        let n_freqs = fft_size / 2 + 1;
        spectrum[..n_freqs].iter().map(|c| c.norm()).collect()
    }

    /// Compute autocorrelation using FFT
    ///
    /// # Arguments
    /// * `input` - Real-valued input samples
    ///
    /// # Returns
    /// Autocorrelation values for lags 0 to n-1
    pub fn autocorrelation(&mut self, input: &[f64]) -> Vec<f64> {
        let n = input.len();
        if n == 0 {
            return Vec::new();
        }

        // Zero-pad to at least 2n for linear (non-circular) autocorrelation
        let fft_size = (2 * n).next_power_of_two();

        // Compute FFT of zero-padded input
        let spectrum = self.real_fft(input, fft_size);

        // Compute power spectrum
        let power: Vec<Complex<f64>> = spectrum.iter().map(|c| Complex::new(c.norm_sqr(), 0.0)).collect();

        // Inverse FFT to get autocorrelation
        let autocorr = self.inverse_fft(&power);

        // Return the first n values (real parts)
        autocorr[..n].iter().map(|c| c.re).collect()
    }

    /// Compute normalized autocorrelation (correlation coefficients)
    ///
    /// The result is normalized so that the lag-0 value is 1.0.
    pub fn normalized_autocorrelation(&mut self, input: &[f64]) -> Vec<f64> {
        let autocorr = self.autocorrelation(input);
        if autocorr.is_empty() || autocorr[0] == 0.0 {
            return autocorr;
        }

        let scale = 1.0 / autocorr[0];
        autocorr.iter().map(|&x| x * scale).collect()
    }

    /// Compute autocorrelation from multiple channels, summing power spectra
    ///
    /// This matches Praat's behavior for multi-channel pitch analysis:
    /// - Compute FFT for each channel
    /// - Sum the power spectra across channels
    /// - Inverse FFT to get combined autocorrelation
    ///
    /// # Arguments
    /// * `inputs` - Slice of input signals (one per channel)
    ///
    /// # Returns
    /// Autocorrelation values for lags 0 to n-1
    pub fn autocorrelation_multichannel(&mut self, inputs: &[&[f64]]) -> Vec<f64> {
        if inputs.is_empty() {
            return Vec::new();
        }

        // Find the maximum length
        let n = inputs.iter().map(|s| s.len()).max().unwrap_or(0);
        if n == 0 {
            return Vec::new();
        }

        // Zero-pad to at least 2n for linear (non-circular) autocorrelation
        let fft_size = (2 * n).next_power_of_two();

        // Sum power spectra from all channels
        let mut total_power = vec![Complex::new(0.0, 0.0); fft_size];

        for input in inputs {
            let spectrum = self.real_fft(input, fft_size);
            for (i, c) in spectrum.iter().enumerate() {
                total_power[i] += Complex::new(c.norm_sqr(), 0.0);
            }
        }

        // Inverse FFT to get autocorrelation
        let autocorr = self.inverse_fft(&total_power);

        // Return the first n values (real parts)
        autocorr[..n].iter().map(|c| c.re).collect()
    }
}

impl Default for Fft {
    fn default() -> Self {
        Self::new()
    }
}

/// Compute the next power of two greater than or equal to n
pub fn next_power_of_two(n: usize) -> usize {
    n.next_power_of_two()
}

/// Compute cross-correlation between two signals using FFT
pub fn cross_correlation(fft: &mut Fft, x: &[f64], y: &[f64]) -> Vec<f64> {
    let n = x.len().max(y.len());
    let fft_size = (2 * n).next_power_of_two();

    // FFT of both signals
    let spectrum_x = fft.real_fft(x, fft_size);
    let spectrum_y = fft.real_fft(y, fft_size);

    // Cross-spectrum: X * conj(Y)
    let cross: Vec<Complex<f64>> = spectrum_x
        .iter()
        .zip(spectrum_y.iter())
        .map(|(a, b)| a * b.conj())
        .collect();

    // Inverse FFT
    let result = fft.inverse_fft(&cross);

    // Return real parts
    result.iter().map(|c| c.re).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use std::f64::consts::PI;

    #[test]
    fn test_fft_dc() {
        let mut fft = Fft::new();

        // Constant signal should have all energy at DC
        let input = vec![1.0; 8];
        let spectrum = fft.real_fft(&input, 8);

        // DC component should be sum of input
        assert_relative_eq!(spectrum[0].re, 8.0, epsilon = 1e-10);
        assert_relative_eq!(spectrum[0].im, 0.0, epsilon = 1e-10);

        // Other components should be zero
        for i in 1..8 {
            assert_relative_eq!(spectrum[i].norm(), 0.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_fft_sine() {
        let mut fft = Fft::new();

        // Pure sine wave at bin 1
        let n = 16;
        let input: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * i as f64 / n as f64).sin())
            .collect();

        let spectrum = fft.real_fft(&input, n);

        // Energy should be at bin 1 (and n-1 for conjugate)
        assert!(spectrum[1].norm() > 1.0);
        assert_relative_eq!(spectrum[0].norm(), 0.0, epsilon = 1e-10);
        assert_relative_eq!(spectrum[2].norm(), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_inverse_fft() {
        let mut fft = Fft::new();

        let input: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let spectrum = fft.real_fft(&input, 8);
        let recovered = fft.inverse_fft(&spectrum);

        for (orig, rec) in input.iter().zip(recovered.iter()) {
            assert_relative_eq!(*orig, rec.re, epsilon = 1e-10);
            assert_relative_eq!(rec.im, 0.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_autocorrelation() {
        let mut fft = Fft::new();

        // Autocorrelation of a signal with itself
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let autocorr = fft.autocorrelation(&input);

        // Lag 0 should be sum of squares
        let expected_lag0: f64 = input.iter().map(|x| x * x).sum();
        assert_relative_eq!(autocorr[0], expected_lag0, epsilon = 1e-10);

        // Autocorrelation should be symmetric (but we only return half)
        assert_eq!(autocorr.len(), 4);
    }

    #[test]
    fn test_normalized_autocorrelation() {
        let mut fft = Fft::new();

        let input = vec![1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0];
        let autocorr = fft.normalized_autocorrelation(&input);

        // Lag 0 should be 1.0
        assert_relative_eq!(autocorr[0], 1.0, epsilon = 1e-10);

        // All other values should be <= 1.0 in absolute value
        for &val in &autocorr[1..] {
            assert!(val.abs() <= 1.0 + 1e-10);
        }
    }

    #[test]
    fn test_power_spectrum() {
        let mut fft = Fft::new();

        // Constant signal
        let input = vec![2.0; 8];
        let power = fft.power_spectrum(&input, 8);

        // DC power should be N^2 * amplitude^2
        assert_relative_eq!(power[0], 256.0, epsilon = 1e-10);

        // Power at other frequencies should be zero
        for &p in &power[1..] {
            assert_relative_eq!(p, 0.0, epsilon = 1e-10);
        }
    }
}
