//! Window functions for spectral analysis
//!
//! This module provides window functions matching Praat's implementations.
//! Window functions are used to reduce spectral leakage in FFT-based analysis.

use std::f64::consts::PI;

/// Window shapes available for analysis
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum WindowShape {
    /// Rectangular window (no windowing)
    Rectangular,
    /// Triangular (Bartlett) window
    Triangular,
    /// Parabolic window
    Parabolic,
    /// Hanning window (raised cosine)
    #[default]
    Hanning,
    /// Hamming window
    Hamming,
    /// Gaussian window with configurable standard deviation
    /// The parameter is stored separately when needed
    Gaussian,
    /// Kaiser window (requires beta parameter)
    Kaiser,
}

impl WindowShape {
    /// Compute the window value at a normalized position
    ///
    /// # Arguments
    /// * `position` - Position in the window, normalized to [-0.5, 0.5]
    ///                where 0 is the center
    /// * `parameter` - Optional parameter (e.g., sigma for Gaussian, beta for Kaiser)
    ///
    /// # Returns
    /// The window amplitude at the given position (0.0 to 1.0)
    pub fn value_at(self, position: f64, parameter: Option<f64>) -> f64 {
        // Position is normalized: -0.5 = left edge, 0 = center, 0.5 = right edge
        if position.abs() > 0.5 {
            return 0.0;
        }

        match self {
            WindowShape::Rectangular => 1.0,

            WindowShape::Triangular => 1.0 - 2.0 * position.abs(),

            WindowShape::Parabolic => 1.0 - 4.0 * position * position,

            WindowShape::Hanning => {
                // Praat's Hanning: 0.5 + 0.5 * cos(2 * pi * position)
                0.5 + 0.5 * (2.0 * PI * position).cos()
            }

            WindowShape::Hamming => {
                // Praat's Hamming: 0.54 + 0.46 * cos(2 * pi * position)
                0.54 + 0.46 * (2.0 * PI * position).cos()
            }

            WindowShape::Gaussian => {
                // Gaussian window: exp(-0.5 * (position / sigma)^2)
                // Default sigma for Praat's Gaussian spectrogram is ~0.4
                let sigma = parameter.unwrap_or(0.4);
                (-0.5 * (position / sigma).powi(2)).exp()
            }

            WindowShape::Kaiser => {
                // Kaiser window: I0(beta * sqrt(1 - (2*position)^2)) / I0(beta)
                // where I0 is the modified Bessel function of order 0
                let beta = parameter.unwrap_or(2.0 * PI);
                let x = 2.0 * position;
                let arg = (1.0 - x * x).max(0.0).sqrt();
                bessel_i0(beta * arg) / bessel_i0(beta)
            }
        }
    }

    /// Generate a complete window of the given size
    ///
    /// # Arguments
    /// * `size` - Number of samples in the window
    /// * `parameter` - Optional parameter for Gaussian/Kaiser windows
    ///
    /// # Returns
    /// Vector of window values
    pub fn generate(self, size: usize, parameter: Option<f64>) -> Vec<f64> {
        if size == 0 {
            return Vec::new();
        }

        (0..size)
            .map(|i| {
                // Map sample index to normalized position [-0.5, 0.5]
                let position = (i as f64 + 0.5) / size as f64 - 0.5;
                self.value_at(position, parameter)
            })
            .collect()
    }

    /// Generate a symmetric window (for filter design)
    ///
    /// This creates a window where the first and last values are equal,
    /// suitable for FIR filter design.
    pub fn generate_symmetric(self, size: usize, parameter: Option<f64>) -> Vec<f64> {
        if size == 0 {
            return Vec::new();
        }
        if size == 1 {
            return vec![1.0];
        }

        (0..size)
            .map(|i| {
                let position = i as f64 / (size - 1) as f64 - 0.5;
                self.value_at(position, parameter)
            })
            .collect()
    }
}

/// Modified Bessel function of order 0 (for Kaiser window)
///
/// Uses polynomial approximation from Abramowitz & Stegun
fn bessel_i0(x: f64) -> f64 {
    let ax = x.abs();

    if ax < 3.75 {
        // Polynomial approximation for small arguments
        let y = (x / 3.75).powi(2);
        1.0 + y
            * (3.5156229
                + y * (3.0899424
                    + y * (1.2067492
                        + y * (0.2659732 + y * (0.0360768 + y * 0.0045813)))))
    } else {
        // Asymptotic expansion for large arguments
        let y = 3.75 / ax;
        (ax.exp() / ax.sqrt())
            * (0.39894228
                + y * (0.01328592
                    + y * (0.00225319
                        + y * (-0.00157565
                            + y * (0.00916281
                                + y * (-0.02057706
                                    + y * (0.02635537
                                        + y * (-0.01647633 + y * 0.00392377))))))))
    }
}

/// Generate Praat's specific Gaussian window for formant analysis
///
/// This matches Praat's exact formula from Sound_to_Formant.cpp:
/// window[i] = (exp(-48.0 * (i - imid)² / (nsamp_window + 1)²) - edge) / (1.0 - edge)
/// where edge = exp(-12.0)
///
/// This window is different from the standard Gaussian window used elsewhere.
pub fn praat_formant_window(size: usize) -> Vec<f64> {
    if size == 0 {
        return Vec::new();
    }

    let edge = (-12.0_f64).exp();
    let imid = (size as f64 - 1.0) / 2.0;
    let denom = (size + 1) as f64;

    (0..size)
        .map(|i| {
            let diff = i as f64 - imid;
            let gaussian = (-48.0 * diff * diff / (denom * denom)).exp();
            (gaussian - edge) / (1.0 - edge)
        })
        .collect()
}

/// Calculate the equivalent noise bandwidth of a window
///
/// This is the ratio of the window's noise bandwidth to that of a rectangular window.
/// Useful for power spectrum normalization.
pub fn equivalent_noise_bandwidth(window: &[f64]) -> f64 {
    if window.is_empty() {
        return 1.0;
    }

    let sum: f64 = window.iter().sum();
    let sum_sq: f64 = window.iter().map(|x| x * x).sum();

    if sum == 0.0 {
        return 1.0;
    }

    let n = window.len() as f64;
    n * sum_sq / (sum * sum)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_rectangular_window() {
        let window = WindowShape::Rectangular.generate(10, None);
        assert_eq!(window.len(), 10);
        for &v in &window {
            assert_relative_eq!(v, 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_hanning_window_properties() {
        let window = WindowShape::Hanning.generate(100, None);

        // Hanning should be symmetric
        for i in 0..50 {
            assert_relative_eq!(window[i], window[99 - i], epsilon = 1e-10);
        }

        // Center should be maximum (close to 1.0)
        let center = window[49];
        assert!(center > 0.99);

        // Edges should be small
        assert!(window[0] < 0.02);
        assert!(window[99] < 0.02);
    }

    #[test]
    fn test_gaussian_window() {
        let window = WindowShape::Gaussian.generate(100, Some(0.4));

        // Should be symmetric
        for i in 0..50 {
            assert_relative_eq!(window[i], window[99 - i], epsilon = 1e-10);
        }

        // Center should be close to 1.0
        let center = window[49];
        assert!(center > 0.99);
    }

    #[test]
    fn test_window_normalization() {
        // Rectangular window should have ENBW = 1.0
        let rect = WindowShape::Rectangular.generate(100, None);
        assert_relative_eq!(equivalent_noise_bandwidth(&rect), 1.0, epsilon = 1e-10);

        // Hanning window should have ENBW ≈ 1.5
        let hanning = WindowShape::Hanning.generate(1000, None);
        let enbw = equivalent_noise_bandwidth(&hanning);
        assert_relative_eq!(enbw, 1.5, epsilon = 0.01);
    }

    #[test]
    fn test_bessel_i0() {
        // Test against known values
        assert_relative_eq!(bessel_i0(0.0), 1.0, epsilon = 1e-6);
        assert_relative_eq!(bessel_i0(1.0), 1.2660658777520082, epsilon = 1e-6);
        assert_relative_eq!(bessel_i0(2.0), 2.2795853023360673, epsilon = 1e-5);
    }

    #[test]
    fn test_kaiser_window() {
        let window = WindowShape::Kaiser.generate(100, Some(2.0 * PI));

        // Should be symmetric
        for i in 0..50 {
            assert_relative_eq!(window[i], window[99 - i], epsilon = 1e-10);
        }

        // All values should be positive
        for &v in &window {
            assert!(v > 0.0);
        }
    }
}
