//! Interpolation methods for querying sampled data
//!
//! This module provides interpolation functions matching Praat's conventions.
//! These are used when querying analysis results (pitch, formants, etc.) at
//! arbitrary time points.

/// Interpolation methods
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Interpolation {
    /// Nearest neighbor (no interpolation)
    Nearest,
    /// Linear interpolation between adjacent samples
    #[default]
    Linear,
    /// Cubic interpolation using 4 neighboring samples
    Cubic,
    /// Sinc interpolation (band-limited)
    Sinc70,
    /// Higher quality sinc interpolation
    Sinc700,
}

impl Interpolation {
    /// Interpolate a value from a uniformly-sampled array
    ///
    /// # Arguments
    /// * `samples` - The array of samples
    /// * `position` - The fractional position in the array (0.0 = first sample)
    ///
    /// # Returns
    /// The interpolated value, or None if the position is outside the valid range
    pub fn interpolate(self, samples: &[f64], position: f64) -> Option<f64> {
        if samples.is_empty() {
            return None;
        }

        let n = samples.len();
        let max_pos = (n - 1) as f64;

        // Clamp position to valid range
        if position < 0.0 || position > max_pos {
            return None;
        }

        match self {
            Interpolation::Nearest => {
                let idx = position.round() as usize;
                let idx = idx.min(n - 1);
                Some(samples[idx])
            }

            Interpolation::Linear => Some(linear_interpolate(samples, position)),

            Interpolation::Cubic => Some(cubic_interpolate(samples, position)),

            Interpolation::Sinc70 => Some(sinc_interpolate(samples, position, 70)),

            Interpolation::Sinc700 => Some(sinc_interpolate(samples, position, 700)),
        }
    }

    /// Interpolate a value, skipping undefined (NaN) values
    ///
    /// This is important for analysis results where some frames may be undefined
    /// (e.g., unvoiced frames in pitch analysis).
    pub fn interpolate_with_undefined(self, samples: &[f64], position: f64) -> Option<f64> {
        if samples.is_empty() {
            return None;
        }

        let n = samples.len();
        let max_pos = (n - 1) as f64;

        if position < 0.0 || position > max_pos {
            return None;
        }

        match self {
            Interpolation::Nearest => {
                let idx = position.round() as usize;
                let idx = idx.min(n - 1);
                let val = samples[idx];
                if val.is_nan() {
                    None
                } else {
                    Some(val)
                }
            }

            Interpolation::Linear => {
                linear_interpolate_with_undefined(samples, position)
            }

            Interpolation::Cubic => {
                // For cubic, fall back to linear if we can't get all 4 defined points
                cubic_interpolate_with_undefined(samples, position)
            }

            // Sinc interpolation with undefined values is complex; fall back to linear
            Interpolation::Sinc70 | Interpolation::Sinc700 => {
                linear_interpolate_with_undefined(samples, position)
            }
        }
    }
}

/// Linear interpolation between two values
pub fn lerp(a: f64, b: f64, t: f64) -> f64 {
    a + t * (b - a)
}

/// Linear interpolation in an array
fn linear_interpolate(samples: &[f64], position: f64) -> f64 {
    let n = samples.len();
    if n == 0 {
        return 0.0;
    }
    if n == 1 {
        return samples[0];
    }

    let idx = position.floor() as usize;
    if idx >= n - 1 {
        return samples[n - 1];
    }

    let frac = position - idx as f64;
    lerp(samples[idx], samples[idx + 1], frac)
}

/// Linear interpolation, skipping NaN values
fn linear_interpolate_with_undefined(samples: &[f64], position: f64) -> Option<f64> {
    let n = samples.len();
    if n == 0 {
        return None;
    }

    let idx = position.floor() as usize;
    let idx = idx.min(n - 1);
    let frac = position - idx as f64;

    // Get the two neighboring values
    let v0 = samples[idx];
    let v1 = if idx + 1 < n { samples[idx + 1] } else { v0 };

    // Handle undefined values
    match (v0.is_nan(), v1.is_nan()) {
        (true, true) => None,
        (true, false) => Some(v1),
        (false, true) => Some(v0),
        (false, false) => Some(lerp(v0, v1, frac)),
    }
}

/// Cubic interpolation using Catmull-Rom spline
///
/// This provides C1 continuity (smooth first derivative).
fn cubic_interpolate(samples: &[f64], position: f64) -> f64 {
    let n = samples.len();
    if n < 2 {
        return if n == 1 { samples[0] } else { 0.0 };
    }
    if n < 4 {
        return linear_interpolate(samples, position);
    }

    let idx = position.floor() as isize;
    let frac = position - idx as f64;

    // Get 4 surrounding points, clamping at boundaries
    let get_sample = |i: isize| -> f64 {
        let clamped = i.max(0).min(n as isize - 1) as usize;
        samples[clamped]
    };

    let y0 = get_sample(idx - 1);
    let y1 = get_sample(idx);
    let y2 = get_sample(idx + 1);
    let y3 = get_sample(idx + 2);

    // Catmull-Rom spline coefficients
    let t = frac;
    let t2 = t * t;
    let t3 = t2 * t;

    // Catmull-Rom formula
    let a0 = -0.5 * y0 + 1.5 * y1 - 1.5 * y2 + 0.5 * y3;
    let a1 = y0 - 2.5 * y1 + 2.0 * y2 - 0.5 * y3;
    let a2 = -0.5 * y0 + 0.5 * y2;
    let a3 = y1;

    a0 * t3 + a1 * t2 + a2 * t + a3
}

/// Cubic interpolation with undefined value handling
fn cubic_interpolate_with_undefined(samples: &[f64], position: f64) -> Option<f64> {
    let n = samples.len();
    if n < 2 {
        return if n == 1 && !samples[0].is_nan() {
            Some(samples[0])
        } else {
            None
        };
    }

    let idx = position.floor() as isize;

    // Check if we have 4 defined points for cubic interpolation
    let get_sample = |i: isize| -> Option<f64> {
        if i < 0 || i >= n as isize {
            None
        } else {
            let val = samples[i as usize];
            if val.is_nan() { None } else { Some(val) }
        }
    };

    // Try to get all 4 points
    if let (Some(y0), Some(y1), Some(y2), Some(y3)) = (
        get_sample(idx - 1),
        get_sample(idx),
        get_sample(idx + 1),
        get_sample(idx + 2),
    ) {
        let frac = position - idx as f64;
        let t = frac;
        let t2 = t * t;
        let t3 = t2 * t;

        let a0 = -0.5 * y0 + 1.5 * y1 - 1.5 * y2 + 0.5 * y3;
        let a1 = y0 - 2.5 * y1 + 2.0 * y2 - 0.5 * y3;
        let a2 = -0.5 * y0 + 0.5 * y2;
        let a3 = y1;

        return Some(a0 * t3 + a1 * t2 + a2 * t + a3);
    }

    // Fall back to linear interpolation
    linear_interpolate_with_undefined(samples, position)
}

/// Sinc (band-limited) interpolation
///
/// This provides the theoretically optimal interpolation for band-limited signals.
fn sinc_interpolate(samples: &[f64], position: f64, num_terms: usize) -> f64 {
    let n = samples.len();
    if n == 0 {
        return 0.0;
    }
    if n == 1 {
        return samples[0];
    }

    let idx = position.floor() as isize;
    let frac = position - idx as f64;

    // If we're exactly on a sample, return it
    if frac.abs() < 1e-10 {
        let clamped_idx = idx.max(0).min(n as isize - 1) as usize;
        return samples[clamped_idx];
    }

    let mut sum = 0.0;
    let mut weight_sum = 0.0;

    let half_terms = num_terms as isize / 2;
    let start = (idx - half_terms).max(0);
    let end = (idx + half_terms + 1).min(n as isize);

    for i in start..end {
        let x = position - i as f64;
        let sinc = if x.abs() < 1e-10 {
            1.0
        } else {
            let pi_x = std::f64::consts::PI * x;
            pi_x.sin() / pi_x
        };

        // Apply a windowing function (Hanning) to reduce ringing
        let window_pos = (i - start) as f64 / (end - start - 1).max(1) as f64;
        let window = 0.5 * (1.0 - (2.0 * std::f64::consts::PI * window_pos).cos());

        let weight = sinc * window;
        sum += samples[i as usize] * weight;
        weight_sum += weight;
    }

    if weight_sum.abs() > 1e-10 {
        sum / weight_sum
    } else {
        samples[idx.max(0).min(n as isize - 1) as usize]
    }
}

/// Parabolic interpolation to find the peak of a discrete signal
///
/// Given three consecutive samples where the middle one is the maximum,
/// returns the fractional offset from the center sample and the interpolated
/// peak value.
///
/// # Returns
/// (offset, peak_value) where offset is in the range [-0.5, 0.5]
pub fn parabolic_peak(y0: f64, y1: f64, y2: f64) -> (f64, f64) {
    // For a parabola passing through (-1, y0), (0, y1), (1, y2):
    // The peak is at offset = (y0 - y2) / (2 * (y0 + y2 - 2*y1))
    let denominator = 2.0 * (y0 + y2 - 2.0 * y1);

    if denominator.abs() < 1e-10 {
        // Flat or nearly flat - return center
        return (0.0, y1);
    }

    let offset = (y0 - y2) / denominator;
    // Peak value: y1 + (y0 - y2)^2 / (8 * (y0 + y2 - 2*y1))
    let peak = y1 - (y0 - y2) * (y0 - y2) / (4.0 * denominator);

    (offset.clamp(-0.5, 0.5), peak)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_linear_interpolation() {
        let samples = vec![0.0, 1.0, 2.0, 3.0];

        // At sample points
        assert_relative_eq!(
            Interpolation::Linear.interpolate(&samples, 0.0).unwrap(),
            0.0,
            epsilon = 1e-10
        );
        assert_relative_eq!(
            Interpolation::Linear.interpolate(&samples, 1.0).unwrap(),
            1.0,
            epsilon = 1e-10
        );

        // Between samples
        assert_relative_eq!(
            Interpolation::Linear.interpolate(&samples, 0.5).unwrap(),
            0.5,
            epsilon = 1e-10
        );
        assert_relative_eq!(
            Interpolation::Linear.interpolate(&samples, 1.5).unwrap(),
            1.5,
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_cubic_interpolation() {
        // For a parabola, cubic interpolation should be exact
        let samples: Vec<f64> = (0..10).map(|i| (i as f64).powi(2)).collect();

        for i in 1..8 {
            let pos = i as f64 + 0.5;
            let interpolated = Interpolation::Cubic.interpolate(&samples, pos).unwrap();
            let expected = pos.powi(2);
            assert_relative_eq!(interpolated, expected, epsilon = 0.01);
        }
    }

    #[test]
    fn test_interpolation_with_nan() {
        let samples = vec![1.0, 2.0, f64::NAN, 4.0, 5.0];

        // Should return None when both neighbors are NaN
        // In this case, position 2.5 has NaN on left
        let result = Interpolation::Linear.interpolate_with_undefined(&samples, 2.5);
        assert!(result.is_some()); // Should fall back to the defined value

        // Position at a NaN value itself with nearest
        let result = Interpolation::Nearest.interpolate_with_undefined(&samples, 2.0);
        assert!(result.is_none());
    }

    #[test]
    fn test_parabolic_peak() {
        // Perfect parabola centered at x=0 with peak at 10
        // y = 10 - x^2
        let y0 = 10.0 - 1.0; // x = -1
        let y1 = 10.0; // x = 0
        let y2 = 10.0 - 1.0; // x = 1

        let (offset, peak) = parabolic_peak(y0, y1, y2);
        assert_relative_eq!(offset, 0.0, epsilon = 1e-10);
        assert_relative_eq!(peak, 10.0, epsilon = 1e-10);

        // Asymmetric case
        // y = 10 - (x - 0.3)^2 at x = -1, 0, 1
        let true_peak_x: f64 = 0.3;
        let y0 = 10.0 - (-1.0_f64 - true_peak_x).powi(2);
        let y1 = 10.0 - (0.0_f64 - true_peak_x).powi(2);
        let y2 = 10.0 - (1.0_f64 - true_peak_x).powi(2);

        let (offset, _peak) = parabolic_peak(y0, y1, y2);
        assert_relative_eq!(offset, true_peak_x, epsilon = 1e-10);
    }
}
