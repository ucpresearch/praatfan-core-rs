//! Utility modules for signal processing
//!
//! This module contains low-level signal processing utilities used by
//! the analysis algorithms.

pub mod fft;
pub mod lpc;

pub use fft::{Fft, FftDirection};
pub use lpc::{lpc_burg, lpc_to_formants};
