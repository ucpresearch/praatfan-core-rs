//! Transcendental functions used by every analysis.
//!
//! Praat calls the C library's `sin`, `cos`, `exp`, ... and its last bits are
//! those of glibc. On noise-dominated frames (resampled digital silence,
//! near-flat pitch peaks) those last bits decide the answer, so where the
//! functions come from is a parity question.
//!
//! - Default: the pure-Rust `libm` crate. Identical results on every
//!   platform, including wasm32; differs from glibc in the last bit on a
//!   small fraction of arguments.
//! - Feature `system-libm`: the platform C library (glibc on Linux), i.e.
//!   exactly what Praat on the same machine computes. Results then depend on
//!   the host library and, under glibc, on the CPU (glibc picks FMA or
//!   non-FMA code at run time). Ignored on wasm32. Tested once (2026-11-02,
//!   Linux x86-64, glibc 2.43, parselmouth 0.4.7): resampling bit-identical,
//!   every pitch case of `scripts/parity_sweep.py` exact. Not a target we
//!   chase: other C libraries or glibc versions may differ, as Praat does.
//!
//! `sqrt`, `floor`, `ceil`, `round` and `fma` are exactly rounded everywhere
//! and stay on `libm` directly.

#[cfg(all(feature = "system-libm", not(target_arch = "wasm32")))]
mod imp {
    mod c {
        extern "C" {
            pub fn sin(x: f64) -> f64;
            pub fn cos(x: f64) -> f64;
            pub fn exp(x: f64) -> f64;
            pub fn log(x: f64) -> f64;
            pub fn log2(x: f64) -> f64;
            pub fn log10(x: f64) -> f64;
            pub fn pow(x: f64, y: f64) -> f64;
            pub fn hypot(x: f64, y: f64) -> f64;
            pub fn atan2(y: f64, x: f64) -> f64;
            pub fn sinh(x: f64) -> f64;
        }
    }
    macro_rules! wrap1 { ($($n:ident),*) => { $(#[inline] pub fn $n(x: f64) -> f64 { unsafe { c::$n(x) } })* } }
    macro_rules! wrap2 { ($($n:ident),*) => { $(#[inline] pub fn $n(x: f64, y: f64) -> f64 { unsafe { c::$n(x, y) } })* } }
    wrap1!(sin, cos, exp, log, log2, log10, sinh);
    wrap2!(pow, hypot, atan2);
    pub const BACKEND: &str = "system";
}

#[cfg(not(all(feature = "system-libm", not(target_arch = "wasm32"))))]
mod imp {
    pub use libm::{atan2, cos, exp, hypot, log, log10, log2, pow, sin, sinh};
    pub const BACKEND: &str = "libm";
}

pub use imp::*;
