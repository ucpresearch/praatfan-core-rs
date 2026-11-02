//! Sound type for audio data representation and manipulation
//!
//! The Sound type is the fundamental data structure representing audio samples.
//! It supports loading from various audio formats (WAV, MP3, FLAC, OGG),
//! extracting segments, and applying filters.

#[cfg(feature = "file-io")]
use std::fs::File;
#[cfg(feature = "file-io")]
use std::path::Path;

#[cfg(feature = "decode")]
use symphonia::core::audio::SampleBuffer;
#[cfg(feature = "decode")]
use symphonia::core::codecs::DecoderOptions;
#[cfg(feature = "decode")]
use symphonia::core::formats::FormatOptions;
#[cfg(feature = "decode")]
use symphonia::core::io::MediaSourceStream;
#[cfg(feature = "decode")]
use symphonia::core::meta::MetadataOptions;
#[cfg(feature = "decode")]
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
    /// Praat's `x1` (time of the first sample's centre) when it is not
    /// `start_time + dx/2`, stored as Praat computes it so that frame
    /// placement is bit-identical. Set by [`Sound::resample`], where Praat keeps
    /// `xmin`/`xmax` and centres the new samples:
    /// `x1 = (xmin + xmax - (n - 1) / fs) / 2`.
    x1_override: Option<f64>,
    /// Praat's `xmax` (end of the time domain) when it is not
    /// `start_time + n / fs`: an extracted part keeps the requested end time,
    /// and a resampled sound keeps its source's domain.
    xmax_override: Option<f64>,
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
    /// use praatfan_core::Sound;
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
            x1_override: None,
            xmax_override: None,
        }
    }

    /// Create a Sound from owned samples (avoids cloning)
    pub fn from_samples_owned(samples: Vec<f64>, sample_rate: f64) -> Self {
        Self {
            samples,
            sample_rate,
            start_time: 0.0,
            x1_override: None,
            xmax_override: None,
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
    #[cfg(feature = "file-io")]
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

        // Fall back to desphere for NIST SPHERE (incl. shorten-compressed), which
        // symphonia/hound cannot decode. Only attempted on the error path, so
        // ordinary formats are unaffected; if desphere can't decode it either, we
        // surface the original symphonia error below.
        if let Ok(sound) = Self::from_file_desphere(path) {
            return Ok(sound);
        }

        // If symphonia failed and the fallbacks didn't apply, return the symphonia error
        Self::from_file_symphonia(path)
    }

    /// Load a Sound from an audio file, keeping channels separate
    ///
    /// Returns a `Vec<Sound>` with one Sound per channel. This is useful for
    /// analysis algorithms that need to process channels separately and then
    /// average results (like Praat's spectrogram for stereo files).
    ///
    /// # Arguments
    /// * `path` - Path to the audio file
    ///
    /// # Returns
    /// A Vec of Sound objects, one per channel. For mono files, returns a Vec with one element.
    #[cfg(feature = "file-io")]
    pub fn from_file_channels<P: AsRef<Path>>(path: P) -> Result<Vec<Self>> {
        let path = path.as_ref();

        // Try symphonia first
        match Self::from_file_symphonia_channels(path) {
            Ok(sounds) => return Ok(sounds),
            Err(_) => {
                if let Some(ext) = path.extension() {
                    if ext.to_string_lossy().to_lowercase() == "wav" {
                        return Self::from_file_wav_channels(path);
                    }
                }
            }
        }

        // NIST SPHERE (incl. shorten-compressed) fallback via desphere.
        if let Ok(sounds) = Self::from_file_desphere_channels(path) {
            return Ok(sounds);
        }

        Self::from_file_symphonia_channels(path)
    }

    /// Load a Sound from any supported format using symphonia
    ///
    /// Supports WAV, MP3, FLAC, OGG Vorbis, and other formats.
    #[cfg(feature = "file-io")]
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
            x1_override: None,
            xmax_override: None,
        })
    }

    /// Load a Sound from a WAV file using hound
    #[cfg(feature = "file-io")]
    fn from_file_wav<P: AsRef<Path>>(path: P) -> Result<Self> {
        Self::from_wav_reader(hound::WavReader::open(path)?)
    }

    /// Decode NIST SPHERE (incl. shorten-compressed) to a mono Sound via desphere.
    ///
    /// Transcodes the file to in-memory WAV, then reuses the hound WAV reader.
    /// Returns an error (not a panic) if the file is not SPHERE desphere can decode.
    #[cfg(feature = "file-io")]
    fn from_file_desphere<P: AsRef<Path>>(path: P) -> Result<Self> {
        let wav = Self::desphere_transcode(path)?;
        Self::from_wav_reader(hound::WavReader::new(std::io::Cursor::new(wav))?)
    }

    /// Decode NIST SPHERE to per-channel Sounds via desphere (see `from_file_desphere`).
    #[cfg(feature = "file-io")]
    fn from_file_desphere_channels<P: AsRef<Path>>(path: P) -> Result<Vec<Self>> {
        let wav = Self::desphere_transcode(path)?;
        Self::from_wav_reader_channels(hound::WavReader::new(std::io::Cursor::new(wav))?)
    }

    /// Read a file and transcode it from NIST SPHERE to WAV bytes via desphere.
    #[cfg(feature = "file-io")]
    fn desphere_transcode<P: AsRef<Path>>(path: P) -> Result<Vec<u8>> {
        let bytes = std::fs::read(path).map_err(PraatError::Io)?;
        desphere::transcode(&bytes)
            .map_err(|e| PraatError::Analysis(format!("desphere SPHERE decode failed: {}", e)))
    }

    /// Load a mono Sound from any hound WAV reader (file or in-memory bytes).
    #[cfg(feature = "file-io")]
    fn from_wav_reader<R: std::io::Read>(reader: hound::WavReader<R>) -> Result<Self> {
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
            x1_override: None,
            xmax_override: None,
        })
    }

    /// Load channels separately using symphonia
    #[cfg(feature = "file-io")]
    fn from_file_symphonia_channels<P: AsRef<Path>>(path: P) -> Result<Vec<Self>> {
        let path = path.as_ref();
        let file = File::open(path).map_err(PraatError::Io)?;
        let mss = MediaSourceStream::new(Box::new(file), Default::default());

        let mut hint = Hint::new();
        if let Some(ext) = path.extension() {
            hint.with_extension(&ext.to_string_lossy());
        }
        Self::symphonia_channels(mss, hint)
    }

    /// Decode per-channel Sounds from any symphonia media stream (file or memory).
    #[cfg(feature = "decode")]
    fn symphonia_channels(mss: MediaSourceStream, hint: Hint) -> Result<Vec<Self>> {
        let format_opts = FormatOptions::default();
        let metadata_opts = MetadataOptions::default();
        let probed = symphonia::default::get_probe()
            .format(&hint, mss, &format_opts, &metadata_opts)
            .map_err(|e| PraatError::Analysis(format!("Failed to probe audio format: {}", e)))?;

        let mut format = probed.format;

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

        let decoder_opts = DecoderOptions::default();
        let mut decoder = symphonia::default::get_codecs()
            .make(&track.codec_params, &decoder_opts)
            .map_err(|e| PraatError::Analysis(format!("Failed to create decoder: {}", e)))?;

        // Collect samples for each channel separately
        let mut channel_samples: Vec<Vec<f64>> = (0..channels).map(|_| Vec::new()).collect();

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

            if packet.track_id() != track_id {
                continue;
            }

            let decoded = match decoder.decode(&packet) {
                Ok(decoded) => decoded,
                Err(symphonia::core::errors::Error::DecodeError(_)) => continue,
                Err(e) => {
                    return Err(PraatError::Analysis(format!("Decode error: {}", e)));
                }
            };

            let spec = *decoded.spec();
            let num_frames = decoded.frames();

            let mut sample_buf = SampleBuffer::<f32>::new(num_frames as u64, spec);
            sample_buf.copy_interleaved_ref(decoded);
            let samples_f32 = sample_buf.samples();

            // Deinterleave into separate channels
            for (i, sample) in samples_f32.iter().enumerate() {
                let channel_idx = i % channels;
                channel_samples[channel_idx].push(*sample as f64);
            }
        }

        // Create a Sound for each channel
        let sounds: Vec<Sound> = channel_samples
            .into_iter()
            .map(|samples| Sound {
                samples,
                sample_rate,
                start_time: 0.0,
                x1_override: None,
                xmax_override: None,
            })
            .collect();

        Ok(sounds)
    }

    /// Decode an in-memory audio file, keeping channels separate.
    ///
    /// Same decoder chain as [`Sound::from_file_channels`] (symphonia, then
    /// hound for WAV, then desphere for NIST SPHERE), without touching the
    /// filesystem. `extension` (e.g. `"flac"`) is only a format hint.
    #[cfg(feature = "decode")]
    pub fn from_bytes_channels(bytes: &[u8], extension: Option<&str>) -> Result<Vec<Self>> {
        let make_hint = || {
            let mut hint = Hint::new();
            if let Some(ext) = extension {
                hint.with_extension(ext);
            }
            hint
        };
        let cursor = std::io::Cursor::new(bytes.to_vec());
        let mss = MediaSourceStream::new(Box::new(cursor), Default::default());
        let symphonia_err = match Self::symphonia_channels(mss, make_hint()) {
            Ok(sounds) => return Ok(sounds),
            Err(e) => e,
        };
        if let Ok(reader) = hound::WavReader::new(std::io::Cursor::new(bytes)) {
            if let Ok(sounds) = Self::from_wav_reader_channels(reader) {
                return Ok(sounds);
            }
        }
        if let Ok(wav) = desphere::transcode(bytes) {
            if let Ok(reader) = hound::WavReader::new(std::io::Cursor::new(wav)) {
                return Self::from_wav_reader_channels(reader);
            }
        }
        Err(symphonia_err)
    }

    /// Load channels separately from WAV using hound
    #[cfg(feature = "file-io")]
    fn from_file_wav_channels<P: AsRef<Path>>(path: P) -> Result<Vec<Self>> {
        Self::from_wav_reader_channels(hound::WavReader::open(path)?)
    }

    /// Load per-channel Sounds from any hound WAV reader (file or in-memory bytes).
    #[cfg(feature = "decode")]
    fn from_wav_reader_channels<R: std::io::Read>(reader: hound::WavReader<R>) -> Result<Vec<Self>> {
        let spec = reader.spec();
        let sample_rate = spec.sample_rate as f64;
        let channels = spec.channels as usize;

        let all_samples: Vec<f64> = match spec.sample_format {
            hound::SampleFormat::Int => {
                let max_value = (1_i64 << (spec.bits_per_sample - 1)) as f64;
                reader
                    .into_samples::<i32>()
                    .map(|s| s.unwrap() as f64 / max_value)
                    .collect()
            }
            hound::SampleFormat::Float => {
                reader
                    .into_samples::<f32>()
                    .map(|s| s.unwrap() as f64)
                    .collect()
            }
        };

        // Deinterleave into separate channels
        let mut channel_samples: Vec<Vec<f64>> = (0..channels).map(|_| Vec::new()).collect();
        for (i, &sample) in all_samples.iter().enumerate() {
            let channel_idx = i % channels;
            channel_samples[channel_idx].push(sample);
        }

        // Create a Sound for each channel
        let sounds: Vec<Sound> = channel_samples
            .into_iter()
            .map(|samples| Sound {
                samples,
                sample_rate,
                start_time: 0.0,
                x1_override: None,
                xmax_override: None,
            })
            .collect();

        Ok(sounds)
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
        self.xmax_override.unwrap_or(self.start_time + self.duration())
    }

    /// Get the sample period (time step between samples)
    ///
    /// This is 1 / sample_rate. In Praat, this is called `dx`.
    pub fn dx(&self) -> f64 {
        1.0 / self.sample_rate
    }

    /// Get the time of the first sample center
    ///
    /// In Praat's terminology, this is x1 = xmin + 0.5 * dx.
    /// Samples are considered to be centered at their time positions.
    pub fn x1(&self) -> f64 {
        self.x1_override.unwrap_or(self.start_time + 0.5 * self.dx())
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

    /// Sample value at `time` with Sinc70 interpolation, Praat's default for
    /// `Sound: Get value at time` (see `get_value_at_time_interpolated`).
    ///
    /// Returns None if the time is outside the sound's duration.
    pub fn get_value_at_time(&self, time: f64) -> Option<f64> {
        // Praat's default interpolation for `Sound: Get value at time`.
        self.get_value_at_time_interpolated(time, crate::Interpolation::Sinc70)
    }

    /// Sample value at `time` with Praat's `Vector_getValueAtX` semantics
    /// (`Sound: Get value at time...`; Praat's default interpolation there is
    /// `Sinc70`). Undefined outside `[x1 - dx/2, x1 + (n - 1/2) dx]`.
    pub fn get_value_at_time_interpolated(
        &self,
        time: f64,
        interpolation: crate::Interpolation,
    ) -> Option<f64> {
        crate::interpolation::vector_value_at_x(&self.samples, self.x1(), self.dx(), time, interpolation)
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
        // Port of Praat's Sound_extractPart (fon/Sound.cpp).
        // Function_unidirectionalAutowindow: an empty range means everything.
        let (mut tmin, mut tmax) = (start_time, end_time);
        if tmin >= tmax {
            tmin = self.start_time;
            tmax = self.end_time();
        }
        // Window tails may extend outside the requested domain.
        if relative_width != 1.0 {
            let margin = 0.5 * (relative_width - 1.0) * (tmax - tmin);
            tmin -= margin;
            tmax += margin;
        }
        // All real or virtual samples that fit within [tmin, tmax] (1-based).
        let dx = self.dx();
        let x1 = self.x1();
        let itmin = 1 + libm::ceil((tmin - x1) / dx) as i64;
        let itmax = 1 + libm::floor((tmax - x1) / dx) as i64;
        if itmax < itmin {
            return Err(PraatError::InvalidParameter(
                "Extracted Sound would contain no samples".to_string(),
            ));
        }
        let n = (itmax - itmin + 1) as usize;
        // Virtual samples (outside the sound) stay zero.
        let nx = self.samples.len() as i64;
        let mut extracted = vec![0.0; n];
        for it in itmin.max(1)..=itmax.min(nx) {
            extracted[(it - itmin) as usize] = self.samples[(it - 1) as usize];
        }
        multiply_by_window_praat(&mut extracted, window_shape);

        let mut new_x1 = x1 + (itmin - 1) as f64 * dx;
        let mut new_xmin = tmin;
        let mut new_xmax = tmax;
        if !preserve_times {
            new_xmin = 0.0;
            new_xmax -= tmin;
            new_x1 -= tmin;
        }
        Ok(Sound {
            samples: extracted,
            sample_rate: self.sample_rate,
            start_time: new_xmin,
            x1_override: Some(new_x1),
            xmax_override: Some(new_xmax),
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
    /// The filter is: `y[n] = x[n] - alpha * x[n-1]`
    /// where alpha = exp(-2 * pi * from_frequency / sample_rate)
    pub fn pre_emphasis(&self, from_frequency: f64) -> Sound {
        if self.samples.is_empty() {
            return self.clone();
        }

        let alpha = crate::math::exp(-2.0 * std::f64::consts::PI * from_frequency / self.sample_rate);

        let mut filtered = Vec::with_capacity(self.samples.len());
        filtered.push(self.samples[0]);

        for i in 1..self.samples.len() {
            filtered.push(self.samples[i] - alpha * self.samples[i - 1]);
        }

        Sound {
            samples: filtered,
            sample_rate: self.sample_rate,
            start_time: self.start_time,
            x1_override: self.x1_override,
            xmax_override: self.xmax_override,
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

        let alpha = crate::math::exp(-2.0 * std::f64::consts::PI * from_frequency / self.sample_rate);

        let mut filtered = Vec::with_capacity(self.samples.len());
        filtered.push(self.samples[0]);

        for i in 1..self.samples.len() {
            filtered.push(self.samples[i] + alpha * filtered[i - 1]);
        }

        Sound {
            samples: filtered,
            sample_rate: self.sample_rate,
            start_time: self.start_time,
            x1_override: self.x1_override,
            xmax_override: self.xmax_override,
        }
    }

    /// Praat's `Vector_scale`: multiply so that the largest absolute sample
    /// becomes `peak` (unchanged if the sound is all zeros). Praat's
    /// pre-/de-emphasis commands finish with `Vector_scale (me, 0.99)`.
    pub fn scale_peak(&self, peak: f64) -> Sound {
        let extremum = self.samples.iter().fold(0.0f64, |m, &x| m.max(x.abs()));
        if extremum == 0.0 {
            self.clone()
        } else {
            self.scale(peak / extremum)
        }
    }

    /// Multiply samples by a constant
    pub fn scale(&self, factor: f64) -> Sound {
        Sound {
            samples: self.samples.iter().map(|&s| s * factor).collect(),
            sample_rate: self.sample_rate,
            start_time: self.start_time,
            x1_override: self.x1_override,
            xmax_override: self.xmax_override,
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
            x1_override: None,
            xmax_override: None,
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
            .map(|i| amplitude * crate::math::sin(omega * i as f64 + phase))
            .collect();

        Sound {
            samples,
            sample_rate,
            start_time: 0.0,
            x1_override: None,
            xmax_override: None,
        }
    }

    /// Create silence
    pub fn create_silence(duration: f64, sample_rate: f64) -> Sound {
        let n_samples = (duration * sample_rate).round() as usize;
        Sound {
            samples: vec![0.0; n_samples],
            sample_rate,
            start_time: 0.0,
            x1_override: None,
            xmax_override: None,
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
    /// This method uses windowed sinc interpolation matching Praat's Sound_resample
    /// exactly. It's used internally for formant analysis where Praat resamples
    /// to 2 × max_formant before LPC.
    ///
    /// # Arguments
    /// * `new_sample_rate` - Target sample rate in Hz
    ///
    /// # Returns
    /// A new Sound at the target sample rate
    pub fn resample(&self, new_sample_rate: f64) -> Sound {
        if self.samples.is_empty() {
            return self.clone();
        }

        let upfactor = new_sample_rate / self.sample_rate;

        // Praat (Sound_resample): an exact doubling goes through the FFT
        // upsampler, not sinc interpolation.
        if (upfactor - 2.0).abs() < 1e-6 {
            return self.upsample_by_two();
        }
        // If ratio is ~1, return copy
        if (upfactor - 1.0).abs() < 1e-6 {
            return self.clone();
        }

        // Praat: numberOfSamples = round((xmax - xmin) * samplingFrequency)
        let xmin = self.start_time;
        let xmax = self.end_time(); // Praat: my xmax
        let new_num_samples = ((xmax - xmin) * new_sample_rate).round() as usize;

        if new_num_samples == 0 {
            return Sound {
                samples: Vec::new(),
                sample_rate: new_sample_rate,
                start_time: self.start_time,
                x1_override: self.x1_override,
                xmax_override: self.xmax_override,
            };
        }

        // For downsampling, apply FFT-based anti-aliasing filter first
        let filtered_samples = if upfactor < 1.0 {
            self.fft_lowpass_filter(upfactor)
        } else {
            self.samples.clone()
        };

        // Praat time conventions:
        // Original: x1_orig = xmin + 0.5 * dx_orig (time of first sample center)
        // The new sound uses:
        // x1_new = 0.5 * (xmin + xmax - (numberOfSamples - 1) / samplingFrequency)
        let dx_orig = 1.0 / self.sample_rate;
        let dx_new = 1.0 / new_sample_rate;
        let _ = dx_orig;
        let x1_orig = self.x1();
        let x1_new = 0.5 * (xmin + xmax - (new_num_samples - 1) as f64 / new_sample_rate);

        // Sinc interpolation precision (Praat default is 50)
        let precision: isize = 50;

        // Each output sample is independent, so interpolate in chunks (in
        // parallel when enabled). The FFT low-pass above stays serial.
        const CHUNK: usize = 16384;
        let num_chunks = new_num_samples.div_ceil(CHUNK);
        let chunks: Vec<Vec<f64>> = crate::par::map_init(num_chunks, || (), |_, c| {
            (c * CHUNK..((c + 1) * CHUNK).min(new_num_samples))
                .map(|i| {
                    // Time of new sample i (0-based, but Praat formula uses 1-based)
                    // Praat: x = Sampled_indexToX(thee, i) = x1 + (i - 1) * dx (1-based)
                    // For 0-based: x = x1_new + i * dx_new
                    let x = x1_new + i as f64 * dx_new;

                    // Index in original (1-based for sinc_interpolate_1based)
                    // Praat: index = Sampled_xToIndex(me, x) = (x - x1) / dx + 1.0
                    let index = (x - x1_orig) / dx_orig + 1.0;

                    crate::interpolation::num_interpolate_sinc(&filtered_samples, index, precision)
                })
                .collect()
        });
        let new_samples = chunks.concat();

        Sound {
            samples: new_samples,
            sample_rate: new_sample_rate,
            start_time: self.start_time,
            x1_override: Some(x1_new),
            xmax_override: Some(self.end_time()),
        }
    }

    /// Apply FFT-based low-pass filter for anti-aliasing (matches Praat's Sound_resample exactly)
    ///
    /// Praat's algorithm from Sound.cpp:
    /// 1. Pad signal with zeros (antiTurnAround = 1000 on each side)
    /// 2. FFT to frequency domain using NUMrealft
    /// 3. Zero frequencies: data[floor(upfactor * nfft)..nfft] and data[2] (Nyquist)
    /// 4. Inverse FFT back to time domain
    /// 5. Scale by 1/nfft
    ///
    /// Praat's NUMrealft packed format (1-based):
    ///   data[1] = DC, data[2] = Nyquist
    ///   data[2k+1], data[2k+2] = Re, Im of frequency bin k (for k >= 1)
    /// Port of Praat's `Sound_upsample` (fon/Sound.cpp): zero-pad by 1000
    /// samples on each side, real FFT of size `nfft`, linear taper over the
    /// top 5% of bins, Nyquist zeroed, inverse real FFT of size `2 * nfft`.
    /// The left edge of the first sample stays put, so `start_time` is
    /// unchanged and there are exactly `2 * n` output samples.
    fn upsample_by_two(&self) -> Sound {
        use crate::utils::fft_fftpack::RealFftPlan;

        let n = self.samples.len();
        let anti_turn_around = 1000;
        let nfft = (n + 2 * anti_turn_around).next_power_of_two();

        let mut data = vec![0.0_f64; 2 * nfft];
        data[anti_turn_around..anti_turn_around + n].copy_from_slice(&self.samples);
        {
            let mut plan = RealFftPlan::new(nfft);
            crate::utils::fft_fftpack::realft_forward_praat(&mut plan, &mut data[..nfft]);
        }
        // Praat (1-based): imin = (integer) (nfft * 0.95);
        //   for i in imin+1 ..= nfft: data[i] *= (nfft - i) / (nfft - imin)
        let imin = (nfft as f64 * 0.95) as usize;
        for i in imin + 1..=nfft {
            data[i - 1] *= (nfft - i) as f64 / (nfft - imin) as f64;
        }
        data[1] = 0.0; // Praat data[2]: Nyquist in NUMrealft's layout
        {
            let mut plan = RealFftPlan::new(2 * nfft);
            crate::utils::fft_fftpack::realft_backward_praat(&mut plan, &mut data);
        }
        let factor = 1.0 / nfft as f64;
        let samples = (0..2 * n)
            .map(|i| data[i + 2 * anti_turn_around] * factor)
            .collect();
        Sound {
            samples,
            sample_rate: 2.0 * self.sample_rate,
            start_time: self.start_time,
            // Praat: new x1 = old x1 - 0.5 * (old dx - new dx)
            x1_override: Some(self.x1() - 0.5 * (self.dx() - 0.5 * self.dx())),
            xmax_override: Some(self.end_time()),
        }
    }

    fn fft_lowpass_filter(&self, upfactor: f64) -> Vec<f64> {
        use crate::utils::fft_fftpack::RealFftPlan;

        let n = self.samples.len();
        // Pad to power of 2 with turnaround (Praat uses 1000 on each side)
        let anti_turn_around = 1000;
        let nfft = (n + 2 * anti_turn_around).next_power_of_two();

        // Create padded signal: zeros, then signal, then zeros
        let mut data = vec![0.0_f64; nfft];
        for i in 0..n {
            data[anti_turn_around + i] = self.samples[i];
        }

        // Forward real FFT using Praat-compatible FFTPACK (drftf1). This
        // leaves `data` in FFTPACK's packed real-spectrum layout:
        //   data[0] = DC
        //   data[2k-1], data[2k] = Re(bin k), Im(bin k)   for k in 1..nfft/2
        //   data[nfft-1]        = Nyquist
        //
        // Praat's NUMrealft rotates Nyquist into data[1] to get
        // `[DC, Nyquist, Re1, Im1, Re2, Im2, ...]`. Praat's downstream code
        // (Sound_resample at line 420) then zeros `data[i]` for
        // `i = floor(upfactor * nfft) .. nfft` AND explicitly zeros
        // `data[2]` (which is the Nyquist in the rotated layout). Below we
        // perform the same operations in Praat's ROTATED layout so that the
        // index arithmetic reads identically to Praat's source.
        let mut plan = RealFftPlan::new(nfft);
        crate::utils::fft_fftpack::realft_forward_praat(&mut plan, &mut data);

        // Praat (1-based): for (i = floor(upfactor * nfft); i <= nfft; i++)
        //                      data[i] = 0;
        //                  data[2] = 0;  // Nyquist
        // In our 0-based array, Praat's `data[i]` is `data[i-1]` for i=1..=nfft.
        let i_start_1based = (upfactor * nfft as f64).floor() as usize;
        // Clamp to valid range [1, nfft]
        let i_start_1based = i_start_1based.max(1);
        for i_1 in i_start_1based..=nfft {
            data[i_1 - 1] = 0.0;
        }
        if nfft >= 2 {
            data[1] = 0.0; // Praat: data[2] in 1-based = Nyquist in rotated layout
        }

        // Inverse real FFT (un-rotate and drftb1). Result is un-normalised
        // (caller must divide by nfft).
        crate::utils::fft_fftpack::realft_backward_praat(&mut plan, &mut data);

        // Extract the original window, scale by 1/nfft.
        let scale = 1.0 / nfft as f64;
        let mut result = Vec::with_capacity(n);
        for i in 0..n {
            result.push(data[anti_turn_around + i] * scale);
        }
        result
    }
}

/// Praat's `Sound_multiplyByWindow` (fon/Sound.cpp). `Gaussian` is Praat's
/// "Gaussian1" and `Kaiser` its "Kaiser1".
fn multiply_by_window_praat(x: &mut [f64], shape: WindowShape) {
    use std::f64::consts::PI;
    let n = x.len();
    if n == 0 {
        return;
    }
    let nf = n as f64;
    let imid = 0.5 * (nf + 1.0);
    match shape {
        WindowShape::Rectangular => {}
        WindowShape::Triangular => {
            for (i, v) in x.iter_mut().enumerate() {
                let phase = (i + 1) as f64 / nf;
                *v *= 1.0 - (2.0 * phase - 1.0).abs();
            }
        }
        WindowShape::Parabolic => {
            for (i, v) in x.iter_mut().enumerate() {
                let phase = (i + 1) as f64 / nf;
                *v *= 1.0 - (2.0 * phase - 1.0) * (2.0 * phase - 1.0);
            }
        }
        WindowShape::Hanning => {
            for (i, v) in x.iter_mut().enumerate() {
                let phase = (i + 1) as f64 / nf;
                *v *= 0.5 * (1.0 - crate::math::cos(2.0 * PI * phase));
            }
        }
        WindowShape::Hamming => {
            for (i, v) in x.iter_mut().enumerate() {
                let phase = (i + 1) as f64 / nf;
                *v *= 0.54 - 0.46 * crate::math::cos(2.0 * PI * phase);
            }
        }
        WindowShape::Gaussian => {
            let edge = crate::math::exp(-3.0);
            let onebyedge1 = 1.0 / (1.0 - edge);
            for (i, v) in x.iter_mut().enumerate() {
                let phase = ((i + 1) as f64 - imid) / nf;
                *v *= (crate::math::exp(-12.0 * phase * phase) - edge) * onebyedge1;
            }
        }
        WindowShape::Kaiser => {
            let factor = 1.0 / crate::intensity::bessel_i0(2.0 * PI);
            for (i, v) in x.iter_mut().enumerate() {
                let phase = 2.0 * ((i + 1) as f64 - imid) / nf;
                let root = 1.0 - phase * phase;
                *v *= if root <= 0.0 {
                    0.0
                } else {
                    factor * crate::intensity::bessel_i0(2.0 * PI * libm::sqrt(root))
                };
            }
        }
    }
}

/// Sinc interpolation matching Praat's NUM_interpolate_sinc exactly
/// From Praat's melder/NUMinterpol.cpp
///
/// Note: This function uses 1-BASED indexing to match Praat exactly.
/// So x=1.0 refers to samples[0], x=2.0 refers to samples[1], etc.
pub(crate) fn sinc_interpolate_1based(samples: &[f64], x: f64, max_depth: isize) -> f64 {
    let n = samples.len() as isize;
    if n < 1 {
        return f64::NAN;
    }

    // Boundary handling: constant extrapolation
    if x < 1.0 {
        return samples[0];
    }
    if x > n as f64 {
        return samples[(n - 1) as usize];
    }

    let midleft = x.floor() as isize;
    let midright = midleft + 1;

    // If x is exactly on a sample, return that sample
    if x == midleft as f64 {
        return samples[(midleft - 1) as usize];
    }

    // Clip max_depth to valid range
    let max_depth = max_depth.min(midright - 1).min(n - midleft);

    // For linear interpolation (maxDepth <= 1)
    if max_depth <= 1 {
        let yl = samples[(midleft - 1) as usize];
        let yr = samples[(midright - 1) as usize];
        return yl + (x - midleft as f64) * (yr - yl);
    }

    let left = midright - max_depth;
    let right = midleft + max_depth;

    let mut result = 0.0;

    // Left half: from midleft down to left.
    //
    // The raised-cosine window width is Praat-version-specific:
    //
    //   Praat ≤ 6.1.38 (parselmouth 0.4.x, our primary parity target):
    //       leftDepth  = x - left + 1.0              (ASYMMETRIC, boundary-aware)
    //       rightDepth = right - x + 1.0
    //
    //   Praat ≥ 6.2.x / current (2025):
    //       leftDepth = rightDepth = maxDepth + 0.5  (symmetric)
    //
    // The 6.1.38 formula makes the window wider on the side with more
    // available samples, which matters near signal boundaries (where
    // maxDepth is clipped asymmetrically). Using `maxDepth + 0.5` there
    // produces ~1.5e-5 per-sample drift from Praat's actual resample
    // output, concentrated in the last/first ~5 samples of each resample.
    // We use the 6.1.38 formula to match parselmouth's bundled Praat.
    {
        let left_depth = x - left as f64 + 1.0;
        let window_phase_step = std::f64::consts::PI / left_depth;
        let sin_window_phase_step = crate::math::sin(window_phase_step);
        let cos_window_phase_step = crate::math::cos(window_phase_step);

        let mut left_phase = std::f64::consts::PI * (x - midleft as f64);
        let mut half_sin_left_phase = 0.5 * crate::math::sin(left_phase);

        let window_phase = left_phase / left_depth;
        let mut sin_window_phase = crate::math::sin(window_phase);
        let mut cos_window_phase = crate::math::cos(window_phase);

        for ix in (left..=midleft).rev() {
            let sinc_times_window = half_sin_left_phase / left_phase * (1.0 + cos_window_phase);
            result += samples[(ix - 1) as usize] * sinc_times_window;

            left_phase += std::f64::consts::PI;
            half_sin_left_phase = -half_sin_left_phase;

            let next_sin = cos_window_phase * sin_window_phase_step + sin_window_phase * cos_window_phase_step;
            let next_cos = cos_window_phase * cos_window_phase_step - sin_window_phase * sin_window_phase_step;
            sin_window_phase = next_sin;
            cos_window_phase = next_cos;
        }
    }

    // Right half: from midright up to right.
    {
        let right_depth = right as f64 - x + 1.0;
        let window_phase_step = std::f64::consts::PI / right_depth;
        let sin_window_phase_step = crate::math::sin(window_phase_step);
        let cos_window_phase_step = crate::math::cos(window_phase_step);

        let mut right_phase = std::f64::consts::PI * (midright as f64 - x);
        let mut half_sin_right_phase = 0.5 * crate::math::sin(right_phase);

        let window_phase = right_phase / right_depth;
        let mut sin_window_phase = crate::math::sin(window_phase);
        let mut cos_window_phase = crate::math::cos(window_phase);

        for ix in midright..=right {
            let sinc_times_window = half_sin_right_phase / right_phase * (1.0 + cos_window_phase);
            result += samples[(ix - 1) as usize] * sinc_times_window;

            right_phase += std::f64::consts::PI;
            half_sin_right_phase = -half_sin_right_phase;

            let next_sin = cos_window_phase * sin_window_phase_step + sin_window_phase * cos_window_phase_step;
            let next_cos = cos_window_phase * cos_window_phase_step - sin_window_phase * sin_window_phase_step;
            sin_window_phase = next_sin;
            cos_window_phase = next_cos;
        }
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
        let samples: Vec<f64> = (0..100).map(|i| crate::math::sin(i as f64 * 0.1)).collect();
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
