/* tslint:disable */
/* eslint-disable */

/**
 * Formant contour - WASM wrapper
 */
export class Formant {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    /**
     * Get all bandwidths for a specific formant as Float64Array
     */
    bandwidth_values(formant_number: number): Float64Array;
    /**
     * Get all values for a specific formant as Float64Array
     */
    formant_values(formant_number: number): Float64Array;
    /**
     * Get bandwidth at a specific time
     */
    get_bandwidth_at_time(formant_number: number, time: number, unit: string, interpolation: string): number | undefined;
    /**
     * Get the time of a specific frame
     */
    get_time_from_frame(frame: number): number;
    /**
     * Get formant value at a specific time
     *
     * @param formant_number - Formant number (1 for F1, 2 for F2, etc.)
     * @param time - Time to query
     * @param unit - Unit: "hertz", "bark", "mel", "erb"
     * @param interpolation - Interpolation: "nearest", "linear", "cubic"
     */
    get_value_at_time(formant_number: number, time: number, unit: string, interpolation: string): number | undefined;
    /**
     * Get all frame times as Float64Array
     */
    times(): Float64Array;
    readonly max_num_formants: number;
    readonly num_frames: number;
    readonly start_time: number;
    readonly time_step: number;
}

/**
 * FormantPath - multi-ceiling Burg analysis with Viterbi path selection.
 *
 * Mirrors Praat's FormantPath. Produced by `Sound.to_formant_path_burg`.
 */
export class FormantPath {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    /**
     * Access one candidate Formant by 0-based index.
     */
    candidate(index: number): Formant;
    /**
     * Ceiling frequencies in Hz as a Float64Array.
     */
    ceilings(): Float64Array;
    /**
     * Build a Formant by selecting, at each frame, the candidate indicated by
     * the path.
     */
    extract_formant(): Formant;
    /**
     * Stress of one candidate over `[t_min, t_max]`.
     */
    get_stress_of_candidate(t_min: number, t_max: number, from_formant: number, to_formant: number, parameters: Int32Array, power: number, candidate: number): number;
    /**
     * Stress for every candidate as a Float64Array.
     */
    get_stress_of_candidates(t_min: number, t_max: number, from_formant: number, to_formant: number, parameters: Int32Array, power: number): Float64Array;
    /**
     * Current per-frame selected candidate (0-based).
     */
    path(): Uint32Array;
    /**
     * Run Viterbi-optimal path selection (Praat's `Path finder`).
     *
     * @param parameters - Int32Array of Legendre coefficients per track.
     */
    path_finder(q_weight: number, frequency_change_weight: number, stress_weight: number, ceiling_change_weight: number, intensity_modulation_step_size: number, path_window_length: number, parameters: Int32Array, power: number): void;
    /**
     * Force every frame overlapping `[t_min, t_max]` to select `candidate`
     * (0-based).
     */
    set_path(t_min: number, t_max: number, candidate: number): void;
    /**
     * Number of candidate analyses (= 2*number_of_steps_up_down+1).
     */
    readonly num_candidates: number;
    readonly num_frames: number;
    readonly start_time: number;
    readonly time_step: number;
}

/**
 * Harmonicity (HNR) - WASM wrapper
 */
export class Harmonicity {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    /**
     * Get HNR value at a specific time
     */
    get_value_at_time(time: number, interpolation: string): number | undefined;
    max(): number | undefined;
    mean(): number | undefined;
    min(): number | undefined;
    /**
     * Get all frame times as Float64Array
     */
    times(): Float64Array;
    /**
     * Get all HNR values as Float64Array
     */
    values(): Float64Array;
    readonly num_frames: number;
    readonly start_time: number;
    readonly time_step: number;
}

/**
 * Intensity contour - WASM wrapper
 */
export class Intensity {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    /**
     * Get intensity value at a specific time
     */
    get_value_at_time(time: number, interpolation: string): number | undefined;
    max(): number | undefined;
    mean(): number | undefined;
    min(): number | undefined;
    /**
     * Get all frame times as Float64Array
     */
    times(): Float64Array;
    /**
     * Get all intensity values as Float64Array
     */
    values(): Float64Array;
    readonly num_frames: number;
    readonly start_time: number;
    readonly time_step: number;
}

/**
 * Pitch contour - WASM wrapper
 */
export class Pitch {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    /**
     * Get the time of a specific frame
     */
    get_time_from_frame(frame: number): number;
    /**
     * Get pitch value at a specific time
     *
     * @param time - Time to query
     * @param unit - Unit: "hertz", "mel", "semitones", "erb"
     * @param interpolation - Interpolation: "nearest", "linear", "cubic"
     * @returns Pitch value or undefined if unvoiced
     */
    get_value_at_time(time: number, unit: string, interpolation: string): number | undefined;
    /**
     * Per-frame strength of the Viterbi-selected candidate.
     *
     * Equivalent to parselmouth's `Pitch.selected_array['strength']`:
     * voiced frames return the winning candidate's autocorrelation strength,
     * unvoiced frames return the computed unvoiced-candidate penalty strength.
     * Returned as-is (no NaN handling) for Praat bit-parity.
     */
    strengths(): Float64Array;
    /**
     * Get all frame times as Float64Array
     */
    times(): Float64Array;
    /**
     * Get all pitch values as Float64Array (NaN for unvoiced)
     */
    values(): Float64Array;
    readonly num_frames: number;
    readonly pitch_ceiling: number;
    readonly pitch_floor: number;
    readonly start_time: number;
    readonly time_step: number;
}

/**
 * Sound type for audio data - WASM wrapper
 */
export class Sound {
    free(): void;
    [Symbol.dispose](): void;
    /**
     * Create silence
     */
    static create_silence(duration: number, sample_rate: number): Sound;
    /**
     * Create a pure tone (sine wave)
     */
    static create_tone(frequency: number, duration: number, sample_rate: number, amplitude: number, phase: number): Sound;
    /**
     * Apply de-emphasis filter
     */
    de_emphasis(from_frequency: number): Sound;
    /**
     * Get the sample value at a specific time using linear interpolation
     */
    get_value_at_time(time: number): number | undefined;
    /**
     * Create a Sound from raw samples
     *
     * @param samples - Float64Array of audio samples
     * @param sample_rate - Sample rate in Hz
     */
    constructor(samples: Float64Array, sample_rate: number);
    /**
     * Get the peak amplitude
     */
    peak(): number;
    /**
     * Apply pre-emphasis filter
     */
    pre_emphasis(from_frequency: number): Sound;
    /**
     * Resample to a new sample rate.
     *
     * Uses Praat's FFT-based windowed-sinc algorithm. If `new_sample_rate` is
     * greater than or equal to the current rate, returns a copy without
     * upsampling (matching Praat's Sound_resample behavior).
     *
     * @param new_sample_rate - Target sample rate in Hz
     */
    resample(new_sample_rate: number): Sound;
    /**
     * Get the root-mean-square amplitude
     */
    rms(): number;
    /**
     * Get the audio samples as a Float64Array
     */
    samples(): Float64Array;
    /**
     * Compute formants using Burg's LPC method
     *
     * @param time_step - Time between analysis frames (0.0 for automatic)
     * @param max_num_formants - Maximum number of formants (typically 5)
     * @param max_formant_hz - Maximum formant frequency (Hz)
     * @param window_length - Analysis window duration (typically 0.025)
     * @param pre_emphasis_from - Pre-emphasis frequency (Hz), typically 50
     */
    to_formant_burg(time_step: number, max_num_formants: number, max_formant_hz: number, window_length: number, pre_emphasis_from: number): Formant;
    /**
     * Compute Burg-LPC formants for a bundle of ceilings.
     *
     * Returns a JS array of `Formant` objects (one per ceiling) in input order.
     * This is the same analysis that powers `to_formant_path_burg` internally
     * but without the path wrapper.
     *
     * @param time_step - Time between frames (0.0 for automatic)
     * @param max_num_formants - Maximum formants (typically 5)
     * @param max_formants_hz - Float64Array of ceiling frequencies
     * @param window_length - Analysis window (typically 0.025)
     * @param pre_emphasis_from - Pre-emphasis (typically 50)
     */
    to_formant_burg_multi(time_step: number, max_num_formants: number, max_formants_hz: Float64Array, window_length: number, pre_emphasis_from: number): Array<any>;
    /**
     * Compute a FormantPath (multi-ceiling Burg analysis with Viterbi path
     * selection). Mirrors Praat's `Sound: To FormantPath (burg)`.
     *
     * References:
     *   Boersma & Weenink — Praat: doing phonetics by computer.
     *   Weenink — FormantPath (Praat: LPC/FormantPath.cpp, GPL-3).
     *   Jadoul, Thompson & de Boer (2018) — Parselmouth. J. Phonetics 71, 1–15.
     *
     * @param time_step - Time between frames (seconds). Typically 0.005.
     * @param max_num_formants - Maximum formants (typically 5).
     * @param middle_formant_ceiling - Center ceiling (Hz).
     * @param window_length - Window duration (typically 0.025).
     * @param pre_emphasis_from - Pre-emphasis (Hz, typically 50).
     * @param ceiling_step_size - Log step (typically 0.05).
     * @param number_of_steps_up_down - Ceilings = 2*N+1.
     */
    to_formant_path_burg(time_step: number, max_num_formants: number, middle_formant_ceiling: number, window_length: number, pre_emphasis_from: number, ceiling_step_size: number, number_of_steps_up_down: number): FormantPath;
    /**
     * Compute harmonicity using autocorrelation method
     */
    to_harmonicity_ac(time_step: number, min_pitch: number, silence_threshold: number, periods_per_window: number): Harmonicity;
    /**
     * Compute harmonicity using cross-correlation method
     */
    to_harmonicity_cc(time_step: number, min_pitch: number, silence_threshold: number, periods_per_window: number): Harmonicity;
    /**
     * Compute intensity contour
     *
     * @param min_pitch - Minimum expected pitch (Hz)
     * @param time_step - Time between frames (0.0 for automatic)
     */
    to_intensity(min_pitch: number, time_step: number): Intensity;
    /**
     * Compute pitch contour
     *
     * @param time_step - Time between analysis frames (0.0 for automatic)
     * @param pitch_floor - Minimum pitch (Hz), typically 75
     * @param pitch_ceiling - Maximum pitch (Hz), typically 600
     */
    to_pitch(time_step: number, pitch_floor: number, pitch_ceiling: number): Pitch;
    /**
     * Compute pitch contour using the cross-correlation (FCC) method
     *
     * Equivalent to Praat's "Sound: To Pitch (cc)". Same Viterbi path-finder
     * as `to_pitch`, but with FCC candidate scoring.
     *
     * @param time_step - Time between analysis frames (0.0 for automatic)
     * @param pitch_floor - Minimum pitch (Hz), typically 75
     * @param pitch_ceiling - Maximum pitch (Hz), typically 600
     */
    to_pitch_cc(time_step: number, pitch_floor: number, pitch_ceiling: number): Pitch;
    /**
     * Compute spectrogram
     *
     * @param effective_analysis_width - Effective window duration (seconds)
     * @param max_frequency - Maximum frequency (Hz), 0 for Nyquist
     * @param time_step - Time between frames (0 for automatic)
     * @param frequency_step - Frequency resolution (Hz), 0 for automatic
     * @param window_shape - Window function: "gaussian", "hanning", "hamming", "rectangular"
     */
    to_spectrogram(effective_analysis_width: number, max_frequency: number, time_step: number, frequency_step: number, window_shape: string): Spectrogram;
    /**
     * Compute spectrum (single-frame FFT)
     *
     * @param fast - If true, use power-of-2 FFT size
     */
    to_spectrum(fast: boolean): Spectrum;
    /**
     * Get the total duration in seconds
     */
    readonly duration: number;
    /**
     * Get the end time
     */
    readonly end_time: number;
    /**
     * Get the number of samples
     */
    readonly num_samples: number;
    /**
     * Get the sample rate in Hz
     */
    readonly sample_rate: number;
    /**
     * Get the start time
     */
    readonly start_time: number;
}

/**
 * Spectrogram - WASM wrapper
 */
export class Spectrogram {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    get_frequency_from_bin(bin: number): number;
    get_time_from_frame(frame: number): number;
    /**
     * Get the spectrogram data as a flat Float64Array
     * Data is in row-major order [freq_0_time_0, freq_0_time_1, ..., freq_n_time_m]
     */
    values(): Float64Array;
    readonly freq_max: number;
    readonly freq_min: number;
    readonly freq_step: number;
    readonly num_frames: number;
    readonly num_freq_bins: number;
    readonly start_time: number;
    readonly time_step: number;
}

/**
 * Spectrum - WASM wrapper
 */
export class Spectrum {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    /**
     * Get band energy between two frequencies
     */
    get_band_energy(freq_min: number, freq_max: number): number;
    /**
     * Get spectral center of gravity
     */
    get_center_of_gravity(power: number): number;
    /**
     * Get spectral kurtosis
     */
    get_kurtosis(power: number): number;
    /**
     * Get spectral skewness
     */
    get_skewness(power: number): number;
    /**
     * Get spectral standard deviation
     */
    get_standard_deviation(power: number): number;
    /**
     * Get total energy
     */
    get_total_energy(): number;
    readonly df: number;
    readonly max_frequency: number;
    readonly num_bins: number;
}

/**
 * Initialize panic hook for better error messages in WASM
 */
export function main(): void;

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly __wbg_formant_free: (a: number, b: number) => void;
    readonly __wbg_formantpath_free: (a: number, b: number) => void;
    readonly __wbg_harmonicity_free: (a: number, b: number) => void;
    readonly __wbg_intensity_free: (a: number, b: number) => void;
    readonly __wbg_pitch_free: (a: number, b: number) => void;
    readonly __wbg_sound_free: (a: number, b: number) => void;
    readonly __wbg_spectrogram_free: (a: number, b: number) => void;
    readonly __wbg_spectrum_free: (a: number, b: number) => void;
    readonly formant_bandwidth_values: (a: number, b: number) => any;
    readonly formant_formant_values: (a: number, b: number) => any;
    readonly formant_get_bandwidth_at_time: (a: number, b: number, c: number, d: number, e: number, f: number, g: number) => [number, number, number, number];
    readonly formant_get_time_from_frame: (a: number, b: number) => number;
    readonly formant_get_value_at_time: (a: number, b: number, c: number, d: number, e: number, f: number, g: number) => [number, number, number, number];
    readonly formant_max_num_formants: (a: number) => number;
    readonly formant_num_frames: (a: number) => number;
    readonly formant_start_time: (a: number) => number;
    readonly formant_time_step: (a: number) => number;
    readonly formant_times: (a: number) => any;
    readonly formantpath_candidate: (a: number, b: number) => [number, number, number];
    readonly formantpath_ceilings: (a: number) => any;
    readonly formantpath_extract_formant: (a: number) => number;
    readonly formantpath_get_stress_of_candidate: (a: number, b: number, c: number, d: number, e: number, f: any, g: number, h: number) => [number, number, number];
    readonly formantpath_get_stress_of_candidates: (a: number, b: number, c: number, d: number, e: number, f: any, g: number) => any;
    readonly formantpath_num_candidates: (a: number) => number;
    readonly formantpath_num_frames: (a: number) => number;
    readonly formantpath_path: (a: number) => any;
    readonly formantpath_path_finder: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: any, i: number) => void;
    readonly formantpath_set_path: (a: number, b: number, c: number, d: number) => [number, number];
    readonly formantpath_start_time: (a: number) => number;
    readonly formantpath_time_step: (a: number) => number;
    readonly harmonicity_get_value_at_time: (a: number, b: number, c: number, d: number) => [number, number, number, number];
    readonly harmonicity_max: (a: number) => [number, number];
    readonly harmonicity_mean: (a: number) => [number, number];
    readonly harmonicity_min: (a: number) => [number, number];
    readonly harmonicity_num_frames: (a: number) => number;
    readonly harmonicity_start_time: (a: number) => number;
    readonly harmonicity_time_step: (a: number) => number;
    readonly harmonicity_times: (a: number) => any;
    readonly harmonicity_values: (a: number) => any;
    readonly intensity_get_value_at_time: (a: number, b: number, c: number, d: number) => [number, number, number, number];
    readonly intensity_max: (a: number) => [number, number];
    readonly intensity_mean: (a: number) => [number, number];
    readonly intensity_min: (a: number) => [number, number];
    readonly intensity_num_frames: (a: number) => number;
    readonly intensity_start_time: (a: number) => number;
    readonly intensity_time_step: (a: number) => number;
    readonly intensity_times: (a: number) => any;
    readonly intensity_values: (a: number) => any;
    readonly main: () => void;
    readonly pitch_get_time_from_frame: (a: number, b: number) => number;
    readonly pitch_get_value_at_time: (a: number, b: number, c: number, d: number, e: number, f: number) => [number, number, number, number];
    readonly pitch_num_frames: (a: number) => number;
    readonly pitch_pitch_ceiling: (a: number) => number;
    readonly pitch_pitch_floor: (a: number) => number;
    readonly pitch_start_time: (a: number) => number;
    readonly pitch_strengths: (a: number) => any;
    readonly pitch_time_step: (a: number) => number;
    readonly pitch_times: (a: number) => any;
    readonly pitch_values: (a: number) => any;
    readonly sound_create_silence: (a: number, b: number) => number;
    readonly sound_create_tone: (a: number, b: number, c: number, d: number, e: number) => number;
    readonly sound_de_emphasis: (a: number, b: number) => number;
    readonly sound_duration: (a: number) => number;
    readonly sound_end_time: (a: number) => number;
    readonly sound_get_value_at_time: (a: number, b: number) => [number, number];
    readonly sound_new: (a: any, b: number) => number;
    readonly sound_num_samples: (a: number) => number;
    readonly sound_peak: (a: number) => number;
    readonly sound_pre_emphasis: (a: number, b: number) => number;
    readonly sound_resample: (a: number, b: number) => number;
    readonly sound_rms: (a: number) => number;
    readonly sound_sample_rate: (a: number) => number;
    readonly sound_samples: (a: number) => any;
    readonly sound_start_time: (a: number) => number;
    readonly sound_to_formant_burg: (a: number, b: number, c: number, d: number, e: number, f: number) => number;
    readonly sound_to_formant_burg_multi: (a: number, b: number, c: number, d: any, e: number, f: number) => any;
    readonly sound_to_formant_path_burg: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number) => number;
    readonly sound_to_harmonicity_ac: (a: number, b: number, c: number, d: number, e: number) => number;
    readonly sound_to_harmonicity_cc: (a: number, b: number, c: number, d: number, e: number) => number;
    readonly sound_to_intensity: (a: number, b: number, c: number) => number;
    readonly sound_to_pitch: (a: number, b: number, c: number, d: number) => number;
    readonly sound_to_pitch_cc: (a: number, b: number, c: number, d: number) => number;
    readonly sound_to_spectrogram: (a: number, b: number, c: number, d: number, e: number, f: number, g: number) => [number, number, number];
    readonly sound_to_spectrum: (a: number, b: number) => number;
    readonly spectrogram_freq_max: (a: number) => number;
    readonly spectrogram_freq_min: (a: number) => number;
    readonly spectrogram_freq_step: (a: number) => number;
    readonly spectrogram_get_frequency_from_bin: (a: number, b: number) => number;
    readonly spectrogram_get_time_from_frame: (a: number, b: number) => number;
    readonly spectrogram_num_frames: (a: number) => number;
    readonly spectrogram_num_freq_bins: (a: number) => number;
    readonly spectrogram_start_time: (a: number) => number;
    readonly spectrogram_time_step: (a: number) => number;
    readonly spectrogram_values: (a: number) => any;
    readonly spectrum_df: (a: number) => number;
    readonly spectrum_get_band_energy: (a: number, b: number, c: number) => number;
    readonly spectrum_get_center_of_gravity: (a: number, b: number) => number;
    readonly spectrum_get_kurtosis: (a: number, b: number) => number;
    readonly spectrum_get_skewness: (a: number, b: number) => number;
    readonly spectrum_get_standard_deviation: (a: number, b: number) => number;
    readonly spectrum_get_total_energy: (a: number) => number;
    readonly spectrum_max_frequency: (a: number) => number;
    readonly spectrum_num_bins: (a: number) => number;
    readonly __wbindgen_free: (a: number, b: number, c: number) => void;
    readonly __wbindgen_malloc: (a: number, b: number) => number;
    readonly __wbindgen_realloc: (a: number, b: number, c: number, d: number) => number;
    readonly __wbindgen_externrefs: WebAssembly.Table;
    readonly __externref_table_dealloc: (a: number) => void;
    readonly __wbindgen_start: () => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;

/**
 * Instantiates the given `module`, which can either be bytes or
 * a precompiled `WebAssembly.Module`.
 *
 * @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
 *
 * @returns {InitOutput}
 */
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
 * If `module_or_path` is {RequestInfo} or {URL}, makes a request and
 * for everything else, calls `WebAssembly.instantiate` directly.
 *
 * @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
 *
 * @returns {Promise<InitOutput>}
 */
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
