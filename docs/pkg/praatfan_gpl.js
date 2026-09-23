/* @ts-self-types="./praatfan_gpl.d.ts" */

/**
 * Formant contour - WASM wrapper
 */
export class Formant {
    static __wrap(ptr) {
        const obj = Object.create(Formant.prototype);
        obj.__wbg_ptr = ptr;
        FormantFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        FormantFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_formant_free(ptr, 0);
    }
    /**
     * Get all bandwidths for a specific formant as Float64Array
     * @param {number} formant_number
     * @returns {Float64Array}
     */
    bandwidth_values(formant_number) {
        const ret = wasm.formant_bandwidth_values(this.__wbg_ptr, formant_number);
        return ret;
    }
    /**
     * Get all values for a specific formant as Float64Array
     * @param {number} formant_number
     * @returns {Float64Array}
     */
    formant_values(formant_number) {
        const ret = wasm.formant_formant_values(this.__wbg_ptr, formant_number);
        return ret;
    }
    /**
     * Get bandwidth at a specific time
     * @param {number} formant_number
     * @param {number} time
     * @param {string} unit
     * @param {string} interpolation
     * @returns {number | undefined}
     */
    get_bandwidth_at_time(formant_number, time, unit, interpolation) {
        const ptr0 = passStringToWasm0(unit, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passStringToWasm0(interpolation, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len1 = WASM_VECTOR_LEN;
        const ret = wasm.formant_get_bandwidth_at_time(this.__wbg_ptr, formant_number, time, ptr0, len0, ptr1, len1);
        if (ret[3]) {
            throw takeFromExternrefTable0(ret[2]);
        }
        return ret[0] === 0 ? undefined : ret[1];
    }
    /**
     * Get the time of a specific frame
     * @param {number} frame
     * @returns {number}
     */
    get_time_from_frame(frame) {
        const ret = wasm.formant_get_time_from_frame(this.__wbg_ptr, frame);
        return ret;
    }
    /**
     * Get formant value at a specific time
     *
     * @param formant_number - Formant number (1 for F1, 2 for F2, etc.)
     * @param time - Time to query
     * @param unit - Unit: "hertz", "bark", "mel", "erb"
     * @param interpolation - Interpolation: "nearest", "linear", "cubic"
     * @param {number} formant_number
     * @param {number} time
     * @param {string} unit
     * @param {string} interpolation
     * @returns {number | undefined}
     */
    get_value_at_time(formant_number, time, unit, interpolation) {
        const ptr0 = passStringToWasm0(unit, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passStringToWasm0(interpolation, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len1 = WASM_VECTOR_LEN;
        const ret = wasm.formant_get_value_at_time(this.__wbg_ptr, formant_number, time, ptr0, len0, ptr1, len1);
        if (ret[3]) {
            throw takeFromExternrefTable0(ret[2]);
        }
        return ret[0] === 0 ? undefined : ret[1];
    }
    /**
     * @returns {number}
     */
    get max_num_formants() {
        const ret = wasm.formant_max_num_formants(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    get num_frames() {
        const ret = wasm.formant_num_frames(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    get start_time() {
        const ret = wasm.formant_start_time(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {number}
     */
    get time_step() {
        const ret = wasm.formant_time_step(this.__wbg_ptr);
        return ret;
    }
    /**
     * Get all frame times as Float64Array
     * @returns {Float64Array}
     */
    times() {
        const ret = wasm.formant_times(this.__wbg_ptr);
        return ret;
    }
}
if (Symbol.dispose) Formant.prototype[Symbol.dispose] = Formant.prototype.free;

/**
 * FormantPath - multi-ceiling Burg analysis with Viterbi path selection.
 *
 * Mirrors Praat's FormantPath. Produced by `Sound.to_formant_path_burg`.
 */
export class FormantPath {
    static __wrap(ptr) {
        const obj = Object.create(FormantPath.prototype);
        obj.__wbg_ptr = ptr;
        FormantPathFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        FormantPathFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_formantpath_free(ptr, 0);
    }
    /**
     * Access one candidate Formant by 0-based index.
     * @param {number} index
     * @returns {Formant}
     */
    candidate(index) {
        const ret = wasm.formantpath_candidate(this.__wbg_ptr, index);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return Formant.__wrap(ret[0]);
    }
    /**
     * Ceiling frequencies in Hz as a Float64Array.
     * @returns {Float64Array}
     */
    ceilings() {
        const ret = wasm.formantpath_ceilings(this.__wbg_ptr);
        return ret;
    }
    /**
     * Build a Formant by selecting, at each frame, the candidate indicated by
     * the path.
     * @returns {Formant}
     */
    extract_formant() {
        const ret = wasm.formantpath_extract_formant(this.__wbg_ptr);
        return Formant.__wrap(ret);
    }
    /**
     * Stress of one candidate over `[t_min, t_max]`.
     * @param {number} t_min
     * @param {number} t_max
     * @param {number} from_formant
     * @param {number} to_formant
     * @param {Int32Array} parameters
     * @param {number} power
     * @param {number} candidate
     * @returns {number}
     */
    get_stress_of_candidate(t_min, t_max, from_formant, to_formant, parameters, power, candidate) {
        const ret = wasm.formantpath_get_stress_of_candidate(this.__wbg_ptr, t_min, t_max, from_formant, to_formant, parameters, power, candidate);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return ret[0];
    }
    /**
     * Stress for every candidate as a Float64Array.
     * @param {number} t_min
     * @param {number} t_max
     * @param {number} from_formant
     * @param {number} to_formant
     * @param {Int32Array} parameters
     * @param {number} power
     * @returns {Float64Array}
     */
    get_stress_of_candidates(t_min, t_max, from_formant, to_formant, parameters, power) {
        const ret = wasm.formantpath_get_stress_of_candidates(this.__wbg_ptr, t_min, t_max, from_formant, to_formant, parameters, power);
        return ret;
    }
    /**
     * Number of candidate analyses (= 2*number_of_steps_up_down+1).
     * @returns {number}
     */
    get num_candidates() {
        const ret = wasm.formantpath_num_candidates(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    get num_frames() {
        const ret = wasm.formantpath_num_frames(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * Current per-frame selected candidate (0-based).
     * @returns {Uint32Array}
     */
    path() {
        const ret = wasm.formantpath_path(this.__wbg_ptr);
        return ret;
    }
    /**
     * Run Viterbi-optimal path selection (Praat's `Path finder`).
     *
     * @param parameters - Int32Array of Legendre coefficients per track.
     * @param {number} q_weight
     * @param {number} frequency_change_weight
     * @param {number} stress_weight
     * @param {number} ceiling_change_weight
     * @param {number} intensity_modulation_step_size
     * @param {number} path_window_length
     * @param {Int32Array} parameters
     * @param {number} power
     */
    path_finder(q_weight, frequency_change_weight, stress_weight, ceiling_change_weight, intensity_modulation_step_size, path_window_length, parameters, power) {
        wasm.formantpath_path_finder(this.__wbg_ptr, q_weight, frequency_change_weight, stress_weight, ceiling_change_weight, intensity_modulation_step_size, path_window_length, parameters, power);
    }
    /**
     * Force every frame overlapping `[t_min, t_max]` to select `candidate`
     * (0-based).
     * @param {number} t_min
     * @param {number} t_max
     * @param {number} candidate
     */
    set_path(t_min, t_max, candidate) {
        const ret = wasm.formantpath_set_path(this.__wbg_ptr, t_min, t_max, candidate);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * @returns {number}
     */
    get start_time() {
        const ret = wasm.formantpath_start_time(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {number}
     */
    get time_step() {
        const ret = wasm.formantpath_time_step(this.__wbg_ptr);
        return ret;
    }
}
if (Symbol.dispose) FormantPath.prototype[Symbol.dispose] = FormantPath.prototype.free;

/**
 * Harmonicity (HNR) - WASM wrapper
 */
export class Harmonicity {
    static __wrap(ptr) {
        const obj = Object.create(Harmonicity.prototype);
        obj.__wbg_ptr = ptr;
        HarmonicityFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        HarmonicityFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_harmonicity_free(ptr, 0);
    }
    /**
     * Get HNR value at a specific time
     * @param {number} time
     * @param {string} interpolation
     * @returns {number | undefined}
     */
    get_value_at_time(time, interpolation) {
        const ptr0 = passStringToWasm0(interpolation, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.harmonicity_get_value_at_time(this.__wbg_ptr, time, ptr0, len0);
        if (ret[3]) {
            throw takeFromExternrefTable0(ret[2]);
        }
        return ret[0] === 0 ? undefined : ret[1];
    }
    /**
     * @returns {number | undefined}
     */
    max() {
        const ret = wasm.harmonicity_max(this.__wbg_ptr);
        return ret[0] === 0 ? undefined : ret[1];
    }
    /**
     * @returns {number | undefined}
     */
    mean() {
        const ret = wasm.harmonicity_mean(this.__wbg_ptr);
        return ret[0] === 0 ? undefined : ret[1];
    }
    /**
     * @returns {number | undefined}
     */
    min() {
        const ret = wasm.harmonicity_min(this.__wbg_ptr);
        return ret[0] === 0 ? undefined : ret[1];
    }
    /**
     * @returns {number}
     */
    get num_frames() {
        const ret = wasm.harmonicity_num_frames(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    get start_time() {
        const ret = wasm.harmonicity_start_time(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {number}
     */
    get time_step() {
        const ret = wasm.harmonicity_time_step(this.__wbg_ptr);
        return ret;
    }
    /**
     * Get all frame times as Float64Array
     * @returns {Float64Array}
     */
    times() {
        const ret = wasm.harmonicity_times(this.__wbg_ptr);
        return ret;
    }
    /**
     * Get all HNR values as Float64Array
     * @returns {Float64Array}
     */
    values() {
        const ret = wasm.harmonicity_values(this.__wbg_ptr);
        return ret;
    }
}
if (Symbol.dispose) Harmonicity.prototype[Symbol.dispose] = Harmonicity.prototype.free;

/**
 * Intensity contour - WASM wrapper
 */
export class Intensity {
    static __wrap(ptr) {
        const obj = Object.create(Intensity.prototype);
        obj.__wbg_ptr = ptr;
        IntensityFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        IntensityFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_intensity_free(ptr, 0);
    }
    /**
     * Get intensity value at a specific time
     * @param {number} time
     * @param {string} interpolation
     * @returns {number | undefined}
     */
    get_value_at_time(time, interpolation) {
        const ptr0 = passStringToWasm0(interpolation, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.intensity_get_value_at_time(this.__wbg_ptr, time, ptr0, len0);
        if (ret[3]) {
            throw takeFromExternrefTable0(ret[2]);
        }
        return ret[0] === 0 ? undefined : ret[1];
    }
    /**
     * @returns {number | undefined}
     */
    max() {
        const ret = wasm.intensity_max(this.__wbg_ptr);
        return ret[0] === 0 ? undefined : ret[1];
    }
    /**
     * @returns {number | undefined}
     */
    mean() {
        const ret = wasm.intensity_mean(this.__wbg_ptr);
        return ret[0] === 0 ? undefined : ret[1];
    }
    /**
     * @returns {number | undefined}
     */
    min() {
        const ret = wasm.intensity_min(this.__wbg_ptr);
        return ret[0] === 0 ? undefined : ret[1];
    }
    /**
     * @returns {number}
     */
    get num_frames() {
        const ret = wasm.intensity_num_frames(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    get start_time() {
        const ret = wasm.intensity_start_time(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {number}
     */
    get time_step() {
        const ret = wasm.intensity_time_step(this.__wbg_ptr);
        return ret;
    }
    /**
     * Get all frame times as Float64Array
     * @returns {Float64Array}
     */
    times() {
        const ret = wasm.intensity_times(this.__wbg_ptr);
        return ret;
    }
    /**
     * Get all intensity values as Float64Array
     * @returns {Float64Array}
     */
    values() {
        const ret = wasm.intensity_values(this.__wbg_ptr);
        return ret;
    }
}
if (Symbol.dispose) Intensity.prototype[Symbol.dispose] = Intensity.prototype.free;

/**
 * Pitch contour - WASM wrapper
 */
export class Pitch {
    static __wrap(ptr) {
        const obj = Object.create(Pitch.prototype);
        obj.__wbg_ptr = ptr;
        PitchFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        PitchFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_pitch_free(ptr, 0);
    }
    /**
     * Get the time of a specific frame
     * @param {number} frame
     * @returns {number}
     */
    get_time_from_frame(frame) {
        const ret = wasm.pitch_get_time_from_frame(this.__wbg_ptr, frame);
        return ret;
    }
    /**
     * Get pitch value at a specific time
     *
     * @param time - Time to query
     * @param unit - Unit: "hertz", "mel", "semitones", "erb"
     * @param interpolation - Interpolation: "nearest", "linear", "cubic"
     * @returns Pitch value or undefined if unvoiced
     * @param {number} time
     * @param {string} unit
     * @param {string} interpolation
     * @returns {number | undefined}
     */
    get_value_at_time(time, unit, interpolation) {
        const ptr0 = passStringToWasm0(unit, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passStringToWasm0(interpolation, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len1 = WASM_VECTOR_LEN;
        const ret = wasm.pitch_get_value_at_time(this.__wbg_ptr, time, ptr0, len0, ptr1, len1);
        if (ret[3]) {
            throw takeFromExternrefTable0(ret[2]);
        }
        return ret[0] === 0 ? undefined : ret[1];
    }
    /**
     * @returns {number}
     */
    get num_frames() {
        const ret = wasm.pitch_num_frames(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    get pitch_ceiling() {
        const ret = wasm.pitch_pitch_ceiling(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {number}
     */
    get pitch_floor() {
        const ret = wasm.pitch_pitch_floor(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {number}
     */
    get start_time() {
        const ret = wasm.pitch_start_time(this.__wbg_ptr);
        return ret;
    }
    /**
     * Per-frame strength of the Viterbi-selected candidate.
     *
     * Equivalent to parselmouth's `Pitch.selected_array['strength']`:
     * voiced frames return the winning candidate's autocorrelation strength,
     * unvoiced frames return the computed unvoiced-candidate penalty strength.
     * Returned as-is (no NaN handling) for Praat bit-parity.
     * @returns {Float64Array}
     */
    strengths() {
        const ret = wasm.pitch_strengths(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {number}
     */
    get time_step() {
        const ret = wasm.pitch_time_step(this.__wbg_ptr);
        return ret;
    }
    /**
     * Get all frame times as Float64Array
     * @returns {Float64Array}
     */
    times() {
        const ret = wasm.pitch_times(this.__wbg_ptr);
        return ret;
    }
    /**
     * Get all pitch values as Float64Array (NaN for unvoiced)
     * @returns {Float64Array}
     */
    values() {
        const ret = wasm.pitch_values(this.__wbg_ptr);
        return ret;
    }
}
if (Symbol.dispose) Pitch.prototype[Symbol.dispose] = Pitch.prototype.free;

/**
 * Sound type for audio data - WASM wrapper
 */
export class Sound {
    static __wrap(ptr) {
        const obj = Object.create(Sound.prototype);
        obj.__wbg_ptr = ptr;
        SoundFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        SoundFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_sound_free(ptr, 0);
    }
    /**
     * Create silence
     * @param {number} duration
     * @param {number} sample_rate
     * @returns {Sound}
     */
    static create_silence(duration, sample_rate) {
        const ret = wasm.sound_create_silence(duration, sample_rate);
        return Sound.__wrap(ret);
    }
    /**
     * Create a pure tone (sine wave)
     * @param {number} frequency
     * @param {number} duration
     * @param {number} sample_rate
     * @param {number} amplitude
     * @param {number} phase
     * @returns {Sound}
     */
    static create_tone(frequency, duration, sample_rate, amplitude, phase) {
        const ret = wasm.sound_create_tone(frequency, duration, sample_rate, amplitude, phase);
        return Sound.__wrap(ret);
    }
    /**
     * Apply de-emphasis filter
     * @param {number} from_frequency
     * @returns {Sound}
     */
    de_emphasis(from_frequency) {
        const ret = wasm.sound_de_emphasis(this.__wbg_ptr, from_frequency);
        return Sound.__wrap(ret);
    }
    /**
     * Get the total duration in seconds
     * @returns {number}
     */
    get duration() {
        const ret = wasm.sound_duration(this.__wbg_ptr);
        return ret;
    }
    /**
     * Get the end time
     * @returns {number}
     */
    get end_time() {
        const ret = wasm.sound_end_time(this.__wbg_ptr);
        return ret;
    }
    /**
     * Get the sample value at a specific time using linear interpolation
     * @param {number} time
     * @returns {number | undefined}
     */
    get_value_at_time(time) {
        const ret = wasm.sound_get_value_at_time(this.__wbg_ptr, time);
        return ret[0] === 0 ? undefined : ret[1];
    }
    /**
     * Create a Sound from raw samples
     *
     * @param samples - Float64Array of audio samples
     * @param sample_rate - Sample rate in Hz
     * @param {Float64Array} samples
     * @param {number} sample_rate
     */
    constructor(samples, sample_rate) {
        const ret = wasm.sound_new(samples, sample_rate);
        this.__wbg_ptr = ret;
        SoundFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * Get the number of samples
     * @returns {number}
     */
    get num_samples() {
        const ret = wasm.sound_num_samples(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * Get the peak amplitude
     * @returns {number}
     */
    peak() {
        const ret = wasm.sound_peak(this.__wbg_ptr);
        return ret;
    }
    /**
     * Apply pre-emphasis filter
     * @param {number} from_frequency
     * @returns {Sound}
     */
    pre_emphasis(from_frequency) {
        const ret = wasm.sound_pre_emphasis(this.__wbg_ptr, from_frequency);
        return Sound.__wrap(ret);
    }
    /**
     * Resample to a new sample rate.
     *
     * Uses Praat's FFT-based windowed-sinc algorithm. If `new_sample_rate` is
     * greater than or equal to the current rate, returns a copy without
     * upsampling (matching Praat's Sound_resample behavior).
     *
     * @param new_sample_rate - Target sample rate in Hz
     * @param {number} new_sample_rate
     * @returns {Sound}
     */
    resample(new_sample_rate) {
        const ret = wasm.sound_resample(this.__wbg_ptr, new_sample_rate);
        return Sound.__wrap(ret);
    }
    /**
     * Get the root-mean-square amplitude
     * @returns {number}
     */
    rms() {
        const ret = wasm.sound_rms(this.__wbg_ptr);
        return ret;
    }
    /**
     * Get the sample rate in Hz
     * @returns {number}
     */
    get sample_rate() {
        const ret = wasm.sound_sample_rate(this.__wbg_ptr);
        return ret;
    }
    /**
     * Get the audio samples as a Float64Array
     * @returns {Float64Array}
     */
    samples() {
        const ret = wasm.sound_samples(this.__wbg_ptr);
        return ret;
    }
    /**
     * Get the start time
     * @returns {number}
     */
    get start_time() {
        const ret = wasm.sound_start_time(this.__wbg_ptr);
        return ret;
    }
    /**
     * Compute formants using Burg's LPC method
     *
     * @param time_step - Time between analysis frames (0.0 for automatic)
     * @param max_num_formants - Maximum number of formants (typically 5)
     * @param max_formant_hz - Maximum formant frequency (Hz)
     * @param window_length - Analysis window duration (typically 0.025)
     * @param pre_emphasis_from - Pre-emphasis frequency (Hz), typically 50
     * @param {number} time_step
     * @param {number} max_num_formants
     * @param {number} max_formant_hz
     * @param {number} window_length
     * @param {number} pre_emphasis_from
     * @returns {Formant}
     */
    to_formant_burg(time_step, max_num_formants, max_formant_hz, window_length, pre_emphasis_from) {
        const ret = wasm.sound_to_formant_burg(this.__wbg_ptr, time_step, max_num_formants, max_formant_hz, window_length, pre_emphasis_from);
        return Formant.__wrap(ret);
    }
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
     * @param {number} time_step
     * @param {number} max_num_formants
     * @param {Float64Array} max_formants_hz
     * @param {number} window_length
     * @param {number} pre_emphasis_from
     * @returns {Array<any>}
     */
    to_formant_burg_multi(time_step, max_num_formants, max_formants_hz, window_length, pre_emphasis_from) {
        const ret = wasm.sound_to_formant_burg_multi(this.__wbg_ptr, time_step, max_num_formants, max_formants_hz, window_length, pre_emphasis_from);
        return ret;
    }
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
     * @param {number} time_step
     * @param {number} max_num_formants
     * @param {number} middle_formant_ceiling
     * @param {number} window_length
     * @param {number} pre_emphasis_from
     * @param {number} ceiling_step_size
     * @param {number} number_of_steps_up_down
     * @returns {FormantPath}
     */
    to_formant_path_burg(time_step, max_num_formants, middle_formant_ceiling, window_length, pre_emphasis_from, ceiling_step_size, number_of_steps_up_down) {
        const ret = wasm.sound_to_formant_path_burg(this.__wbg_ptr, time_step, max_num_formants, middle_formant_ceiling, window_length, pre_emphasis_from, ceiling_step_size, number_of_steps_up_down);
        return FormantPath.__wrap(ret);
    }
    /**
     * Compute harmonicity using autocorrelation method
     * @param {number} time_step
     * @param {number} min_pitch
     * @param {number} silence_threshold
     * @param {number} periods_per_window
     * @returns {Harmonicity}
     */
    to_harmonicity_ac(time_step, min_pitch, silence_threshold, periods_per_window) {
        const ret = wasm.sound_to_harmonicity_ac(this.__wbg_ptr, time_step, min_pitch, silence_threshold, periods_per_window);
        return Harmonicity.__wrap(ret);
    }
    /**
     * Compute harmonicity using cross-correlation method
     * @param {number} time_step
     * @param {number} min_pitch
     * @param {number} silence_threshold
     * @param {number} periods_per_window
     * @returns {Harmonicity}
     */
    to_harmonicity_cc(time_step, min_pitch, silence_threshold, periods_per_window) {
        const ret = wasm.sound_to_harmonicity_cc(this.__wbg_ptr, time_step, min_pitch, silence_threshold, periods_per_window);
        return Harmonicity.__wrap(ret);
    }
    /**
     * Compute intensity contour
     *
     * @param min_pitch - Minimum expected pitch (Hz)
     * @param time_step - Time between frames (0.0 for automatic)
     * @param {number} min_pitch
     * @param {number} time_step
     * @returns {Intensity}
     */
    to_intensity(min_pitch, time_step) {
        const ret = wasm.sound_to_intensity(this.__wbg_ptr, min_pitch, time_step);
        return Intensity.__wrap(ret);
    }
    /**
     * Compute pitch contour
     *
     * @param time_step - Time between analysis frames (0.0 for automatic)
     * @param pitch_floor - Minimum pitch (Hz), typically 75
     * @param pitch_ceiling - Maximum pitch (Hz), typically 600
     * @param {number} time_step
     * @param {number} pitch_floor
     * @param {number} pitch_ceiling
     * @returns {Pitch}
     */
    to_pitch(time_step, pitch_floor, pitch_ceiling) {
        const ret = wasm.sound_to_pitch(this.__wbg_ptr, time_step, pitch_floor, pitch_ceiling);
        return Pitch.__wrap(ret);
    }
    /**
     * Compute pitch contour using the cross-correlation (FCC) method
     *
     * Equivalent to Praat's "Sound: To Pitch (cc)". Same Viterbi path-finder
     * as `to_pitch`, but with FCC candidate scoring.
     *
     * @param time_step - Time between analysis frames (0.0 for automatic)
     * @param pitch_floor - Minimum pitch (Hz), typically 75
     * @param pitch_ceiling - Maximum pitch (Hz), typically 600
     * @param {number} time_step
     * @param {number} pitch_floor
     * @param {number} pitch_ceiling
     * @returns {Pitch}
     */
    to_pitch_cc(time_step, pitch_floor, pitch_ceiling) {
        const ret = wasm.sound_to_pitch_cc(this.__wbg_ptr, time_step, pitch_floor, pitch_ceiling);
        return Pitch.__wrap(ret);
    }
    /**
     * Compute spectrogram
     *
     * @param effective_analysis_width - Effective window duration (seconds)
     * @param max_frequency - Maximum frequency (Hz), 0 for Nyquist
     * @param time_step - Time between frames (0 for automatic)
     * @param frequency_step - Frequency resolution (Hz), 0 for automatic
     * @param window_shape - Window function: "gaussian", "hanning", "hamming", "rectangular"
     * @param {number} effective_analysis_width
     * @param {number} max_frequency
     * @param {number} time_step
     * @param {number} frequency_step
     * @param {string} window_shape
     * @returns {Spectrogram}
     */
    to_spectrogram(effective_analysis_width, max_frequency, time_step, frequency_step, window_shape) {
        const ptr0 = passStringToWasm0(window_shape, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.sound_to_spectrogram(this.__wbg_ptr, effective_analysis_width, max_frequency, time_step, frequency_step, ptr0, len0);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return Spectrogram.__wrap(ret[0]);
    }
    /**
     * Compute spectrum (single-frame FFT)
     *
     * @param fast - If true, use power-of-2 FFT size
     * @param {boolean} fast
     * @returns {Spectrum}
     */
    to_spectrum(fast) {
        const ret = wasm.sound_to_spectrum(this.__wbg_ptr, fast);
        return Spectrum.__wrap(ret);
    }
}
if (Symbol.dispose) Sound.prototype[Symbol.dispose] = Sound.prototype.free;

/**
 * Spectrogram - WASM wrapper
 */
export class Spectrogram {
    static __wrap(ptr) {
        const obj = Object.create(Spectrogram.prototype);
        obj.__wbg_ptr = ptr;
        SpectrogramFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        SpectrogramFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_spectrogram_free(ptr, 0);
    }
    /**
     * @returns {number}
     */
    get freq_max() {
        const ret = wasm.spectrogram_freq_max(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {number}
     */
    get freq_min() {
        const ret = wasm.spectrogram_freq_min(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {number}
     */
    get freq_step() {
        const ret = wasm.spectrogram_freq_step(this.__wbg_ptr);
        return ret;
    }
    /**
     * @param {number} bin
     * @returns {number}
     */
    get_frequency_from_bin(bin) {
        const ret = wasm.spectrogram_get_frequency_from_bin(this.__wbg_ptr, bin);
        return ret;
    }
    /**
     * @param {number} frame
     * @returns {number}
     */
    get_time_from_frame(frame) {
        const ret = wasm.spectrogram_get_time_from_frame(this.__wbg_ptr, frame);
        return ret;
    }
    /**
     * @returns {number}
     */
    get num_frames() {
        const ret = wasm.spectrogram_num_frames(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    get num_freq_bins() {
        const ret = wasm.spectrogram_num_freq_bins(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    get start_time() {
        const ret = wasm.spectrogram_start_time(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {number}
     */
    get time_step() {
        const ret = wasm.spectrogram_time_step(this.__wbg_ptr);
        return ret;
    }
    /**
     * Get the spectrogram data as a flat Float64Array
     * Data is in row-major order [freq_0_time_0, freq_0_time_1, ..., freq_n_time_m]
     * @returns {Float64Array}
     */
    values() {
        const ret = wasm.spectrogram_values(this.__wbg_ptr);
        return ret;
    }
}
if (Symbol.dispose) Spectrogram.prototype[Symbol.dispose] = Spectrogram.prototype.free;

/**
 * Spectrum - WASM wrapper
 */
export class Spectrum {
    static __wrap(ptr) {
        const obj = Object.create(Spectrum.prototype);
        obj.__wbg_ptr = ptr;
        SpectrumFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        SpectrumFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_spectrum_free(ptr, 0);
    }
    /**
     * @returns {number}
     */
    get df() {
        const ret = wasm.spectrum_df(this.__wbg_ptr);
        return ret;
    }
    /**
     * Get band energy between two frequencies
     * @param {number} freq_min
     * @param {number} freq_max
     * @returns {number}
     */
    get_band_energy(freq_min, freq_max) {
        const ret = wasm.spectrum_get_band_energy(this.__wbg_ptr, freq_min, freq_max);
        return ret;
    }
    /**
     * Get spectral center of gravity
     * @param {number} power
     * @returns {number}
     */
    get_center_of_gravity(power) {
        const ret = wasm.spectrum_get_center_of_gravity(this.__wbg_ptr, power);
        return ret;
    }
    /**
     * Get spectral kurtosis
     * @param {number} power
     * @returns {number}
     */
    get_kurtosis(power) {
        const ret = wasm.spectrum_get_kurtosis(this.__wbg_ptr, power);
        return ret;
    }
    /**
     * Get spectral skewness
     * @param {number} power
     * @returns {number}
     */
    get_skewness(power) {
        const ret = wasm.spectrum_get_skewness(this.__wbg_ptr, power);
        return ret;
    }
    /**
     * Get spectral standard deviation
     * @param {number} power
     * @returns {number}
     */
    get_standard_deviation(power) {
        const ret = wasm.spectrum_get_standard_deviation(this.__wbg_ptr, power);
        return ret;
    }
    /**
     * Get total energy
     * @returns {number}
     */
    get_total_energy() {
        const ret = wasm.spectrum_get_total_energy(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {number}
     */
    get max_frequency() {
        const ret = wasm.spectrum_max_frequency(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {number}
     */
    get num_bins() {
        const ret = wasm.spectrum_num_bins(this.__wbg_ptr);
        return ret >>> 0;
    }
}
if (Symbol.dispose) Spectrum.prototype[Symbol.dispose] = Spectrum.prototype.free;

/**
 * Initialize panic hook for better error messages in WASM
 */
export function main() {
    wasm.main();
}
function __wbg_get_imports() {
    const import0 = {
        __proto__: null,
        __wbg_Error_67e7344beaa85059: function(arg0, arg1) {
            const ret = Error(getStringFromWasm0(arg0, arg1));
            return ret;
        },
        __wbg___wbindgen_throw_5d9e815e6fdf150f: function(arg0, arg1) {
            throw new Error(getStringFromWasm0(arg0, arg1));
        },
        __wbg_error_757e9472f8410341: function(arg0, arg1) {
            let deferred0_0;
            let deferred0_1;
            try {
                deferred0_0 = arg0;
                deferred0_1 = arg1;
                console.error(getStringFromWasm0(arg0, arg1));
            } finally {
                wasm.__wbindgen_free(deferred0_0, deferred0_1, 1);
            }
        },
        __wbg_formant_new: function(arg0) {
            const ret = Formant.__wrap(arg0);
            return ret;
        },
        __wbg_length_598aff98d0c65a59: function(arg0) {
            const ret = arg0.length;
            return ret;
        },
        __wbg_length_fbfce129368ed79c: function(arg0) {
            const ret = arg0.length;
            return ret;
        },
        __wbg_new_227d7c05414eb861: function() {
            const ret = new Error();
            return ret;
        },
        __wbg_new_ffa92086ea89f79c: function() {
            const ret = new Array();
            return ret;
        },
        __wbg_new_from_slice_3b4c7f1456059f80: function(arg0, arg1) {
            const ret = new Float64Array(getArrayF64FromWasm0(arg0, arg1));
            return ret;
        },
        __wbg_new_from_slice_a500ec81601be48f: function(arg0, arg1) {
            const ret = new Uint32Array(getArrayU32FromWasm0(arg0, arg1));
            return ret;
        },
        __wbg_prototypesetcall_5c62be9626638df9: function(arg0, arg1, arg2) {
            Float64Array.prototype.set.call(getArrayF64FromWasm0(arg0, arg1), arg2);
        },
        __wbg_prototypesetcall_e20f3c85f78376cf: function(arg0, arg1, arg2) {
            Int32Array.prototype.set.call(getArrayI32FromWasm0(arg0, arg1), arg2);
        },
        __wbg_push_bfdf956ba476f65b: function(arg0, arg1) {
            const ret = arg0.push(arg1);
            return ret;
        },
        __wbg_stack_3b0d974bbf31e44f: function(arg0, arg1) {
            const ret = arg1.stack;
            const ptr1 = passStringToWasm0(ret, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
            const len1 = WASM_VECTOR_LEN;
            getDataViewMemory0().setInt32(arg0 + 4 * 1, len1, true);
            getDataViewMemory0().setInt32(arg0 + 4 * 0, ptr1, true);
        },
        __wbindgen_init_externref_table: function() {
            const table = wasm.__wbindgen_externrefs;
            const offset = table.grow(4);
            table.set(0, undefined);
            table.set(offset + 0, undefined);
            table.set(offset + 1, null);
            table.set(offset + 2, true);
            table.set(offset + 3, false);
        },
    };
    return {
        __proto__: null,
        "./praatfan_gpl_bg.js": import0,
    };
}

const FormantFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_formant_free(ptr, 1));
const FormantPathFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_formantpath_free(ptr, 1));
const HarmonicityFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_harmonicity_free(ptr, 1));
const IntensityFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_intensity_free(ptr, 1));
const PitchFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_pitch_free(ptr, 1));
const SoundFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_sound_free(ptr, 1));
const SpectrogramFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_spectrogram_free(ptr, 1));
const SpectrumFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_spectrum_free(ptr, 1));

function getArrayF64FromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return getFloat64ArrayMemory0().subarray(ptr / 8, ptr / 8 + len);
}

function getArrayI32FromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return getInt32ArrayMemory0().subarray(ptr / 4, ptr / 4 + len);
}

function getArrayU32FromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return getUint32ArrayMemory0().subarray(ptr / 4, ptr / 4 + len);
}

let cachedDataViewMemory0 = null;
function getDataViewMemory0() {
    if (cachedDataViewMemory0 === null || cachedDataViewMemory0.buffer.detached === true || (cachedDataViewMemory0.buffer.detached === undefined && cachedDataViewMemory0.buffer !== wasm.memory.buffer)) {
        cachedDataViewMemory0 = new DataView(wasm.memory.buffer);
    }
    return cachedDataViewMemory0;
}

let cachedFloat64ArrayMemory0 = null;
function getFloat64ArrayMemory0() {
    if (cachedFloat64ArrayMemory0 === null || cachedFloat64ArrayMemory0.byteLength === 0) {
        cachedFloat64ArrayMemory0 = new Float64Array(wasm.memory.buffer);
    }
    return cachedFloat64ArrayMemory0;
}

let cachedInt32ArrayMemory0 = null;
function getInt32ArrayMemory0() {
    if (cachedInt32ArrayMemory0 === null || cachedInt32ArrayMemory0.byteLength === 0) {
        cachedInt32ArrayMemory0 = new Int32Array(wasm.memory.buffer);
    }
    return cachedInt32ArrayMemory0;
}

function getStringFromWasm0(ptr, len) {
    return decodeText(ptr >>> 0, len);
}

let cachedUint32ArrayMemory0 = null;
function getUint32ArrayMemory0() {
    if (cachedUint32ArrayMemory0 === null || cachedUint32ArrayMemory0.byteLength === 0) {
        cachedUint32ArrayMemory0 = new Uint32Array(wasm.memory.buffer);
    }
    return cachedUint32ArrayMemory0;
}

let cachedUint8ArrayMemory0 = null;
function getUint8ArrayMemory0() {
    if (cachedUint8ArrayMemory0 === null || cachedUint8ArrayMemory0.byteLength === 0) {
        cachedUint8ArrayMemory0 = new Uint8Array(wasm.memory.buffer);
    }
    return cachedUint8ArrayMemory0;
}

function passStringToWasm0(arg, malloc, realloc) {
    if (realloc === undefined) {
        const buf = cachedTextEncoder.encode(arg);
        const ptr = malloc(buf.length, 1) >>> 0;
        getUint8ArrayMemory0().subarray(ptr, ptr + buf.length).set(buf);
        WASM_VECTOR_LEN = buf.length;
        return ptr;
    }

    let len = arg.length;
    let ptr = malloc(len, 1) >>> 0;

    const mem = getUint8ArrayMemory0();

    let offset = 0;

    for (; offset < len; offset++) {
        const code = arg.charCodeAt(offset);
        if (code > 0x7F) break;
        mem[ptr + offset] = code;
    }
    if (offset !== len) {
        if (offset !== 0) {
            arg = arg.slice(offset);
        }
        ptr = realloc(ptr, len, len = offset + arg.length * 3, 1) >>> 0;
        const view = getUint8ArrayMemory0().subarray(ptr + offset, ptr + len);
        const ret = cachedTextEncoder.encodeInto(arg, view);

        offset += ret.written;
        ptr = realloc(ptr, len, offset, 1) >>> 0;
    }

    WASM_VECTOR_LEN = offset;
    return ptr;
}

function takeFromExternrefTable0(idx) {
    const value = wasm.__wbindgen_externrefs.get(idx);
    wasm.__externref_table_dealloc(idx);
    return value;
}

let cachedTextDecoder = new TextDecoder('utf-8', { ignoreBOM: true, fatal: true });
cachedTextDecoder.decode();
const MAX_SAFARI_DECODE_BYTES = 2146435072;
let numBytesDecoded = 0;
function decodeText(ptr, len) {
    numBytesDecoded += len;
    if (numBytesDecoded >= MAX_SAFARI_DECODE_BYTES) {
        cachedTextDecoder = new TextDecoder('utf-8', { ignoreBOM: true, fatal: true });
        cachedTextDecoder.decode();
        numBytesDecoded = len;
    }
    return cachedTextDecoder.decode(getUint8ArrayMemory0().subarray(ptr, ptr + len));
}

const cachedTextEncoder = new TextEncoder();

if (!('encodeInto' in cachedTextEncoder)) {
    cachedTextEncoder.encodeInto = function (arg, view) {
        const buf = cachedTextEncoder.encode(arg);
        view.set(buf);
        return {
            read: arg.length,
            written: buf.length
        };
    };
}

let WASM_VECTOR_LEN = 0;

let wasmModule, wasmInstance, wasm;
function __wbg_finalize_init(instance, module) {
    wasmInstance = instance;
    wasm = instance.exports;
    wasmModule = module;
    cachedDataViewMemory0 = null;
    cachedFloat64ArrayMemory0 = null;
    cachedInt32ArrayMemory0 = null;
    cachedUint32ArrayMemory0 = null;
    cachedUint8ArrayMemory0 = null;
    wasm.__wbindgen_start();
    return wasm;
}

async function __wbg_load(module, imports) {
    if (typeof Response === 'function' && module instanceof Response) {
        if (!module.ok) {
            throw new Error(`failed to fetch Wasm: ${module.status} ${module.statusText} fetching '${module.url}'`);
        }

        if (typeof WebAssembly.instantiateStreaming === 'function') {
            try {
                return await WebAssembly.instantiateStreaming(module, imports);
            } catch (e) {
                const validResponse = expectedResponseType(module.type);

                if (validResponse && module.headers.get('Content-Type') !== 'application/wasm') {
                    console.warn("`WebAssembly.instantiateStreaming` failed because your server does not serve Wasm with `application/wasm` MIME type. Falling back to `WebAssembly.instantiate` which is slower. Original error:\n", e);

                } else { throw e; }
            }
        }

        const bytes = await module.arrayBuffer();
        return await WebAssembly.instantiate(bytes, imports);
    } else {
        const instance = await WebAssembly.instantiate(module, imports);

        if (instance instanceof WebAssembly.Instance) {
            return { instance, module };
        } else {
            return instance;
        }
    }

    function expectedResponseType(type) {
        switch (type) {
            case 'basic': case 'cors': case 'default': return true;
        }
        return false;
    }
}

function initSync(module) {
    if (wasm !== undefined) return wasm;


    if (module !== undefined) {
        if (Object.getPrototypeOf(module) === Object.prototype) {
            ({module} = module)
        } else {
            console.warn('using deprecated parameters for `initSync()`; pass a single object instead')
        }
    }

    const imports = __wbg_get_imports();
    if (!(module instanceof WebAssembly.Module)) {
        module = new WebAssembly.Module(module);
    }
    const instance = new WebAssembly.Instance(module, imports);
    return __wbg_finalize_init(instance, module);
}

async function __wbg_init(module_or_path) {
    if (wasm !== undefined) return wasm;


    if (module_or_path !== undefined) {
        if (Object.getPrototypeOf(module_or_path) === Object.prototype) {
            ({module_or_path} = module_or_path)
        } else {
            console.warn('using deprecated parameters for the initialization function; pass a single object instead')
        }
    }

    if (module_or_path === undefined) {
        module_or_path = new URL('praatfan_gpl_bg.wasm', import.meta.url);
    }
    const imports = __wbg_get_imports();

    if (typeof module_or_path === 'string' || (typeof Request === 'function' && module_or_path instanceof Request) || (typeof URL === 'function' && module_or_path instanceof URL)) {
        module_or_path = fetch(module_or_path);
    }

    const { instance, module } = await __wbg_load(await module_or_path, imports);

    return __wbg_finalize_init(instance, module);
}

export { initSync, __wbg_init as default };
