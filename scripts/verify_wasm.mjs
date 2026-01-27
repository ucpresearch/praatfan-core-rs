#!/usr/bin/env node
/**
 * Verify WASM outputs match native Rust outputs.
 *
 * This script loads an audio file, runs analysis through both the WASM module
 * and the native Rust JSON examples, and compares the results.
 *
 * Usage:
 *   node scripts/verify_wasm.mjs [audio_file]
 *
 * Requirements:
 *   - WASM package built: cd wasm && wasm-pack build --target nodejs
 *   - Rust examples built: cargo build --release --examples
 */

import { readFileSync, existsSync } from 'fs';
import { execSync } from 'child_process';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const projectRoot = join(__dirname, '..');

// Import WASM module
const wasmPath = join(projectRoot, 'wasm/pkg/praatfan_gpl.js');
if (!existsSync(wasmPath)) {
    console.error('WASM package not found. Build it first:');
    console.error('  cd wasm && ~/.cargo/bin/wasm-pack build --target nodejs');
    process.exit(1);
}

const wasm = await import(wasmPath);

// Default test file
const audioFile = process.argv[2] || join(projectRoot, 'tests/fixtures/one_two_three_four_five.wav');

if (!existsSync(audioFile)) {
    console.error(`Audio file not found: ${audioFile}`);
    process.exit(1);
}

console.log(`\n=== WASM vs Rust Verification ===`);
console.log(`Audio file: ${audioFile}\n`);

// Load audio file as raw samples (WAV only for simplicity)
function loadWavFile(path) {
    const buffer = readFileSync(path);
    const view = new DataView(buffer.buffer, buffer.byteOffset, buffer.byteLength);

    // Parse WAV header
    const riff = String.fromCharCode(view.getUint8(0), view.getUint8(1), view.getUint8(2), view.getUint8(3));
    if (riff !== 'RIFF') throw new Error('Not a WAV file');

    const numChannels = view.getUint16(22, true);
    const sampleRate = view.getUint32(24, true);
    const bitsPerSample = view.getUint16(34, true);

    // Find data chunk
    let dataOffset = 12;
    while (dataOffset < buffer.length - 8) {
        const chunkId = String.fromCharCode(
            view.getUint8(dataOffset), view.getUint8(dataOffset + 1),
            view.getUint8(dataOffset + 2), view.getUint8(dataOffset + 3)
        );
        const chunkSize = view.getUint32(dataOffset + 4, true);
        if (chunkId === 'data') {
            dataOffset += 8;
            break;
        }
        dataOffset += 8 + chunkSize;
    }

    // Read samples
    const bytesPerSample = bitsPerSample / 8;
    const numSamples = Math.floor((buffer.length - dataOffset) / bytesPerSample / numChannels);
    const samples = new Float64Array(numSamples);

    for (let i = 0; i < numSamples; i++) {
        let sum = 0;
        for (let ch = 0; ch < numChannels; ch++) {
            const offset = dataOffset + (i * numChannels + ch) * bytesPerSample;
            let sample;
            if (bitsPerSample === 16) {
                sample = view.getInt16(offset, true) / 32768;
            } else if (bitsPerSample === 24) {
                const b0 = view.getUint8(offset);
                const b1 = view.getUint8(offset + 1);
                const b2 = view.getInt8(offset + 2);
                sample = ((b2 << 16) | (b1 << 8) | b0) / 8388608;
            } else if (bitsPerSample === 32) {
                sample = view.getFloat32(offset, true);
            } else {
                throw new Error(`Unsupported bit depth: ${bitsPerSample}`);
            }
            sum += sample;
        }
        samples[i] = sum / numChannels;
    }

    return { samples, sampleRate, numChannels, bitsPerSample };
}

// Run Rust example and parse JSON output
function runRustExample(example, args) {
    const exePath = join(projectRoot, 'target/release/examples', example);
    if (!existsSync(exePath)) {
        console.error(`Rust example not found: ${exePath}`);
        console.error('Build with: cargo build --release --examples');
        process.exit(1);
    }
    const cmd = `"${exePath}" ${args.map(a => `"${a}"`).join(' ')}`;
    const output = execSync(cmd, { encoding: 'utf-8', cwd: projectRoot });
    return JSON.parse(output);
}

// Compare values with tolerance
function compareValues(wasmVal, rustVal, tolerance, label) {
    if (wasmVal === null && rustVal === null) return { match: true };
    if (wasmVal === null || rustVal === null) return { match: false, diff: 'null mismatch' };
    if (isNaN(wasmVal) && isNaN(rustVal)) return { match: true };
    if (isNaN(wasmVal) || isNaN(rustVal)) return { match: false, diff: 'NaN mismatch' };

    const diff = Math.abs(wasmVal - rustVal);
    const match = diff <= tolerance;
    return { match, diff, wasmVal, rustVal };
}

// Load audio
const { samples, sampleRate } = loadWavFile(audioFile);
console.log(`Loaded: ${samples.length} samples at ${sampleRate} Hz\n`);

// Create WASM Sound
const sound = new wasm.Sound(new Float64Array(samples), sampleRate);

// Test parameters
const timeStep = 0.01;
const pitchFloor = 75.0;
const pitchCeiling = 600.0;
const maxFormants = 5;
const maxFormantHz = 5500.0;
const windowLength = 0.025;
const preEmphasis = 50.0;
const minPitch = 100.0;
const silenceThreshold = 0.1;
const periodsPerWindow = 1.0;

let allPassed = true;

// ============ PITCH ============
console.log('--- Pitch Analysis ---');
try {
    const wasmPitch = sound.to_pitch(timeStep, pitchFloor, pitchCeiling);
    const rustPitch = runRustExample('pitch_json', [audioFile, timeStep, pitchFloor, pitchCeiling]);

    const wasmValues = wasmPitch.values();
    const wasmTimes = wasmPitch.times();

    // Helper to check if a value represents "unvoiced" (NaN, 0, or very small)
    const isUnvoiced = (v) => isNaN(v) || v === null || v === undefined || v === 0 || v < 1;

    let matches = 0, total = 0, maxDiff = 0;
    for (let i = 0; i < Math.min(wasmValues.length, rustPitch.frames.length); i++) {
        const wasmF0 = wasmValues[i];
        const rustF0 = rustPitch.frames[i].frequency;
        const rustVoiced = rustPitch.frames[i].voiced;

        // Both unvoiced = match
        if (isUnvoiced(wasmF0) && !rustVoiced) {
            matches++;
            total++;
            continue;
        }

        const result = compareValues(wasmF0, rustF0, 0.01, 'pitch');
        if (result.match) matches++;
        if (result.diff && !isNaN(result.diff)) maxDiff = Math.max(maxDiff, result.diff);
        total++;
    }

    const pct = (matches / total * 100).toFixed(1);
    console.log(`  Frames: WASM=${wasmValues.length}, Rust=${rustPitch.frames.length}`);
    console.log(`  Match rate: ${matches}/${total} (${pct}%) within 0.01 Hz`);
    console.log(`  Max difference: ${maxDiff.toFixed(4)} Hz`);

    if (pct < 99) allPassed = false;
    wasmPitch.free();
} catch (e) {
    console.log(`  ERROR: ${e.message}`);
    allPassed = false;
}

// ============ FORMANT ============
console.log('\n--- Formant Analysis ---');
try {
    const wasmFormant = sound.to_formant_burg(timeStep, maxFormants, maxFormantHz, windowLength, preEmphasis);
    const rustFormant = runRustExample('formant_json', [audioFile, timeStep, maxFormants, maxFormantHz, windowLength, preEmphasis]);

    const wasmTimes = wasmFormant.times();

    for (let fn = 1; fn <= 3; fn++) {
        const wasmFn = wasmFormant.formant_values(fn);
        const rustFn = rustFormant.formant[`f${fn}`];

        let matches = 0, total = 0, maxDiff = 0;
        for (let i = 0; i < Math.min(wasmFn.length, rustFn.length); i++) {
            const result = compareValues(wasmFn[i], rustFn[i], 1.0, `F${fn}`);
            if (result.match) matches++;
            if (result.diff && !isNaN(result.diff)) maxDiff = Math.max(maxDiff, result.diff);
            total++;
        }

        const pct = (matches / total * 100).toFixed(1);
        console.log(`  F${fn}: ${matches}/${total} (${pct}%) within 1 Hz, max diff: ${maxDiff.toFixed(2)} Hz`);
        if (pct < 99) allPassed = false;
    }

    wasmFormant.free();
} catch (e) {
    console.log(`  ERROR: ${e.message}`);
    allPassed = false;
}

// ============ INTENSITY ============
console.log('\n--- Intensity Analysis ---');
try {
    const wasmIntensity = sound.to_intensity(minPitch, timeStep);
    const rustIntensity = runRustExample('intensity_json', [audioFile, minPitch, timeStep]);

    const wasmValues = wasmIntensity.values();
    const rustValues = rustIntensity.intensity.values;

    let matches = 0, total = 0, maxDiff = 0;
    for (let i = 0; i < Math.min(wasmValues.length, rustValues.length); i++) {
        const result = compareValues(wasmValues[i], rustValues[i], 0.001, 'intensity');
        if (result.match) matches++;
        if (result.diff && !isNaN(result.diff)) maxDiff = Math.max(maxDiff, result.diff);
        total++;
    }

    const pct = (matches / total * 100).toFixed(1);
    console.log(`  Frames: WASM=${wasmValues.length}, Rust=${rustValues.length}`);
    console.log(`  Match rate: ${matches}/${total} (${pct}%) within 0.001 dB`);
    console.log(`  Max difference: ${maxDiff.toFixed(6)} dB`);

    if (pct < 99) allPassed = false;
    wasmIntensity.free();
} catch (e) {
    console.log(`  ERROR: ${e.message}`);
    allPassed = false;
}

// ============ SPECTRUM ============
console.log('\n--- Spectrum Analysis ---');
try {
    const wasmSpectrum = sound.to_spectrum(true);
    const rustSpectrum = runRustExample('spectrum_json', [audioFile]);

    const wasmCoG = wasmSpectrum.get_center_of_gravity(2.0);
    const wasmStd = wasmSpectrum.get_standard_deviation(2.0);
    const wasmSkew = wasmSpectrum.get_skewness(2.0);
    const wasmKurt = wasmSpectrum.get_kurtosis(2.0);

    const cogDiff = Math.abs(wasmCoG - rustSpectrum.spectrum.center_of_gravity);
    const stdDiff = Math.abs(wasmStd - rustSpectrum.spectrum.standard_deviation);
    const skewDiff = Math.abs(wasmSkew - rustSpectrum.spectrum.skewness);
    const kurtDiff = Math.abs(wasmKurt - rustSpectrum.spectrum.kurtosis);

    console.log(`  Center of Gravity: WASM=${wasmCoG.toFixed(2)}, Rust=${rustSpectrum.spectrum.center_of_gravity.toFixed(2)}, diff=${cogDiff.toFixed(4)}`);
    console.log(`  Std Deviation: WASM=${wasmStd.toFixed(2)}, Rust=${rustSpectrum.spectrum.standard_deviation.toFixed(2)}, diff=${stdDiff.toFixed(4)}`);
    console.log(`  Skewness: WASM=${wasmSkew.toFixed(4)}, Rust=${rustSpectrum.spectrum.skewness.toFixed(4)}, diff=${skewDiff.toFixed(6)}`);
    console.log(`  Kurtosis: WASM=${wasmKurt.toFixed(4)}, Rust=${rustSpectrum.spectrum.kurtosis.toFixed(4)}, diff=${kurtDiff.toFixed(6)}`);

    if (cogDiff > 0.01 || stdDiff > 0.01) allPassed = false;
    wasmSpectrum.free();
} catch (e) {
    console.log(`  ERROR: ${e.message}`);
    allPassed = false;
}

// ============ HARMONICITY ============
console.log('\n--- Harmonicity Analysis (AC) ---');
try {
    const wasmHnr = sound.to_harmonicity_ac(timeStep, pitchFloor, silenceThreshold, periodsPerWindow);
    const rustHnr = runRustExample('harmonicity_json', [audioFile, timeStep, pitchFloor, silenceThreshold, periodsPerWindow, 'ac']);

    const wasmValues = wasmHnr.values();
    const rustValues = rustHnr.values;

    let matches = 0, total = 0, maxDiff = 0;
    for (let i = 0; i < Math.min(wasmValues.length, rustValues.length); i++) {
        const rustHnrVal = rustValues[i].hnr;
        const result = compareValues(wasmValues[i], rustHnrVal, 0.01, 'hnr');
        if (result.match) matches++;
        if (result.diff && !isNaN(result.diff)) maxDiff = Math.max(maxDiff, result.diff);
        total++;
    }

    const pct = (matches / total * 100).toFixed(1);
    console.log(`  Frames: WASM=${wasmValues.length}, Rust=${rustValues.length}`);
    console.log(`  Match rate: ${matches}/${total} (${pct}%) within 0.01 dB`);
    console.log(`  Max difference: ${maxDiff.toFixed(4)} dB`);

    if (pct < 99) allPassed = false;
    wasmHnr.free();
} catch (e) {
    console.log(`  ERROR: ${e.message}`);
    allPassed = false;
}

// Clean up
sound.free();

// Summary
console.log('\n=== Summary ===');
if (allPassed) {
    console.log('✓ All tests PASSED - WASM output matches native Rust');
    process.exit(0);
} else {
    console.log('✗ Some tests FAILED - check differences above');
    process.exit(1);
}
