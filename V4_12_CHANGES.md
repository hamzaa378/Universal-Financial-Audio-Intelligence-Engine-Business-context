# v4.12 Optimized STT-Guard Privacy

v4.12 is a reliability + runtime optimization release built on v4.11. It deliberately does **not** lower privacy thresholds, change the Faster-Whisper model/profile, reduce audio-mask padding, disable Semantic AI/NER, or skip any supported PII validator. The optimization work is concentrated in auxiliary audio loading/signal analysis and in safer rendering/telemetry around conservative STT recovery.

## 1. Conservative audio guards are visible in the safe transcript

When the primary ASR loses a sensitive reply and targeted re-decode still cannot safely reconstruct it, the audio is already beeped/muted. v4.12 now reflects that protection in the safe transcript without inventing the missing value.

Example:

```text
The CVV is located on the back of the card. Mine ID, address is ...
```

with a guarded CVV recovery window becomes:

```text
The CVV is located on the back of the card. Mine is [CVV AUDIO PROTECTED]. address is ...
```

The alternate ASR text is never exposed. The marker means only that the bounded audio interval was protected.

## 2. Documentation/example clauses are hard mask-span boundaries

v4.11 already had semantic example/documentation vetoes. v4.12 also makes those transitions geometric boundaries for raw ownership fallbacks and recovery windows. This prevents a damaged owned field from swallowing a following manual/example clause when ASR omits punctuation.

Example:

```text
My bank IFSC is Hotel Delta Foxtrott ID Charlie 00001234,
and the software manual uses HDFC 0001234 as an example IFSC.
```

becomes:

```text
My bank IFSC is [IFSC REDACTED],
and the software manual uses HDFC 0001234 as an example IFSC.
```

The reset is clause-aware, so values such as `example.com` do not trigger a boundary by themselves.

## 3. Per-entity recovery telemetry

Recovery telemetry now includes a `by_type` summary. Example:

```json
{
  "CVV": {
    "planned": 1,
    "attempted": 1,
    "recovered": 0,
    "guarded": 1,
    "unresolved": 0,
    "decode_inference_ms": 597.3
  }
}
```

The Streamlit recovery audit displays this table. No alternate transcript or raw PII is stored in normal telemetry.

## 4. Fast audio decode with identical high-quality resampling contract

For containers supported by libsndfile, v4.12 uses:

```text
SoundFile -> mono float32 -> SOXR HQ -> 16 kHz
```

instead of the higher-overhead `librosa.load` wrapper. Unsupported containers continue to use the previous Librosa fallback.

On the local 48 kHz WAV comparison used while building v4.12, the old and new 16 kHz waveforms had a maximum absolute sample difference of `0.0`.

## 5. Vectorized NumPy signal analysis

The previous shared acoustic pass depended on Librosa spectral-centroid and onset helpers. On some machines the first run can incur large initialization/JIT overhead. v4.12 replaces that hot path with vectorized/chunked NumPy calculations:

- centered 1024-sample frames / 512-sample hop;
- RMS;
- zero-crossing rate;
- clipping and silence ratios;
- chunked FFT spectral centroid;
- positive spectral-flux activity proxy.

The quality score still uses the same SNR/silence/clipping/volume formula. Stress markers are auxiliary and are not used to decide whether PII is masked.

## 6. Tamper-analysis cleanup

The tamper detector now computes median/MAD once instead of recalculating them inside the threshold expression. Its scoring logic is unchanged.

## 7. More detailed performance telemetry

`performance` now includes:

```text
acoustic_analysis
tamper_analysis
signal_analysis
```

in addition to audio load, ASR, privacy/NLP, privacy re-decode, audio redaction, total, duration and RTF.

## Local regression results

All synthetic privacy suites remained clean after the optimization:

| Suite | TP | FP | FN |
|---|---:|---:|---:|
| Original PII | 61 | 0 | 0 |
| v4.6 context precision | 16 | 0 | 0 |
| v4.7 context recall | 18 | 0 | 0 |
| v4.8 precision guard | 8 | 0 | 0 |
| v4.9 ASR robustness | 14 | 0 | 0 |
| v4.10 STT guard | 4 | 0 | 0 |
| v4.11 reliability guard | 6 | 0 | 0 |
| v4.12 optimized guard | 3 | 0 | 0 |

Unit tests: **90/90 passed** during packaging.

These are synthetic regression results, not production-world accuracy claims.

## Local pre-ASR speed comparison

On a synthetic 60-second, 48 kHz WAV in the build environment:

| Stage | v4.11 | v4.12 | Observed speedup |
|---|---:|---:|---:|
| Audio load/resample | 2455 ms | 38.6 ms | ~63.7x |
| Signal analysis | 371.7 ms | 62.0 ms | ~6.0x |
| Combined pre-ASR | 2827 ms | 100.6 ms | ~28.1x |

The quality score was identical (`0.462`) on that test. The stress score changed slightly (`0.081` -> `0.067`) but remained in the same `low` band; stress is auxiliary and never changes PII masking decisions. Windows results will depend on file format, storage, CPU, and library builds.
