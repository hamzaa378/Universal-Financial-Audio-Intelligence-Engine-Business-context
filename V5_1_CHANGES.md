# v5.1 Privacy Hardening

v5.1 is a conservative hardening release on top of v5.0 Correction-Aware Privacy. It does not lower any PII threshold, remove any validator, shrink mask padding, disable Semantic AI/NER, or relax existing example/reference vetoes.

## Changes admitted under the regression lock

### 1. Correction decisions require reliable alignment

A correction is one of the few places where the engine may intentionally stop masking an earlier accepted value. v5.1 therefore requires both the earlier and later values to have usable Whisper token ownership and alignment confidence >= 0.68 before the earlier value may be exposed. If timing is weak, both values remain masked.

### 2. Cross-speaker correction guard

When optional diarization has assigned speakers, two same-type values from disjoint speakers are never collapsed into one speaker self-correcting. The system keeps both accepted PII values protected. No diarization inference is added by v5.1; the guard reuses speaker labels only when they already exist.

### 3. Reused superseded-value protection

A value that appears to be a local mistake may still be a real old/secondary identifier. If the same normalized value is independently present in another strongly-owned occurrence of the same PII type, v5.1 refuses to expose it during correction resolution.

### 4. Accepted-PII audio coverage invariant

An already-accepted PII entity is not allowed to end with zero audio protection merely because character offsets and ASR segment text diverge. Normal token timing and character-ratio fallback remain first choice. Only when both fail does v5.1 add a short bounded guard around the nearest ASR segment. This does not create new text PII decisions.

### 5. Broader explicitly-labeled secret coverage

The existing ownership-gated credential detector now also supports labels such as:

- client secret
- secret key
- signing key
- private key
- refresh token
- session token
- bearer token

Values must still pass the existing real-secret checks, and documentation/sample/placeholder contexts remain vetoed. No global entropy scanner was added.

### 6. Raw debug artifact TTL

Opt-in raw privacy diagnostics intentionally contain unredacted PII. v5.1 adds best-effort retention cleanup with a default 24-hour TTL before new debug bundles are written.

Environment variable:

```text
FINAI_PRIVACY_DEBUG_TTL_HOURS=24
```

Set debug mode only on synthetic/consented data.

## Regression results

- 106/106 unit tests pass.
- Legacy v4.4-v5.0 text regression suites remain at 0 FP / 0 FN.
- New v5.1 hardening text suite: 5 TP / 0 FP / 0 FN with 3 negative cases.
- Detector-only timing on the v5.0 correction corpus remained effectively unchanged in local tests (~0.27 ms/case in both v5.0 and v5.1).

These are synthetic regression results, not production accuracy claims.
