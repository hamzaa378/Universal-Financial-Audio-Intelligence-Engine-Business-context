# v5.0 Correction-Aware Privacy

v5.0 keeps the v4.12 detector, ownership, STT recovery, masking geometry and performance optimizations intact, and adds a conservative self-correction resolver for natural speech.

## Goal

Handle cases where a speaker gives the wrong private value and immediately says the intended value without an explicit correction keyword.

Examples:

```text
My phone number is 9876543210 ... 9123456789.
```

With a real acoustic restart/pause, the protected result is:

```text
My phone number is 9876543210 ... [PHONE REDACTED].
```

Only the later committed value is masked. The superseded mistaken value is not treated as the final personal value.

## Safety architecture

The correction layer is downstream of normal detection. It cannot create a correction simply because a later number looks plausible.

A correction is accepted only when:

1. both values are the same PII type;
2. the earlier value has strong explicit ownership;
3. both occur in the same local field episode;
4. no alternate/secondary/primary/other field begins between them;
5. no `and` / `or` list relationship is present;
6. the values are different; and
7. Whisper timing shows a meaningful restart/pause, or text contains an unmistakable repair boundary.

If correction evidence is ambiguous, v5.0 keeps both accepted PII values masked. This is intentionally privacy-conservative.

## ASR punctuation reset recovery

A later corrected value can be missed by ordinary ownership because Whisper inserts punctuation:

```text
My OTP is 638291... 638292.
```

v5.0 may rediscover `638292` using the same normal OTP validator under synthetic local ownership, but it is promoted only after timing proves a restart. This does not lower global thresholds.

## NAME corrections

Explicit name fields now also include:

- first name
- middle name
- last name
- given name
- family name
- maiden name
- surname

For a spoken restart such as:

```text
My name is Sara Khan, [pause] Aisha Noor.
```

text-only analysis conservatively masks both names. Audio-mode timing can resolve the restart and protect only `Aisha Noor`.

## False-positive guards

The following are not treated as corrections:

```text
My phone number is 9876543210 and 9123456789.
My primary phone number is 9876543210 and my alternate phone number is 9123456789.
```

A short hesitation without enough pause evidence also leaves both values masked.

Public/example roles remain governed by all existing v4.12 context vetoes. `synthetic identifier(s)` was added as an explicit example/reference cue after the new correction benchmark exposed that phrase as an ambiguous phone-shape false positive.

## Auditability

Audio results now contain:

```text
privacy.correction_resolution.count
privacy.correction_resolution.events
```

Audit rows expose only type, character spans, method, pause duration and whether the mistaken/final values were masked. Raw values are not included.

The Streamlit UI shows a `Correction resolution audit` expander when a repair is resolved.

## Regression results

- **98/98 unit tests pass**.
- All legacy v4.4-v4.12 synthetic privacy suites remain at **0 FP / 0 FN**.
- New v5.0 text precision suite: **9 TP / 0 FP / 0 FN**.
- Timing-specific no-keyword correction behavior is covered by unit tests with synthetic Whisper word timestamps.

These are regression results, not production accuracy claims.

## Runtime

No neural model or extra ASR pass is added for ordinary corrections. The correction resolver reuses already-computed Whisper word timings. On the local 171-case detector corpus used during packaging, v5.0 detector latency remained effectively unchanged relative to v4.12 (roughly 0.3 ms/case in this environment).
