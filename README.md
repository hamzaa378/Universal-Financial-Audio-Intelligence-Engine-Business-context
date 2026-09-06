# Universal Financial Audio Intelligence Engine — v4.5 Privacy-First GPU Hybrid

v4.5 keeps strict CUDA/FP16 Faster-Whisper from v4.4 and hardens the privacy path around the failure modes that matter most in real calls: ASR punctuation/format variation, spoken identifiers, name/address overcapture, and character-to-audio alignment.

## Processing architecture

```text
Audio / microphone
  -> one 16 kHz decode
  -> shared signal analysis
  -> Faster-Whisper CUDA FP16
  -> word IDs + timestamps + confidence
  -> ASR-aware normalization
       - spoken digits
       - multi-level spoken email
       - separator-normalized IFSC
       - natural-language dates
  -> deterministic privacy candidates / validators
  -> optional token NER support for ambiguous clauses
  -> optional semantic privacy judge
  -> conflict resolver
  -> token-owned PII/profanity entities
  -> protected transcript + token-time audio redaction
  -> financial entities / intent / obligations / regulatory screening
  -> Streamlit review UI
```

## IFSC robustness

All of these are recognized when appropriate:

```text
ABCD0123456
ABCD-0123456
ABCD 0123456
ABCD_0123456
My IFSC is ABCD zero one two three four five six
```

The separator/spoken forms are context-gated so a generic reference such as `Reference ABCD-0123456` is not automatically hidden in Balanced mode.

## Install and GPU verification

Run:

```bat
install_v4.bat
run_frontend.bat
```

The strict GPU launcher now performs a real tiny Faster-Whisper CUDA inference before Streamlit starts. It requires:

```text
cublas64_12.dll
cublasLt64_12.dll
cudnn64_9.dll
CTranslate2 CUDA device > 0
actual Whisper CUDA/FP16 smoke inference = PASS
```

There is no silent CPU fallback in `run_frontend.bat`. CPU troubleshooting remains available through `run_frontend_cpu.bat`.

For a full component report run:

```bat
verify_components.bat
```

The report separates:

```text
ASR GPU
Semantic AI
Privacy Judge
Token NER AI
Diarization
```

## Optional token NER AI

The core program does not require a large token-classification model. To add the optional ONNX second opinion:

```bat
install_ner_ai.bat
```

Then enable **Token NER second opinion** in the frontend and click **Warm up & verify AI**.

The bundled default ONNX model is supporting evidence only; deterministic validators remain authoritative for PAN, IFSC, Luhn-valid cards, Aadhaar checks/context, and other strong structures. The default model is not treated as a Hindi/Hinglish production NER model.

## v4.5 privacy improvements

- IFSC separator normalization with false-positive guardrails.
- Spoken IFSC recognition for digit-by-digit branch codes.
- Spoken-number normalization for phone/Aadhaar/card/account/OTP/CVV/PIN contexts.
- Multi-level spoken email reconstruction such as `name dot x at bank dot co dot in`.
- Natural-language DOB patterns such as `12 March 1998`, only under DOB/birth context.
- Token-bounded NAME extraction so `My name is John Doe and my account...` masks only `John Doe`.
- Clause-bounded ADDRESS extraction so the mask stops before the next phone/email/account field.
- Optional ONNX token-NER support only for ambiguous contextual candidates.
- Batched semantic privacy decisions retained from v4.4.
- PII/profanity/financial-ID entities are attached to Whisper token IDs/timestamps before audio masking.
- Audio redaction prefers direct token timing; character-ratio timing is now a padded fallback only.

## Recommended frontend settings

For RTX 4060 demo work:

```text
Detection profile       Balanced hybrid
Semantic AI             ON
Token NER               OFF initially, then ON after install_ner_ai.bat
Full PII redaction      ON
Sensitive financial IDs ON
Profanity reduction     ON
Whisper                 small + Fast demo while debugging
                         medium + Balanced for stronger accuracy
Protected audio         Bleep tone
Diarization             OFF unless needed
```

## Regression benchmark

```bat
run_privacy_benchmark.bat
```

The bundled synthetic regression suite is for development regression only. It is not a production-accuracy claim.

## Real audio manifest benchmark

Use `benchmarks\audio_manifest_example.jsonl` as the schema, then run:

```bat
run_audio_manifest_benchmark.bat "C:\path\to\manifest.jsonl" small
```

It reports per sample:

- WER,
- PII precision/recall/F1,
- profanity precision/recall/F1,
- token-alignment coverage,
- optional gold audio-redaction coverage,
- ASR time,
- text-AI time,
- total RTF,
- ASR CUDA/compute type,
- semantic provider,
- NER provider.

Use consented or synthetic labeled audio for real evaluation claims.
