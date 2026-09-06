# Universal Financial Audio Intelligence Engine — v4.6 Occurrence-Aware Privacy Hybrid

v4.6 keeps the v4.5 GPU/audio privacy path and adds an occurrence-aware decision layer aimed specifically at reducing contextual false positives while improving masking recall. A validator now proves identifier shape; a separate ownership/context stage decides whether that exact occurrence should be hidden.

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
  -> occurrence-specific ownership + role context
  -> example/document/reference veto
  -> optional token NER support for ambiguous clauses
  -> optional semantic privacy judge using local grammatical role
  -> conflict resolver
  -> token-owned PII/profanity entities
  -> protected transcript + token-time audio redaction
  -> financial entities / intent / obligations / regulatory screening
  -> Streamlit review UI
```


## v4.6 context-precision improvements

- Identical values are classified independently per occurrence; a DOB/phone role is never cached by literal value.
- Documentation, sample, tutorial, test-value, source-code and reference roles can veto valid-looking PII when ownership is absent.
- Example detection analyzes the surrounding clause with the candidate removed, avoiding self-vetoes such as `example.com`.
- Implicit phone ownership now covers `his/her number`, named contact phrases, and `for future contact`.
- Delivery destinations can become ADDRESS candidates without requiring the word `address`.
- ADDRESS capture stops before contrastive clauses such as `, but ...`.
- Added context-gated PASSWORD, USERNAME, API_KEY and AUTH_TOKEN masking with placeholder suppression.
- Hard-coded `*_API_KEY=...` / `*_TOKEN=...` assignments are protected when the value is real-looking, while placeholders remain visible.
- Spoken `+91` numbers and parenthesized phone formatting are supported; spoken digit normalization now distinguishes `one` from the ASR variant `o` correctly.
- NATO/phonetic-alphabet PAN and IFSC dictation is normalized under explicit PAN/IFSC context.
- Transaction/ticket/complaint/customer/application identifiers are still available, but their separate operational-ID layer is disabled by default to reduce false positives.

See `V4_6_CHANGES.md` for the detailed rationale and regression cases.

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

The bundled default ONNX model is supporting evidence only; deterministic validators remain authoritative for identifier shape/checks, while the occurrence-context layer decides whether that valid shape is actually private in the current clause. The default model is not treated as a Hindi/Hinglish production NER model.

## v4.5 privacy improvements retained

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
Operational/customer IDs OFF by default
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

The bundled synthetic regression suite is for development regression only. It is not a production-accuracy claim. v4.6 also runs `benchmarks\context_precision_v46.jsonl`, which specifically covers documentation/examples, same-value role changes, implicit phone ownership, delivery addresses, self-identification, and credentials.

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
