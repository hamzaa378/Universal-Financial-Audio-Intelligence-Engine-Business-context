# Universal Financial Audio Intelligence Engine — v4.7 Context-Recall Privacy Hybrid

v4.7 keeps the v4.6 occurrence-aware precision layer and adds field-clause segmentation, one-sentence discourse ownership, ownership-weighted conflict resolution, and stronger spoken identifier normalization. The goal is to recover difficult true PII without giving up the false-positive reductions from v4.6.

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
       - spoken UPI/VPA handles
       - spoken DOB phrases
       - spoken passport / driving-licence forms
       - separator-normalized IFSC
       - natural-language dates
  -> field-aware clause segmentation
  -> deterministic privacy candidates / validators
  -> occurrence-specific ownership + role context
  -> one-sentence expected-field state for owned follow-up answers
  -> example/document/reference veto
  -> optional token NER support for ambiguous clauses
  -> optional semantic privacy judge using local grammatical role
  -> ownership-weighted conflict resolver
  -> token-owned PII/profanity entities
  -> protected transcript + token-time audio redaction
  -> financial entities / intent / obligations / regulatory screening
  -> Streamlit review UI
```


## v4.7 context-recall improvements

- Long multi-PII sentences are segmented at field transitions before semantic/context decisions, preventing one field from contaminating another.
- A one-sentence `expected_type` state can connect prompts such as `Please enter your Aadhaar number.` to owned replies such as `Mine is ...`; it expires after exactly one sentence and requires response ownership.
- Conflict resolution ranks explicit ownership before ambiguous format confidence, e.g. `My account is 4111...` resolves to ACCOUNT_NUMBER rather than CARD.
- Spoken DOB, UPI, passport and Indian driving-licence dictation are normalized under explicit type context.
- `my full name is ...` and `my registered name is ...` are supported with bounded name extraction.

## v4.6 context-precision improvements retained

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

See `V4_7_CHANGES.md` for the new recall architecture and `V4_6_CHANGES.md` for the precision-layer rationale.

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

The bundled synthetic regression suites are for development regression only. They are not production-accuracy claims. v4.7 retains `benchmarks\context_precision_v46.jsonl` and adds `benchmarks\context_recall_v47.jsonl` for field segmentation, one-sentence discourse ownership, ownership conflicts, spoken DOB/UPI/passport/driving-licence values, and full/registered names.

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
