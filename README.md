# Universal Financial Audio Intelligence Engine — v4.9 ASR-Robust Privacy Hybrid

v4.9 keeps the v4.7/v4.8 occurrence-aware architecture and adds an ASR-robust privacy layer: ownership-gated raw-span fallback for damaged values, mixed spoken/written email and UPI normalization, mixed NATO/digit alphanumeric decoding, conservative fuzzy driving-licence state handling, stronger explanation/public-role vetoes, and broader natural ownership phrases. Strict validation remains the default path; the fallback hides only strongly owned source spans and never invents missing private characters.

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
  -> ownership-gated raw-span fallback for ASR-damaged values
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


## v4.9 ASR-robust improvements

- Strong personal ownership can mask a bounded raw span even when ASR corruption breaks strict validation; the system never guesses the missing value.
- Mixed forms such as `arove.private at mail.co.in` and `rf.con at OKSBI` are normalized under email/UPI ownership.
- PAN/passport/IFSC-style dictation can mix NATO words with written numeric chunks.
- Indian driving-licence state names tolerate conservative ASR spelling errors only under DL context and structural validation.
- Explanatory language such as `may contain`, `usually contains`, `identifies a`, and `has a structured format` receives a stronger context veto.
- Public/customer-support helplines are suppressed as personal-phone candidates.
- Broader ownership phrases cover preferred/account names, best contact numbers, SMS/contact wording, email/payment instructions, mailing addresses and `you can find me at ...`.

See `V4_9_CHANGES.md` for implementation details and `LIMITATIONS_V4_9.md` for remaining risks and remedies.

## v4.8 precision-guard improvements

- **PUBLIC_GEO_CONTEXT veto for PIN codes:** public postal-region statements such as `The postal code 411001 covers part of Pune.` remain visible, while personal forms such as `My PIN code is 411045.` and `The postal code for my address is 110016.` still mask.
- **Stricter ADDRESS end boundaries:** ADDRESS capture now stops before new fields introduced with modifiers such as `alternate`, `secondary`, `backup`, `other`, `primary`, and `registered`.
- **Field-clause segmentation also understands those modifiers**, preventing a later email/phone field from contaminating the context decision for an earlier address.
- **Ownership-gated partial financial identifiers:** explicit phrases such as `My card ends with 1111` and `The last four digits of my account are 9012` now protect only the four digits. Unowned look-alikes such as payment amounts, reference numbers, or test-card statements remain visible.
- Partial-identifier candidates are fully hidden even when the global output mode is `partial`, because revealing their suffix would reveal the entire sensitive value.

See `V4_8_CHANGES.md` for implementation details and `LIMITATIONS_V4_8.md` for known limitations, possible remedies, implementation difficulty, and trade-offs.

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

See `V4_9_CHANGES.md` for the ASR-robust layer, `V4_8_CHANGES.md` for the focused precision changes, `V4_7_CHANGES.md` for the recall architecture, and `V4_6_CHANGES.md` for the occurrence-aware precision-layer rationale.

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

The bundled synthetic regression suites are for development regression only. They are not production-accuracy claims. v4.9 retains all prior suites and adds `benchmarks\asr_robust_v49.jsonl` for ASR-corrupted owned values, mixed spoken/written email/UPI, public-line suppression, explanation vetoes, and broader ownership language.

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
