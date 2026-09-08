# Universal Financial Audio Intelligence Engine — v5.1 Privacy Hardening

v5.1 keeps the v5.0 correction-aware architecture and hardens the cases where uncertainty could accidentally expose an earlier value or leave accepted PII without audio coverage. It adds alignment-confidence and cross-speaker correction guards, reused-value protection, a last-resort accepted-PII audio guard, broader explicitly labeled secret coverage, and TTL cleanup for opt-in raw debug artifacts. No masking threshold is lowered.

## v5.1 hardening improvements

- **Low-confidence correction fail-safe:** weak Whisper alignment leaves both values masked instead of exposing the earlier value.
- **Cross-speaker correction guard:** when diarization labels exist, different speakers cannot be collapsed into one self-repair.
- **Reused-value safeguard:** a superseded value independently owned elsewhere remains masked.
- **Accepted-PII audio coverage invariant:** accepted PII cannot silently end with zero protected audio if normal alignment fails.
- **Expanded explicitly labeled secrets:** client/secret/signing/private keys and refresh/session/bearer tokens use the existing secret validation and example vetoes.
- **Raw-debug retention:** opt-in unredacted diagnostic bundles are best-effort cleaned after 24 hours by default.
- **Regression lock:** 106/106 unit tests pass; all legacy text suites remain 0 FP / 0 FN; v5.1 hardening suite is 5 TP / 0 FP / 0 FN.

See `V5_1_CHANGES.md` and `LIMITATIONS_V5_1.md`.


v5.0 builds on the full v4.12 optimized STT-Guard pipeline and adds conservative correction-aware masking for natural speech. A speaker can restart a phone/account/OTP/name/etc. value without saying a correction keyword; when timing and same-field ownership strongly establish a self-repair, the later committed value is masked and the superseded mistaken value is left unchanged. Ambiguous cases remain privacy-conservative and keep both values masked.

## v5.0 correction-aware improvements

- **Keyword-free self-repair detection:** uses existing PII decisions plus Whisper pause/timestamp evidence.
- **Correct-value authority:** a later committed same-type value can supersede the earlier mistaken value.
- **No broad threshold reduction:** the resolver can only operate inside a strongly owned field episode.
- **List/alternate-field protection:** `and/or`, primary/alternate/secondary/backup roles are not treated as corrections.
- **ASR punctuation recovery:** a corrected value after an ASR-inserted period/ellipsis can be revalidated with the normal type validator, then promoted only if timing confirms a restart.
- **Name-component coverage:** first/middle/last/given/family/maiden name and surname are explicitly owned NAME fields.
- **Safe correction audit:** the UI and result JSON report spans/type/pause/method without exposing raw values.
- **Regression lock:** 98/98 unit tests pass; all previous v4.4-v4.12 synthetic privacy suites retain 0 FP / 0 FN; the new v5.0 text precision suite is 9 TP / 0 FP / 0 FN.

See `V5_0_CHANGES.md` and `LIMITATIONS_V5_0.md`.

## v4.12 baseline retained

v4.12 retains the v4.11 reliability architecture, adds explicit `[TYPE AUDIO PROTECTED]` markers for conservatively guarded speech, makes documentation/example transitions hard raw-span boundaries, adds per-entity recovery telemetry, and removes major pre-ASR runtime overhead with a fast SoundFile/SOXR loader plus vectorized NumPy acoustic analysis. Privacy thresholds, ASR profiles, mask padding, ownership policy, NER/Semantic AI and recovery behavior are not reduced for speed.

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
  -> ownership-weighted conflict resolver + full owned-span authority
  -> token-owned PII/profanity entities
  -> unresolved expected-field check
       - targeted second ASR decode on a small audio window
       - alternate-hypothesis confirmation
       - bounded conservative audio guard if still unresolved
  -> protected transcript + token-time/audio-recovery redaction
  -> financial entities / intent / obligations / regulatory screening
  -> Streamlit review UI
```


## v4.12 optimized STT-guard improvements

- **Safe audio-guard transcript markers:** if a targeted privacy re-decode still cannot reconstruct a sensitive value but the bounded audio is beeped/muted, the safe transcript now shows a marker such as `Mine is [CVV AUDIO PROTECTED].` rather than leaving corrupted wording like `Mine ID`. The marker never contains guessed or alternate-ASR PII.
- **Hard documentation/example boundaries:** ownership-gated raw spans and recovery windows stop before manual/documentation/training/test/example clauses, even when ASR loses punctuation.
- **Per-entity recovery telemetry:** planned/attempted/recovered/guarded/unresolved counts and targeted-decode milliseconds are summarized by PII type and shown in the frontend.
- **Fast lossless-equivalent WAV/FLAC-style loading path:** SoundFile + SOXR HQ replaces high-overhead `librosa.load` when the container is directly supported. Unsupported containers retain the Librosa fallback.
- **Vectorized NumPy acoustic analysis:** removes the expensive Librosa spectral/onset hot path and its possible first-run JIT overhead. PII masking does not depend on the auxiliary stress marker.
- **More detailed performance metrics:** `acoustic_analysis` and `tamper_analysis` are reported separately in addition to total `signal_analysis`.
- **No privacy-speed trade:** v4.12 does not lower detector thresholds, shrink audio padding, disable AI checks, use a smaller ASR model, or skip STT recovery.

Local packaging regressions: **90/90 unit tests**, with all v4.4-v4.12 synthetic privacy suites retaining 0 FP / 0 FN. See `V4_12_CHANGES.md` and `LIMITATIONS_V4_12.md`.

To profile only the optimized audio-load/signal stage on your Windows recording without paying for ASR, run `profile_pre_asr.bat "C:\path\to\audio.wav"`.

## v4.11 reliability-guard improvements

- **One-shot typed expected-field state:** a field expectation is consumed once and is never propagated when the introducing sentence already supplied that field. This prevents `CVV 391` from contaminating later values such as `482 transactions`.
- **Quantity-role veto for CVV:** 3–4 digit values followed by roles such as transactions, records, items, units, rupees, pages, or samples are not CVV candidates merely because CVV appeared earlier in the clause.
- **Hard raw-fallback span authority:** ownership-gated fallbacks are clamped to the current field clause, next known role, and type-specific token/character budgets. Confidence cannot override these boundaries.
- **Two-lane recovery controller:** recovery now handles both previous-field follow-up answers and malformed same-sentence owned fields such as `my registered name is 2210`.
- **Broader recovery coverage:** NAME, PAN, IFSC, UPI, EMAIL, PHONE, Aadhaar, card, account, DOB, passport, driving licence, address and other supported core types can schedule targeted recovery when explicit ownership exists but primary ASR produced no accepted value.
- **Cross-punctuation IFSC protection:** a single ASR sentence break inside phonetic IFSC dictation can be treated as one bounded owned span without merging unrelated following fields.
- **Recovery telemetry:** reports planned/attempted/recovered/guarded/unresolved windows plus targeted-decode inference time; alternate transcript text is never included in normal telemetry.
- **Opt-in raw/final debug comparison:** `FINAI_PRIVACY_DEBUG_ARTIFACTS=1` writes raw ASR, safe transcript, redaction spans, recovery telemetry and hashes to a diagnostic folder. It is disabled by default because the raw file contains PII.
- **NER installer path fix retained:** the optional NER installer adds the project root to `PYTHONPATH` and the helper script is independently import-safe.

See `V4_11_CHANGES.md` and `LIMITATIONS_V4_11.md` for the previous reliability layer.

## v4.10 STT-guard improvements

- **Full owned-span masking:** ownership-gated raw fallbacks now control the complete sensitive value span, while stopping before new roles such as product batches, reference IDs and ordinary trailing prose.
- **Strict normalized support is preserved:** when a wider raw span overlaps a strict normalized candidate, v4.10 keeps the normalized canonical interpretation while using the safer wider mask boundary.
- **Stronger teaching/example vetoes:** phrases such as `can be spoken as`, `may be written as`, `formatted as`, `represented as`, `syntax is`, and training-document examples are treated as explanations unless personal ownership is explicit.
- **Expectation-aware privacy re-decode:** if a sentence establishes a sensitive field and the immediately following owned response contains no accepted PII, only that small audio window is decoded again with a field-specific privacy prompt and stronger beam search.
- **No invented PII:** alternate ASR text is internal. If recovery confirms the expected type, only the recovered word timings are used for audio redaction.
- **Conservative audio guard:** if the second decode also fails, a short type-bounded response interval can still be muted/beeped. This protects audio when the primary ASR deletes the value entirely.
- The extra ASR pass is conditional; ordinary calls with no unresolved expected field do not pay the second-decode cost.

See `V4_10_CHANGES.md` and `LIMITATIONS_V4_10.md`.

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

The bundled synthetic regression suites are for development regression only. They are not production-accuracy claims. v4.10 retains all prior suites, keeps `benchmarks\asr_robust_v49.jsonl`, and adds `benchmarks\stt_guard_v410.jsonl` for spoken-as explanation vetoes and full raw-fallback mask boundaries. Audio recovery behavior is covered by `tests\test_v410_stt_guard.py` because it depends on ASR timing/state rather than text-only cases.

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
