# v5.2 Partial-Correction & Recovery Hardening

v5.2 is an additive privacy-hardening release on top of v5.1. It does not lower any normal PII confidence threshold and does not add a generic short-number or entropy detector.

## 1. Partial self-correction

v5.2 supports deterministic suffix repairs for numeric PII types when an already-accepted, strongly-owned value exists. Supported types are PHONE, CARD, ACCOUNT_NUMBER, OTP, CVV, PINCODE and AADHAAR.

Examples:

- `My phone number is 9876543210. Last digit is 1.`
- `My account number is 1234567890. Change the last two digits to 42.`

Only the spoken replacement fragment is protected. The program does **not** reconstruct or store the corrected full identifier. The earlier value is exposed only when Whisper alignment is strong, same-speaker ownership is safe, and that earlier value is not independently owned elsewhere. If timing is weak, both the old value and the replacement fragment remain masked.

Arity is strict. `last two digits ... 7` is rejected because one replacement digit cannot satisfy a two-digit edit. Example/documentation/reference contexts are vetoed.

## 2. Correction after an audio-only guard

When the primary field is already protected as `[TYPE AUDIO PROTECTED]` because both the primary and targeted ASR paths could not safely recover it, a nearby explicit self-repair can carry that field type forward.

Example:

`My OTP is <unresolved>. I made a mistake. 124981.`

renders safely as:

`My OTP is [OTP AUDIO PROTECTED] I made a mistake. [OTP AUDIO PROTECTED].`

The second correction uses the normal OTP validator under a synthetic ownership prefix and the primary Whisper timestamps. No additional recovery-ASR decode is needed for that second value.

Partial edits after an audio guard are also protected without reconstructing the full identifier, e.g. `Last digit is 1.` protects only the spoken `1` span.

## 3. False-positive controls

The new paths require an existing owned/guarded sensitive-field episode. They reject:

- standalone `I made a mistake. 124981.`
- standalone `The last digit is 1.`
- documentation/example/sample/test contexts
- ticket/order/transaction/complaint/reference/experiment contexts
- replacement arity mismatches
- low-confidence timing when exposing an earlier superseded value

Ambiguous cases remain mask-both rather than guess-and-expose.

## 4. Safer debug diagnostics

Normal `FINAI_PRIVACY_DEBUG_ARTIFACTS=1` operation is now span/safe-text only. A raw unredacted ASR transcript is written only when the second explicit opt-in is set:

```bat
set FINAI_PRIVACY_DEBUG_ARTIFACTS=1
set FINAI_PRIVACY_DEBUG_RAW=1
```

The existing TTL cleanup remains active. On platforms where supported, the raw file is also written with owner-only permissions.

## 5. Runtime/model drift fingerprint

`verify_components.bat` now records `reports/runtime_fingerprint.json` after a successful component check. The fingerprint tracks privacy-sensitive runtimes such as Faster-Whisper, CTranslate2, ONNX Runtime, FastEmbed, Tokenizers and Transformers plus the semantic/NER model identifiers. Changes are surfaced as a warning so the privacy benchmark can be rerun before trusting an updated environment.

You can also run:

```bat
capture_runtime_fingerprint.bat
```

## 6. Validation

- 114 / 114 unit tests pass.
- Accumulated text privacy suites: 144 TP / 0 FP / 0 FN.
- v5.2 adds five negative correction-like text cases, all remaining visible.
- Base deterministic privacy benchmark in this environment: mean about 0.276 ms/case, P95 about 0.710 ms/case.
- The new partial-correction resolver itself measured about 0.025 ms mean on a small synthetic correction case in this environment.

These are development regression results, not production accuracy claims. Real noisy audio should still be evaluated on consented/synthetic labeled recordings.
