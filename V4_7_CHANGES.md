# v4.7 changes — context recall without precision regression

## Why this release changes the architecture

Benchmark 3 showed that v4.6 successfully reduced false positives but became too conservative in three situations: long sentences containing many different PII fields, short follow-up answers whose field was established in the previous sentence, and spoken forms that needed type-specific normalization.

v4.7 keeps the v4.6 example/reference veto and occurrence-specific semantics, then adds:

```text
ASR transcript
  -> sentence boundaries
  -> field-aware clause segmentation
  -> candidate generation + type validators
  -> current-clause ownership
  -> one-sentence expected-field state
  -> example/reference/public-role veto
  -> optional NER / semantic judge
  -> ownership-weighted conflict resolver
  -> mask + token/audio alignment
```

## P0 changes

### 1. Field-clause segmentation

Context is now clipped at field transitions such as:

```text
my name is X, DOB Y, phone Z, email A, account number B, IFSC C, and address D
```

Each field is evaluated in its own local clause. This prevents a word in a later field, such as `example` inside a spoken email domain, from vetoing an earlier NAME, DOB, PHONE, ACCOUNT_NUMBER or IFSC candidate.

### 2. One-sentence expected-type state

A previous sentence may establish one expected PII type. The immediately following sentence may consume that state only when it begins with an ownership response such as `Mine is`, `It is`, or `I received`.

Examples:

```text
Please enter your Aadhaar number. Mine is 4567 8901 2345.  -> AADHAAR
An OTP will be sent shortly. I received 843291.            -> OTP
```

Safety constraints:

- TTL is exactly one sentence.
- If the next sentence does not consume the state, it expires.
- A matching-looking identifier in a non-response sentence is not rescued.
- The state carries a type expectation, not a remembered literal value.

### 3. Ownership-weighted conflict resolution

Overlapping candidates are now ranked by explicit ownership before raw format confidence. For example:

```text
My account is 4111 1111 1111 1111.
```

The digits are Luhn-valid, but the explicit ACCOUNT ownership wins, so the final type is ACCOUNT_NUMBER rather than CARD.

## P1 changes

- Spoken DOB: `twenty-first January two thousand two` -> canonical `2002-01-21` under DOB context.
- Spoken UPI: `sara dot khan at oksbi` -> `sara.khan@oksbi` under UPI/VPA context.
- Spoken passport: NATO/phonetic letter + spoken digits, e.g. `Papa one two ...` -> `P1234567`.
- Spoken driving licence: Indian state-name + spoken digits, e.g. `Karnataka zero one ...` -> `KA0120210012345`; phonetic two-letter state codes are also accepted when the final canonical structure is valid.
- Name ownership: `my full name is ...` and `my registered name is ...` are now supported.

## Regression coverage

New files:

- `tests/test_v47_context_recall.py`
- `benchmarks/context_recall_v47.jsonl`
- `reports/context_recall_v47.json`
- `reports/context_recall_v47.csv`

Bundled local regression results at the balanced threshold:

- Unit tests: 45 / 45 passed.
- Original PII regression: 61 TP, 0 FP, 0 FN.
- v4.6 context-precision regression: 16 TP, 0 FP, 0 FN.
- New v4.7 context-recall regression: 18 TP, 0 FP, 0 FN.

These are synthetic regression results, not a production-world accuracy claim. Real call audio, ASR errors, accents, mixed languages and unseen layouts still require separate evaluation.
