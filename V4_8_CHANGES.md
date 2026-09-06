# v4.8 changes — focused precision guardrails

v4.8 intentionally keeps the v4.7 architecture. It makes three narrow changes chosen from Benchmark 3 because they improve precision or protection without making the whole detector more aggressive.

## 1. PUBLIC_GEO_CONTEXT veto for PIN codes

A six-digit postal code is not automatically personal information. v4.7 would correctly mask:

```text
My PIN code is 411045.
```

but could also mask public geographic statements such as:

```text
The postal code 411001 covers part of Pune.
```

v4.8 introduces a local public-geography veto. The decision is occurrence-specific and only applies when explicit personal/address ownership is absent.

Expected behavior:

```text
The postal code 411001 covers part of Pune.
-> unchanged

My PIN code is 411045.
-> My PIN code is [PINCODE REDACTED].

The postal code for my address is 110016.
-> The postal code for my address is [PINCODE REDACTED].
```

This remains a conservative lexical policy rather than a general geographic knowledge system.

## 2. Stricter ADDRESS end boundaries

v4.7 could over-capture the beginning of a following field when that field contained a modifier such as `alternate` or `secondary`.

Before:

```text
My address is 15 Rose Garden Road, Chennai 600028, and my alternate phone number is 9988776655.
```

could produce an ADDRESS span extending into the following field label.

v4.8 expands field-boundary grammar to recognize modifiers including:

```text
alternate
secondary
backup
other
primary
registered
```

The expected protected transcript is now:

```text
My address is [ADDRESS REDACTED], and my alternate phone number is [PHONE REDACTED].
```

The same modifier grammar is used by field-clause segmentation, so a later email such as `sara@example.com` cannot contaminate the context decision for the preceding ADDRESS field.

## 3. Ownership-gated partial card/account identifiers

Four digits by themselves are too ambiguous to mask safely. v4.8 therefore does not add a generic four-digit detector.

It instead recognizes explicit ownership + partial-identifier grammar, for example:

```text
My card ends with 1111.
-> My card ends with [CARD REDACTED].

The last four digits of my account are 9012.
-> The last four digits of my account are [ACCOUNT_NUMBER REDACTED].
```

Negative controls remain visible:

```text
I paid 1111 rupees yesterday.
Reference number 9012 appears on page two.
The test card ends with 1111.
```

Partial identifiers carry `partial_identifier=True` in internal metadata. Even if the global masking mode is `partial`, these values are fully hidden because revealing a suffix would reveal the entire detected value.

## Architecture retained from v4.7

```text
ASR transcript
  -> field-aware clause segmentation
  -> candidate generation / spoken normalization
  -> deterministic type validators
  -> occurrence-specific ownership
  -> one-sentence expected-field state
  -> example/reference/public-context vetoes
  -> optional NER / semantic second opinion
  -> ownership-weighted conflict resolver
  -> transcript masking
  -> token/time alignment for protected audio
```

No new large model is required by the v4.8 changes.

## Regression results

Synthetic development regression after the v4.8 changes:

| Suite | TP | FP | FN |
|---|---:|---:|---:|
| Original PII suite | 61 | 0 | 0 |
| v4.6 context-precision suite | 16 | 0 | 0 |
| v4.7 context-recall suite | 18 | 0 | 0 |
| v4.8 precision-guard suite | 8 | 0 | 0 |
| Unit tests | 53 / 53 passed |  |  |

The deterministic text detector averaged well below 1 ms per short synthetic transcript in the local regression environment. This does **not** represent end-to-end audio latency; ASR, optional AI inference, audio I/O, and GPU setup dominate real execution time.

These are synthetic regression results only. They are not production-world accuracy claims.
