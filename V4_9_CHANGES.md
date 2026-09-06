# v4.9 changes — ASR-robust privacy and broader ownership language

v4.9 keeps the v4.7/v4.8 occurrence-aware architecture and adds a bounded robustness layer for real ASR mistakes. The core principle is that the detector should not have to reconstruct a corrupted private value perfectly before it can protect the raw transcript span.

## 1. Ownership-gated raw masking fallback

Strict validators remain the primary path. When ASR damages a value so that strict validation fails, v4.9 can still mask the original span if the local grammar establishes very strong personal ownership.

Examples now protected:

```text
You can also contact me at plus 911-23456789.
My credit card number is 4111-1111-111.
My PIN code is 4111045.
My bank IFSC is Hotel Delta Foxtrott ID, Charlie 00001234.
```

The fallback never guesses missing digits or letters. It only hides the source span already present in the transcript.

Fallbacks are deliberately limited to explicit ownership/assignment forms and type-specific length/shape bounds. Public helplines, operational IDs, examples and documentation still pass through the context veto layer.

## 2. Mixed spoken/written email and UPI normalization

ASR often emits only the `at` sign as a word while retaining literal dots:

```text
arov.com at example.com
arove.private at mail.co.in
rf.con at OKSBI
```

v4.9 normalizes these occurrence spans to canonical email/UPI candidates without rewriting the transcript. It also handles command grammar where another `at` occurs earlier:

```text
Mail me at sara.private at mail.co.in.
Pay me at sara dot khan at oksbi.
```

The normalizers use overlapping candidate searches so the command preposition does not steal the actual address/handle span.

## 3. Mixed NATO + written-digit alphanumeric normalization

Spoken alphanumeric runs may now combine phonetic words and already-decoded numeric chunks:

```text
Alpha Bravo Charlie Delta Echo 1234 Foxtrot
```

normalizes to:

```text
ABCDE1234F
```

This improves PAN/passport/IFSC-style dictation without loosening their final type validators.

## 4. Fuzzy Indian driving-licence state names

Under explicit driving-licence context, state names can tolerate conservative ASR spelling errors:

```text
Carnitaka0120210012345
Karnatika 01 2021 0012345
```

can normalize to `KA0120210012345` when the resulting DL structure validates. Fuzzy matching is not run globally.

## 5. Stronger explanation/public-context suppression

v4.9 adds definition-language vetoes such as:

```text
may contain
can contain
usually contains
typically contains
consists of
identifies a
has a structured format
is printed on
```

This prevents sentences like these from becoming PII values when ASR punctuation is poor:

```text
A residential address may contain a house number, road, city and PIN code.
A UPI ID usually contains an at sign.
A mobile number may contain ten digits.
```

ADDRESS candidate creation also has an early guard for explanatory phrases, preventing the address span itself from swallowing the negative evidence.

## 6. Public/business phone suppression

Phone-like values under public-line roles are no longer treated as personal phone numbers in Balanced mode:

```text
The customer support helpline is 911-2456789.
The support hotline is 98765-43210.
```

Explicit personal contact grammar still wins.

## 7. Broader personal ownership language

Additional natural forms include:

```text
I prefer to be called Aisha Noor.
Please address me as Sara Khan.
The name on my account is Sara Khan.
The best number to reach me is 9988776655.
SMS me on 9876543210.
Mail me at sara.private at mail.co.in.
Send the confirmation to sara.private at mail.co.in.
Pay me at sara dot khan at oksbi.
My mailing address is 10 Green Road, Pune 411001.
You can find me at Flat 2, Lake Road, Pune 411001.
Mail the correspondence to Flat 3, Lake Road, Pune 411001.
```

## 8. Regression and performance

Synthetic development regression at the Balanced threshold:

| Suite | TP | FP | FN |
|---|---:|---:|---:|
| Original PII suite | 61 | 0 | 0 |
| v4.6 context precision | 16 | 0 | 0 |
| v4.7 context recall | 18 | 0 | 0 |
| v4.8 precision guard | 8 | 0 | 0 |
| v4.9 ASR robustness | 14 | 0 | 0 |

Unit tests: **65/65 passed** in the development environment.

On the same bundled text workload, detector/text-privacy mean time increased from roughly **0.14 ms/case in v4.8 to 0.26 ms/case in v4.9** in the development container. This is a large percentage increase in detector-only work but still sub-millisecond and normally negligible beside ASR inference.

These figures are synthetic regression measurements, not production accuracy claims.
