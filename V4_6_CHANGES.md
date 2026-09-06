# v4.6 changes — occurrence-aware privacy decisions

## Why this release changes the architecture

v4.5 validated identifier shapes well, but Benchmark 2 exposed a recurring precision problem: a valid-looking email/PAN/IFSC/UPI/card value could still be hidden when it appeared in documentation, examples, product IDs, or other non-private reference roles. It also exposed several recall gaps for implicit phone ownership, delivery addresses, self-identification names, and credentials.

v4.6 changes the privacy path to:

```text
ASR text
  -> normalization
  -> candidate generation
  -> type-specific validator
  -> occurrence-specific grammatical context
  -> deterministic example/reference veto
  -> optional NER support
  -> optional batched semantic judge
  -> conflict resolver
  -> mask + token/audio alignment
```

The important rule is that **shape does not imply ownership**. The same literal value is classified independently at every occurrence.

## False-positive reductions

- Documentation/example/test/tutorial/manual/source-code clauses can veto otherwise valid EMAIL, PAN, IFSC, UPI, CARD, Aadhaar, phone, and PIN-like candidates.
- The candidate text is removed before running the example/reference detector, so values such as `customer.one@example.com` cannot trigger their own `example` veto.
- PAN-shaped values no longer mask solely because they match the PAN regex; explicit PAN ownership or a successful ambiguous-context decision is required.
- Valid Luhn card values without ownership context are treated as ambiguous instead of automatically authoritative.
- Product serial, asset, batch, shipment, invoice, order, employee, tracking, and reference roles suppress conflicting PII interpretations.
- DOB ownership is occurrence-specific, so `My DOB is X, while the deadline was also X` masks only the first `X`.
- Operational/customer identifiers (transaction, complaint/ticket, customer/application IDs) remain available as a separate layer but are **off by default**.

## Recall improvements

- Phone ownership recognizes `his number`, `her number`, `Reach Rahul on ...`, and numbers saved `for future contact`.
- Delivery semantics recognize structured destinations after `send/deliver/ship/courier ... to`, even when the word `address` is absent.
- ADDRESS spans stop before contrastive clauses such as `, but ...` and continue to stop before subsequent PII fields.
- `postal code for my address is 110016` is classified as PINCODE rather than turning the six digits into an ADDRESS span.
- Self-identification such as `I am Neha Kapoor` is supported with conservative multi-token/title-case gating.
- Added context-gated PASSWORD, USERNAME, API_KEY, and AUTH_TOKEN masking.
- API/token placeholders such as `YOUR_API_KEY_HERE`, environment-variable names, and prefix-only documentation are left visible.
- Hard-coded assignments such as `OPENAI_API_KEY=...` and `GITHUB_TOKEN=...` are detected when the assigned value looks real; placeholder assignments remain visible.
- Spoken `+91` phone numbers are normalized correctly, including a fix where the single-letter ASR variant `o` could previously consume the beginning of the word `one`.
- Spoken PAN/IFSC values can use NATO/phonetic alphabet words plus spoken digits, e.g. `Alpha Bravo ... one two...` and `Hotel Delta Foxtrot Charlie zero...`.
- Phone formats such as `+91 (98765) 43210` are supported.

## Semantic AI changes

- Added positive/negative prototype banks for EMAIL, PAN, IFSC, UPI, CARD, and DOB ambiguity.
- Semantic context is clipped around contrastive grammatical boundaries so identical values in different roles are judged independently.
- Field transitions such as `and the transaction ID...` reset ownership, preventing an earlier account/phone label from leaking onto a later identical value.
- Hard deterministic example/reference vetoes run before neural review, reducing unnecessary embedding work.

## New regression coverage

- `tests/test_v46_context_precision.py`
- `benchmarks/context_precision_v46.jsonl`
- `reports/context_precision_v46.json`
- `reports/context_precision_v46.csv`

The new synthetic context benchmark is intentionally built around the v4.5 failures. On the bundled run it reports 16 TP, 0 FP, 0 FN at the balanced threshold. This is a regression result on synthetic cases, **not a production accuracy claim**.

The older bundled PII regression suite also remains at 61 TP, 0 FP, 0 FN after the changes.
