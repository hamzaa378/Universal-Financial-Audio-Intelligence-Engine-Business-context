# v4.11 Reliability-Guard Privacy

v4.11 is a reliability update for failures caused by real speech-to-text transitions. It does **not** lower global privacy thresholds or replace the deterministic/context architecture with a larger model. The architectural changes are limited to state, ownership boundaries, and recovery orchestration because those directly improve masking recall while reducing false positives.

## 1. One-shot typed expected-field state

The previous one-sentence expectation mechanism could still allow a recently mentioned type to influence a later number in a malformed/under-punctuated ASR sentence. v4.11 treats an expected field as a consumable state:

- if the introducing sentence already contains a credible value of that type, the state is immediately satisfied and is not propagated;
- otherwise it can be consumed by only the immediately following owned response;
- if that response does not consume it, the state expires.

A CVV-specific quantity-role guard also rejects values followed by roles such as `transactions`, `records`, `items`, `units`, `rupees`, `pages`, and `samples`.

Example:

```text
The CVV on my card is 391, there were 482 transactions yesterday.
-> The CVV on my card is [CVV REDACTED], there were 482 transactions yesterday.
```

## 2. Hard fallback-span boundaries

Raw ownership fallbacks are now physically bounded independently of confidence. A fallback stops at:

1. the current field clause;
2. a recognized next field/operational role;
3. a type-specific character budget; and
4. a type-specific token budget.

This prevents a malformed identifier from swallowing transaction IDs, product batches, addresses or unrelated trailing prose.

## 3. Two-lane privacy recovery controller

Recovery now has two lanes:

### Follow-up expectation

```text
The CVV is located on the back of the card.
Mine ID ...
```

The prior field establishes a one-shot expectation. If the primary transcript has no accepted CVV, the small response window is re-decoded.

### Same-sentence owned corruption

```text
My registered name is 2210.
My PAN is alpha bravo 12 foxtrot.
```

Explicit ownership plus no accepted entity now schedules targeted re-decoding even without a `Mine is` response.

The recovery planner supports the core privacy types already protected by the detector, including NAME, PHONE, EMAIL, PAN, AADHAAR, IFSC, UPI, CARD, CVV, OTP, ACCOUNT_NUMBER, PINCODE, DOB, PASSPORT, DRIVING_LICENSE, VOTER_ID and ADDRESS.

## 4. Cross-punctuation IFSC protection

ASR can insert a period inside phonetic dictation:

```text
My IFSC is Hotel Delta Foxtrot. Charlie 001234 and my address is ...
```

Under explicit personal IFSC ownership, v4.11 can protect the bounded two-fragment raw IFSC span while stopping before the next field. It does not invent the missing character or claim a canonical IFSC if strict reconstruction is impossible.

## 5. Recovery telemetry

`privacy.asr_privacy_recovery.telemetry` now exposes:

```text
windows_planned
windows_attempted
recovered
guarded
unresolved
decode_inference_ms
total_ms
```

Each audit entry includes type, reason, source lane, protected time window, decode time and whether alternate ASR returned any text. Alternate transcript text itself is intentionally omitted.

## 6. Raw/final transcript diagnostic bundle

Set:

```bat
set FINAI_PRIVACY_DEBUG_ARTIFACTS=1
set FINAI_PRIVACY_DEBUG_DIR=privacy_debug
```

The pipeline writes:

```text
raw_asr_transcript.txt
final_safe_transcript.txt
redaction_spans.json
recovery_telemetry.json
comparison.json
```

This answers a critical diagnostic question: did content disappear in primary ASR, or was it removed by the privacy/redaction layer?

**Warning:** `raw_asr_transcript.txt` contains unredacted PII. Debug artifacts are disabled by default and should be deleted after diagnosis.

## 7. Verification

Local synthetic regressions after the v4.11 changes:

| Suite | TP | FP | FN |
|---|---:|---:|---:|
| Original PII | 61 | 0 | 0 |
| v4.6 context precision | 16 | 0 | 0 |
| v4.7 context recall | 18 | 0 | 0 |
| v4.8 precision guard | 8 | 0 | 0 |
| v4.9 ASR robustness | 14 | 0 | 0 |
| v4.10 STT guard | 4 | 0 | 0 |
| v4.11 reliability guard | 6 | 0 | 0 |

`85/85` unit tests pass in the packaging environment. These are synthetic regression results, not production accuracy claims.
