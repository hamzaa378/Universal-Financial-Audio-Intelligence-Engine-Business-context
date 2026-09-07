# v4.10 — STT-Guard Privacy

v4.10 focuses on the last major failures seen in real spoken tests rather than adding a new broad detector architecture.

## 1. Full owned-span masking for ASR raw fallbacks

v4.9 could correctly identify an ASR-damaged field but choose a suboptimal overlap span. v4.10 gives strong-ownership raw fallbacks explicit mask-span authority. When a strict normalized candidate overlaps the raw span, its canonical value and confidence support are retained while the safer owned span controls redaction.

Value trimming was also tightened. New-field/reference boundaries such as `product batch`, `document template`, `support ticket`, invoice/shipment references and ordinary trailing prose stop the fallback span. This avoids both partial masking and accidental masking of a later public/reference value.

## 2. Stronger spoken-as / teaching / format veto

The explanation layer now understands language such as:

- `can be spoken as`
- `may be written as`
- `formatted as`
- `represented as`
- `syntax is`
- `format looks like`
- training document/material/guide examples

These cues suppress normalized PII-looking examples when personal ownership is absent. A real owned sentence such as `My email is sara dot khan at example dot com` still masks.

## 3. Expectation-aware targeted ASR recovery

When protected audio is requested, v4.10 inspects the primary transcript for a narrow pattern:

1. one sentence clearly establishes a sensitive field;
2. the immediately following sentence begins like an owned response (`Mine...`, `I received...`, etc.); and
3. the primary PII layer accepted no value of the expected type in that response.

Only then does v4.10 schedule a second Faster-Whisper decode for that small response window. The recovery prompt is field-specific and asks Whisper to preserve digits, letters, phonetic alphabet words, `at`, `dot`, and separators exactly.

The alternate hypothesis is never published as raw PII. It is used only to confirm the expected type and recover audio timings.

If alternate decoding still fails, the optional conservative guard protects a short type-specific response interval. This is specifically intended for primary-ASR deletion cases such as a spoken CVV disappearing from the text.

## Runtime controls

- `FINAI_PRIVACY_REDECODE=1`
- `FINAI_PRIVACY_REDECODE_MAX_WINDOWS=4`
- `FINAI_PRIVACY_CONSERVATIVE_AUDIO_GUARD=1`

Recovery is conditional. Normal calls with no unresolved expected reply do not invoke the second ASR pass.

## Regression coverage

- Existing unit tests: retained.
- v4.10 tests cover spoken-as example vetoes, complete raw fallback spans, trailing-role boundaries, recovery planning, alternate spoken-CVV confirmation, conservative guard behavior, and audio-only recovery intervals.
- Synthetic text suites retained: original PII, v4.6 precision, v4.7 recall, v4.8 precision guard and v4.9 ASR robustness.

Synthetic regression success is not a production-world accuracy claim. Real accented/noisy audio remains necessary for external validation.
