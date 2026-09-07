# v5.0 Correction-Aware Privacy — limitations

## Ambiguous corrections

Natural language can be inherently ambiguous. Two values spoken one after another may be a correction or may be two legitimate values. v5.0 therefore resolves a correction only when ownership, field continuity and timing all agree. If uncertain, both values remain masked rather than exposing one.

## The mistaken value is intentionally left unmasked after a high-confidence repair

This follows the requested behavior: only the final committed value is treated as the correct personal value. There is an unavoidable trade-off: a speaker's "wrong" value could itself happen to be real sensitive information (for example an old phone number). If that risk matters more than presentation accuracy, a future policy option should support `mask_superseded_values=true`.

## Partial corrections are still difficult

Example:

```text
My phone number is 9876543210 ... last digit one.
```

The speaker did not repeat the full intended number. v5.0 does not reconstruct or guess the final identifier from a partial repair. A future typed edit-state machine could support suffix/prefix replacement, but it must be heavily validated to avoid false positives.

## NAME repairs require stronger timing evidence

A comma in a name can introduce a location/title rather than a correction, so text punctuation alone never unmasks an earlier NAME. Audio timing is required for NAME self-repair resolution.

## Address corrections remain conservative

Addresses contain many internal numbers, pauses and commas. Automatically treating a second location fragment as a replacement would create unacceptable false positives. Repeated full address fields or a dedicated address edit-state model are safer future directions.

## ASR deletion still cannot be reconstructed truthfully

If both primary and targeted recovery ASR completely delete the corrected value, the existing STT guard can beep/mute the bounded response but cannot recover the actual text.

## Timing quality matters

Keyword-free correction resolution depends on Whisper word timestamps. Very poor timing/alignment may cause the resolver to keep both values masked. This is safer than selecting the wrong one.

## Speaker overlap

Without reliable diarization, two nearby values from different speakers can resemble a self-repair. The same-field and list/role guards reduce this risk, but overlapping speech remains difficult.

## Real-world validation remains required

The bundled benchmark is synthetic. Correction-aware behavior should be evaluated on consented recordings containing natural hesitations, restarts, accents, noise and multiple speakers before production accuracy claims are made.
