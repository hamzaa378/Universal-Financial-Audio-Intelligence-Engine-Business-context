# v5.2 remaining limitations, difficulty and trade-offs

Language coverage and country-specific financial-format coverage are intentionally excluded from this table.

| Remaining limitation | Severity | Difficulty | Possible remedy | Negative aspects / trade-offs |
|---|---|---:|---|---|
| Primary ASR and targeted recovery can both delete a sensitive value | High | High | N-best/lattice access or a specialized digit/alphanumeric recognizer for only high-risk fields | Extra GPU inference, latency and more competing hypotheses. Conservative audio guarding remains safer than guessing. |
| ASR can delete the ownership phrase itself | High | High | Bounded turn-level dialogue expectation using neighboring prompts and speaker state | Longer context memory can wrongly connect unrelated later speech and over-mask. |
| Diarization remains optional and disabled by default | High | High | Enable/calibrate a diarization backend | Significant model/runtime cost, RAM/VRAM use and speaker-label errors. |
| Overlapping speakers remain difficult | High | Very High | Source separation plus speaker-attributed ASR | Large compute cost; separation artifacts can hurt ASR and masking alignment. |
| Audio timing can still be inaccurate on severely corrupted ASR | High | High | Forced alignment with confidence-adaptive padding | Extra processing and potentially wider beeps over harmless words. |
| Address detection and address correction are still heuristic/conservative | High | High | Dedicated address parser/NER plus ownership-aware component state | More inference; risk of masking public locations or swallowing adjacent fields. |
| Visual PII is outside this audio/transcript pipeline | High | Very High | Separate OCR/UIA/layout privacy engine synchronized with audio | Major scope, compute and another false-positive surface. |
| Synthetic regressions do not establish real production accuracy | High | Very High | Consented labeled corpus with noise, accents, repairs, overlap and exact mask timing labels | Data collection, annotation and privacy-governance cost. |
| Partial self-correction currently supports deterministic suffix edits only (last/final 1-4 digits) | Medium-High | Medium | Add prefix and explicit digit-position edit state | More edit grammar increases ambiguity and can expose/mask the wrong digit if interpretation is wrong. |
| Partial corrections for names, email, UPI, IFSC and addresses are not reconstructed | Medium-High | High | Type-specific structured edit models | These values have nontrivial token structure; a mistaken edit can synthesize incorrect sensitive data. |
| Recovery windows are still capped | Medium-High | Medium | Risk-aware adaptive scheduling or reuse of encoder features | Raising the cap increases GPU latency; complex scheduling can delay safe output. |
| Unexpected ASR punctuation can still split less-common identifier forms | Medium-High | Medium | More type-specific timestamp continuation rules | Wider continuation windows can over-mask neighboring prose. |
| Unknown unlabeled secrets can still be missed | Medium-High | Medium | Organization-specific schemas or tightly gated entropy classifier | Generic entropy detection often masks hashes, UUIDs and benign random IDs. |
| A superseded mistaken value may itself be sensitive even if not reused elsewhere | High | Low-Medium | Optional policy to always mask superseded values | Safer privacy, but conflicts with the desired “mask the final corrected value, not the mistaken value” presentation. |
| Raw debug PII can still be written when `FINAI_PRIVACY_DEBUG_RAW=1` is explicitly enabled | Medium | Low-Medium | OS-protected/encrypted diagnostic storage | Key management and less convenient debugging. v5.2 no longer writes raw text under ordinary debug mode. |
| Semantic/NER behavior can still change after upgrades | Medium | Low-Medium | Pin exact package/model revisions and require regression lock before deployment | More release maintenance and slower upgrades. v5.2 now detects and reports runtime drift but does not prevent upgrades. |
| Operational IDs remain policy-dependent | Medium | Low-Medium | Organization-specific profiles for transaction/ticket/order/employee IDs | Enabling broad operational-ID masking can create significant over-redaction. |
| Recovery may create occasional latency spikes | Medium | Medium-High | Batch nearby recovery windows / encoder feature reuse where supported | More scheduling complexity. v5.2 avoids an extra decode for validated post-guard corrections. |

## Intentionally not added

v5.2 still does not use global short-number masking, global entropy scanning, unbounded correction memory, automatic speaker diarization, or speculative reconstruction of partially corrected identifiers. Each of those could raise recall at the cost of false positives, latency, or incorrect masking.
