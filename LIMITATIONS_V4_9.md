# v4.9 major limitations and next remedies

v4.9 improves robustness to real ASR corruption, but several limitations remain.

| Limitation | Possible remedy | Difficulty | Trade-off / negative effect |
|---|---|---:|---|
| ASR can delete the private value entirely, e.g. `Mine ID` with no digits | N-best ASR hypotheses, privacy-biased re-decoding, or acoustic token recovery around strong field prompts | High | More GPU/CPU use and latency; alternate hypotheses can increase false positives |
| Raw-span fallback is intentionally privacy-biased under strong ownership | Add calibrated ownership/format fusion trained on labeled noisy call data | Medium–High | A learned calibrator needs representative data and can become domain-specific |
| Fuzzy driving-licence state normalization is India-specific | Locale/country plug-in validators and fuzzy dictionaries | High | More maintenance, configuration and country-specific false-positive risks |
| Mixed `at` email/UPI normalization still depends on a reasonably recognizable local/provider structure | Character/word-level noisy-channel normalizer or small sequence model | Medium–High | More model cost and potential hallucinated normalization if not constrained |
| Explanation veto is lexical and may miss unusual teaching/definition language | Lightweight clause classifier for VALUE vs EXPLANATION vs REFERENCE | Medium | Classifier errors can suppress real PII or mask harmless prose |
| Broader name ownership is still English/Hinglish oriented | Multilingual intent/ownership model plus locale-specific grammars | High | Larger model footprint and cross-language testing burden |
| Public vs personal phone/address roles are heuristic | Entity-role classifier with BUSINESS/PUBLIC/PERSONAL ownership classes | Medium–High | More semantic inference and possible public/private role mistakes |
| One-sentence expected-field state is intentionally short | Speaker-aware typed discourse state with TTL and turn ownership | Medium–High | Longer state can incorrectly attach unrelated later values |
| Severe punctuation loss can still merge clauses | Use ASR word timestamps, pauses and prosody to create field-aware boundaries | Medium | Extra preprocessing and edge cases around rapid speech |
| Audio redaction quality still depends on token/time alignment | Forced alignment or confidence-dependent temporal padding | High | More compute; padding can remove surrounding harmless audio |
| Overlapping speakers remain difficult | Diarization/channel separation before privacy alignment | High | Significant runtime and deployment complexity |
| Unknown PII categories remain outside the detector | Add policy-driven plug-in types only when validation/ownership rules are available | Medium | Broad generic detectors can sharply increase false positives |
| Synthetic tests remain much easier than real calls | Build a consented, labeled, noisy multilingual audio benchmark with accents, crosstalk and ASR errors | High–Very High | Data collection, labeling and privacy-governance cost |

## Recommended next priorities

1. **P0: real-audio benchmark and error corpus.** Capture ASR mistakes and evaluate end-to-end privacy recall, not only clean transcript accuracy.
2. **P0: N-best/privacy-aware re-decoding for strong ownership spans.** This addresses values that disappear or become too corrupted for the text layer.
3. **P1: timestamp/pause-aware clause segmentation.** Reduce dependence on ASR punctuation.
4. **P1: small VALUE vs EXPLANATION vs REFERENCE role classifier.** Replace an ever-growing list of lexical veto phrases.
5. **P2: locale plug-ins and multilingual ownership models.** Expand beyond India/English/Hinglish without making one universal detector overly aggressive.
