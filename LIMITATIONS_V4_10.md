# v4.10 limitations and next remedies

## 1. Targeted re-decode adds variable latency
**Why:** each unresolved expected response can trigger another Whisper inference window.  
**Remedy:** batch nearby recovery windows, use a smaller recovery model, or cap recovery to the highest-risk types.  
**Difficulty:** Medium–High.  
**Negative effect:** smaller/faster recovery models can reduce recognition accuracy; aggressive batching may enlarge audio windows.

## 2. Conservative audio guard can over-redact
**Why:** if both ASR passes fail, v4.10 protects a bounded response interval without knowing the exact sensitive token timing.  
**Remedy:** use acoustic digit/character confidence, N-best/lattice APIs if available, or forced alignment against alternate hypotheses.  
**Difficulty:** High.  
**Negative effect:** more sophisticated recovery costs compute and may still produce uncertain boundaries.

## 3. Recovery depends on an explicit expected-field discourse pattern
**Why:** the second decode is deliberately gated to avoid re-decoding arbitrary low-confidence speech.  
**Remedy:** add a lightweight discourse model that tracks requested/expected fields across speaker turns and punctuation loss.  
**Difficulty:** Medium–High.  
**Negative effect:** longer-lived state can connect unrelated later numbers and increase false positives.

## 4. Complete ASR deletion before the response cue can still be invisible
If the primary ASR removes both the value and enough of the surrounding ownership grammar, no targeted recovery window may be scheduled.  
**Remedy:** use acoustic anomaly/low-confidence regions plus dialogue-state expectations from the previous speaker turn.  
**Difficulty:** High.  
**Negative effect:** more windows will be re-decoded and some ordinary speech may be conservatively muted.

## 5. Faster-Whisper normal API is not a true N-best lattice interface
v4.10 therefore uses targeted second decoding rather than consuming a native N-best list.  
**Remedy:** integrate a recognizer/backend that exposes token lattices or multiple hypotheses, or maintain multiple decode settings for the same short window.  
**Difficulty:** High.  
**Negative effect:** model/backend complexity, GPU memory use and latency increase.

## 6. Raw-fallback span expansion is still grammar-bound
Unexpected punctuation or a new field label not in the boundary lexicon can cause slight over-mask or under-mask.  
**Remedy:** learn field boundaries from token timestamps and a small sequence tagger while retaining hard privacy limits.  
**Difficulty:** Medium.  
**Negative effect:** a learned boundary model may become less predictable and needs a labeled corpus.

## 7. Explanation suppression can create rare false negatives
A real user might say something like `My email can be spoken as...` and then dictate the real value.  
**Remedy:** distinguish generic placeholders from owned examples using speaker role, lexical specificity and follow-up validation.  
**Difficulty:** Medium.  
**Negative effect:** weakening the veto too much reintroduces documentation/example false positives.

## 8. Multilingual/code-switched ASR remains uneven
The context grammar is still English/Hinglish-heavy and Indian financial identifiers are best supported.  
**Remedy:** language-specific ownership policies plus multilingual normalization and NER.  
**Difficulty:** High.  
**Negative effect:** model size, maintenance cost and cross-language false positives increase.

## 9. Audio redaction quality still depends on Whisper word timing
Confirmed alternate decoding improves missing-value recovery, but inaccurate word timestamps can still shift beep/mute edges.  
**Remedy:** phoneme/forced alignment and confidence-dependent padding.  
**Difficulty:** High.  
**Negative effect:** extra latency and wider padding can remove nearby harmless speech.

## 10. Synthetic tests do not establish real-world accuracy
The regression suites are intentionally small and failure-focused.  
**Remedy:** build a consented real-audio benchmark covering accents, noise, code-switching, speaking rates, microphones and overlapping speakers.  
**Difficulty:** Very High.  
**Negative effect:** collection, labeling and privacy governance are expensive but necessary for credible production metrics.
