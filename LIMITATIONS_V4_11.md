# v4.11 major limitations and possible remedies

| Limitation | Possible remedy | Difficulty | Negative effects / trade-offs |
|---|---|---:|---|
| Primary ASR can delete the sensitive value completely | Targeted second decode (implemented); future N-best/lattice access or a dedicated digit/alphanumeric acoustic recognizer | High | Extra inference on difficult windows; conservative fallback can mute harmless speech |
| Recovery is capped to a small number of windows | Dynamic risk-based budget based on call length and unresolved high-risk fields | Medium | More GPU work and latency on noisy calls |
| Same-sentence recovery grammar is still language/phrase dependent | Multilingual ownership parser or compact sequence classifier trained on field-ownership utterances | High | Model maintenance; wrong ownership predictions can increase false positives |
| Cross-punctuation IFSC logic permits only one short continuation | General typed continuation state using word timestamps and pause duration | Medium | Longer continuation windows can swallow neighboring speech if boundaries are poor |
| Raw fallback hard limits can truncate unusually long valid dictation | Type-specific learned/empirical span budgets plus acoustic pause boundaries | Medium | Larger budgets raise over-masking risk |
| Address extraction is still heuristic | Dedicated multilingual address NER/parser plus address ownership model | High | More latency and potential masking of public places |
| Public/operational-role vocabulary is finite | Organization-configurable role taxonomy and policy profiles | Medium | Configuration burden; an over-broad taxonomy may suppress real PII |
| Optional Semantic AI pooling behavior can change with FastEmbed versions | Pin a tested model/runtime combination or register the pooling configuration explicitly, then recalibrate thresholds | Medium | Dependency pinning and maintenance overhead |
| Debug artifacts intentionally contain raw PII | Keep disabled by default; encrypt/TTL-delete diagnostic bundles; generate span-only diagnostics where possible | Medium | Encryption/key management complexity; span-only logs are less useful for ASR diagnosis |
| NER/Semantic AI run on CPU by default | Profile first; move only a proven bottleneck to CUDA | Low–Medium | GPU contention with Faster-Whisper and increased VRAM pressure |
| Diarization is absent unless separately installed | Optional speaker-attributed ASR/pyannote integration | High | Significant compute, model downloads, speaker-label errors |
| Synthetic tests cannot establish production accuracy | Build a consented, labeled real-audio corpus across accents, noise, code-switching and ASR corruption | Very High | Data collection/labeling cost and privacy governance |

## Recommended next work

The detector itself is no longer the dominant known risk. The next evidence-driven priorities are:

1. test v4.11 with real noisy recordings and inspect the new recovery telemetry;
2. use the opt-in raw/safe debug bundle only on synthetic/consented recordings to classify failures as **ASR deletion**, **candidate failure**, **span failure**, or **policy veto**;
3. optimize signal analysis separately if end-to-end performance remains dominated by acoustic preprocessing.
