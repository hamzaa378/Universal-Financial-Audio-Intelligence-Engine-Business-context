# v4.12 limitations and next remedies

v4.12 improves real-STT protection and runtime without weakening the privacy policy, but several production limitations remain.

| Limitation | Possible remedy | Difficulty | Trade-off / negative effect |
|---|---|---:|---|
| Primary and targeted ASR can both delete a value | Keep conservative audio guard; optionally add a third ultra-short decode/profile only for very high-risk fields | High | More latency and GPU work; extra hypotheses can create false-positive recovery candidates |
| Recovery marker is based on bounded ownership, not reconstructed truth | Keep `[TYPE AUDIO PROTECTED]` as an audit marker only | Low | Transcript is less verbatim, but safer and more honest than guessing |
| Audio-guard marker may normalize malformed wording (`Mine ID` -> `Mine is ...`) | Preserve a separate audit table with original timing/status | Low | Small presentation change; raw wording remains available only in opt-in debug artifacts |
| ASR punctuation can still split uncommon identifiers in unexpected places | Extend type-specific one-boundary continuation grammars and use word timing gaps | Medium | Overly broad continuation can swallow neighboring prose if not carefully bounded |
| Hard documentation boundaries rely partly on English/Hinglish lexical cues | Add multilingual clause-role classifier / language-specific reset dictionaries | High | Larger maintenance surface and possible false vetoes across languages |
| Optional Semantic AI/NER can shift with model/runtime updates | Pin model revisions and run threshold-calibration CI after dependency upgrades | Medium | More release-management work and larger cached artifacts |
| Signal-analysis stress score is an auxiliary heuristic | If exact historical stress-score continuity matters, keep a legacy-analysis compatibility switch | Low | Legacy mode is slower and can reintroduce first-run Librosa overhead |
| SoundFile does not decode every MP4/M4A/WebM build | Librosa/FFmpeg fallback remains enabled | Low | Those formats may not receive the full loader speedup |
| Recovery can add ~0.5-1+ s per failed sensitive window | Keep window cap and prioritize CVV/OTP/PAN/IFSC before lower-risk fields | Low | A very corrupted call with many missing fields can still be slower |
| Diarization remains optional/unavailable unless installed | Add pyannote or another diarization backend if speaker ownership becomes important | High | Extra model, RAM/VRAM, latency, access token and speaker-assignment errors |
| Synthetic tests cannot establish production accuracy | Build a consented, labeled noisy/accented multilingual audio benchmark with exact audio-mask scoring | Very High | Data collection and annotation effort; privacy governance required |

## Optimization safety rule

Future optimization should continue to obey the v4.12 rule: never gain speed by lowering PII thresholds, disabling validators/NER/semantic vetoes, shrinking audio mask padding, using a smaller ASR model than the selected user profile, or skipping targeted recovery. Optimize shared computation, I/O, batching, caching and auxiliary analysis first.
