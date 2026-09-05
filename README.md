# Universal Financial Audio Intelligence Engine — AI v3 Privacy/Profanity Upgrade

AI v3 keeps the confidence-aware financial ASR pipeline from v2 and adds a substantially stronger privacy/safety layer plus automated regression benchmarking.

## Major v3 changes
- Context-aware PII detector designed to reduce false positives from generic numbers.
- Deterministic validators for structured identifiers:
  - Luhn validation for payment cards.
  - Verhoeff validation where applicable for Aadhaar-like values.
  - Exact PAN and IFSC structures.
  - Email/UPI disambiguation.
- Context-only detection for ambiguous values such as OTP, CVV, account numbers, PIN codes and DOB.
- Context-aware NAME and ADDRESS detection to avoid masking explanations such as `Address is required for KYC`.
- ASR-aware spoken PII handling, e.g. `hamza dot ahmad at gmail dot com` and spoken digit sequences.
- Conflict resolution so the same character span is not classified as several PII types.
- Safe metadata output: normal results do not store raw PII values in the PII metadata list.
- Privacy-safe pipeline output by default. Use `--include-raw` only for controlled debugging.
- Profanity detection with whole-word/obfuscation handling, including common English and Hinglish forms, while avoiding substring false positives such as `assistant`, `asset`, or `class`.
- Profanity replacement with `[BLEEP]` in the safe transcript.
- Windows batch benchmark for functionality, precision/recall/F1, false positives, false negatives and detector throughput.
- Optional full-audio benchmark for ASR/pipeline real-time factor.

## Setup on Windows
```powershell
cd Devsoc_AI_v3
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-core.txt
```

Core mode does **not** require torchvision or torchaudio.

## Run one call
```powershell
python app.py sample_call.wav --model small --output result.json
```

For higher ASR accuracy on an RTX 4060, test:
```powershell
python app.py sample_call.wav --model medium --output result.json
```

Normal output is privacy-safe. Raw transcript output is intentionally opt-in:
```powershell
python app.py sample_call.wav --include-raw
```
Do not use `--include-raw` for normal UI/logging.

## Run the automated privacy/profanity benchmark
Double-click or run:
```bat
run_privacy_benchmark.bat
```

It performs:
1. Functional unit tests.
2. PII detection/masking regression tests.
3. Profanity detection/reduction regression tests.
4. Precision, recall, F1, false-discovery rate, false-negative rate and negative-case false-positive rate.
5. Confidence-threshold sweep.
6. Text privacy-layer throughput and latency.

Reports are written to:
```text
reports/privacy_benchmark.json
reports/privacy_cases.csv
```

### Optional full audio/ASR efficiency benchmark
Put consented/synthetic test calls in a folder and run:
```bat
run_privacy_benchmark.bat "C:\path\to\test_audio" medium
```

This adds:
```text
reports/audio_benchmark.json
reports/audio_benchmark.csv
```
with audio duration, processing time, real-time factor, ASR confidence, PII/profanity counts and overall trust.

## Benchmark interpretation
The included benchmark corpus is **synthetic regression data**, not evidence of production accuracy. A 100% score on these cases means the code passed the cases packaged with this repository. For hackathon validation, add unseen synthetic calls and then a consented/labeled hold-out set of real financial-call audio.

Recommended real evaluation metrics:
- ASR: WER and CER.
- Financial entity extraction: precision/recall/F1.
- PII: entity precision/recall/F1 + negative-case false-positive rate.
- Profanity: precision/recall/F1 by language and obfuscation type.
- Diarization: DER.
- Full pipeline: processing time and real-time factor.

## Threshold tuning
`detect_pii(text, min_confidence=...)` supports high-precision tuning. The benchmark writes a threshold sweep for 0.80, 0.90, 0.93, 0.95 and 0.97. Raising the threshold generally reduces uncertain detections but can reduce recall.

## Optional AI intent model
Install `transformers` and `sentencepiece`, then set:
```powershell
$env:FINAI_ZERO_SHOT="1"
```
For a final product, fine-tune a multilingual financial intent model on a labeled corpus instead of relying only on zero-shot classification.

## Optional diarization
Install `pyannote.audio`, accept the relevant Hugging Face model terms, configure `HF_TOKEN`, then:
```powershell
$env:FINAI_DIARIZATION="1"
```

## Important limitations
- Tamper/replay and stress outputs are screening signals, not definitive forensic or psychological findings.
- Spoken-number reconstruction currently targets digit-by-digit ASR output; expressions such as `ninety-eight lakh...` need a more general spoken-number normalizer.
- NAME/ADDRESS detection deliberately favors precision and explicit context; unlabeled free-form names/addresses may be missed.
- Profanity vocabulary should be governed by an organisation-approved policy lexicon before deployment.
- The bundled benchmark is synthetic and must not be presented as real-world accuracy.
