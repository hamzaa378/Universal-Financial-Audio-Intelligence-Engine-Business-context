# Devsoc AI v4.4 — GPU-First Hybrid Financial Audio Intelligence

## What changed

### Strict GPU ASR
- `run_frontend.bat` forces Faster-Whisper to `cuda` + `float16`.
- `FINAI_STRICT_GPU=1` prevents silent CPU fallback.
- CUDA/cuBLAS/cuDNN visibility and CTranslate2 GPU access are checked before Streamlit starts.
- `Warm up AI models` performs a real tiny ASR inference so inference-time CUDA/cuDNN failures are caught early.
- The already-decoded 16 kHz waveform is passed directly to Faster-Whisper, avoiding duplicate audio decoding.

### Faster semantic AI
- All missing prototype embeddings are generated in batches.
- Prototype vectors are cached across Streamlit reruns.
- Intent + abusive-tone classification share one batched semantic pass.
- Ambiguous PII candidates share one batched privacy-judge pass.
- Optional `fastembed-gpu` support uses `CUDAExecutionProvider` when available.

### PII/privacy improvements
Existing: email, phone, PAN, IFSC, UPI, card, Aadhaar, account number, OTP, CVV, pincode, DOB, name, address.

Added context-gated:
- Passport
- Voter ID / EPIC
- Driving licence/license

False-positive controls remain local-context based so order/case/reference numbers are not borrowed into nearby phone labels.

### Sensitive financial IDs separated from classic PII
A new independent layer can hide:
- Transaction IDs/references
- Loan IDs
- Customer IDs
- Application IDs
- Complaint/grievance/case IDs

This avoids corrupting PII precision/FPR measurements while still protecting financially sensitive identifiers.

### Profanity improvements
- More English and Hinglish variants.
- Controlled leetspeak/censor detection.
- Spaced-letter forms.
- Word-boundary logic prevents substring false positives such as `assistant`, `asset`, and `class`.

### Financial understanding improvements
- EMI amount
- Outstanding amount
- Due amount
- Loan amount
- Settlement amount
- Late fee / bounce charge
- Interest rate / APR
- Tenure
- Payment/due dates
- Payment method
- Payment status
- Promise-to-pay with date/amount
- Repayment difficulty
- Payment refusal
- Debt/payment dispute
- Expanded compliance-risk phrase screening

### Efficiency improvements
- Shared acoustic feature pass for call quality + stress markers.
- Cached pyannote pipeline when diarization is enabled.
- Audio masking can reuse the already-loaded waveform.
- UI now reports ASR time, text-AI time, total time and real-time factor (RTF).

## Recommended RTX 4060 demo settings

- Detection profile: **Balanced hybrid**
- Semantic AI: **ON** after warm-up
- Full PII redaction: **ON**
- Sensitive financial IDs: **ON**
- Whisper: **small** for fastest demo; **medium** when accuracy is more important
- ASR mode: **Fast demo** first, then **Balanced**
- Protected audio: **Bleep tone**
- Speaker diarization: OFF unless pyannote dependencies and HF token are configured

## Start

```bat
install_v4.bat
run_frontend.bat
```

Then click **Warm up AI models** once before processing the first call.

## GPU checks

```bat
diagnose_gpu.bat
gpu_smoke_test.bat
```

Normal `run_frontend.bat` is strict GPU. For intentional troubleshooting only:

```bat
run_frontend_cpu.bat
```

## Benchmark

```bat
run_privacy_benchmark.bat
```

For a folder of audio files:

```bat
run_privacy_benchmark.bat "C:\path\to\audio" small
```

Results are written to `reports\`.

Synthetic regression results are regression checks only and must not be presented as real-world production accuracy.
